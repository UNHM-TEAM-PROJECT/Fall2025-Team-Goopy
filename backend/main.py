import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pydantic import BaseModel

import json
from functools import lru_cache
from text_fragments import build_text_fragment_url, choose_snippet, is_synthetic_label
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi.staticfiles import StaticFiles
import os

# --- CSV logging (added) ---
import csv
import threading
from datetime import datetime

CHAT_LOG_PATH = "chat_logs.csv"
_LOG_LOCK = threading.Lock()

# create file with header if not exists
if not os.path.isfile(CHAT_LOG_PATH):
    with open(CHAT_LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "question", "answer", "sources_json"])
# --- end CSV logging ---

# FastAPI backend app
app = FastAPI()

# Allow CORS for frontend
PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:8003/")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[PUBLIC_URL],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

# for chunking text
chunks_embeddings = None
chunk_texts = []
chunk_sources = []

# paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "scraper"

# embeddings model (local + small)
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# load local Flan-T5 Small for queries
qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=-1,  # CPU currently better performance on GPU (0)
)

def load_json_file(path):
    """Load a JSON file into global embeddings/texts/sources."""
    global chunks_embeddings, chunk_texts, chunk_sources

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_texts = []
    new_sources = []

    def recurse_sections(sections, parent_title=""):
        for sec in sections:
            title = sec.get("title", "")
            full_title = f"{parent_title} > {title}" if parent_title else title

            # add plain text
            for t in sec.get("text", []):
                new_texts.append(t)
                new_sources.append({
                    "title": full_title,
                    "url": data.get("url", "")
                })

            # add links (label + URL)
            for link in sec.get("links", []):
                label = link.get("label")
                url = link.get("url")
                if label and url:
                    new_texts.append(f"Courses: {label}")
                    new_sources.append({
                        "title": label,
                        "url": url
                    })

            if "subsections" in sec:
                recurse_sections(sec["subsections"], parent_title=full_title)

    recurse_sections(data.get("sections", []))

    # append new chunks to existing ones
    if new_texts:
        if chunks_embeddings is None:
            chunks_embeddings = embed_model.encode(new_texts, convert_to_numpy=True)
            chunk_texts.extend(new_texts)
            chunk_sources.extend(new_sources)
        else:
            new_embeds = embed_model.encode(new_texts, convert_to_numpy=True)
            chunks_embeddings = np.vstack([chunks_embeddings, new_embeds])
            chunk_texts.extend(new_texts)
            chunk_sources.extend(new_sources)

        print(f"Loaded {len(new_texts)} chunks from {path}")
    else:
        print(f"WARNING: no text found in {path}")

# retrieval utilities
def get_top_chunks(question, top_k=3):
    if chunks_embeddings is None or len(chunks_embeddings) == 0:
        return []
    question_vec = embed_model.encode([question], convert_to_numpy=True)[0]
    scores = np.dot(chunks_embeddings, question_vec) / (
        np.linalg.norm(chunks_embeddings, axis=1) * np.linalg.norm(question_vec) + 1e-10
    )
    top_indices = scores.argsort()[-top_k:][::-1]
    return [(chunk_texts[i], chunk_sources[i]) for i in top_indices]

def _wrap_sources_with_text_fragments(sources_with_passages, question: str):
    """
    Input: list of tuples (passage_text, source_dict)
    Output: list of dicts like source_dict but with url replaced by a text-fragment URL
    """
    wrapped = []
    for passage, src in sources_with_passages:
        url = src.get("url", "")
        # Some chunks are synthetic labels (e.g., "Courses: …"); don't build fragments for those
        if not url or is_synthetic_label(passage):
            wrapped.append({**src, "url": url})
            continue

        # Choose a compact snippet (tries to align to the question)
        snippet = choose_snippet(passage, hint=question, max_chars=160)

        if snippet:
            frag_url = build_text_fragment_url(url, text=snippet)
            wrapped.append({**src, "url": frag_url})
        else:
            wrapped.append({**src, "url": url})
    return wrapped

# core answer function (not cached directly)
def _answer_question(question):
    top_chunks = get_top_chunks(question)
    context = " ".join([text for text, _ in top_chunks])

    prompt = (
        "Answer the question ONLY using the provided context. "
        "If the answer cannot be found, say you don't know. "
        "If the context does not mention the degree, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    try:
        result = qa_pipeline(prompt, max_new_tokens=128)  # limit tokens
        answer = result[0]["generated_text"].strip()

        # Build citation links WITH text fragments, one per unique (title,url)
        enriched_sources = _wrap_sources_with_text_fragments(top_chunks, question)

        seen = set()
        citation_lines = []
        for src in enriched_sources:
            key = (src.get("title"), src.get("url"))
            if key in seen:
                continue
            seen.add(key)
            line = f"- {src.get('title','Source')}"
            if src.get("url"):
                line += f" ({src['url']})"
            citation_lines.append(line)

        return answer, citation_lines
    except Exception as e:
        return f"ERROR running local model: {e}", []

# cached wrapper for answers
@lru_cache(maxsize=128)
def cached_answer_tuple(question_str):
    return _answer_question(question_str)

# FastAPI request model
class ChatRequest(BaseModel):
    message: str
    history: list[str] = None

# FastAPI response model
class ChatResponse(BaseModel):
    answer: str
    sources: list[str] = []

# CSV logging helper
def log_chat_to_csv(question, answer, sources):
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    row = [ts, question, answer, json.dumps(sources, ensure_ascii=False)]
    with _LOG_LOCK:
        with open(CHAT_LOG_PATH, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)

# FastAPI chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def answer_question(request: ChatRequest):
    message = request.message
    if isinstance(message, list):
        message = " ".join(message)
    answer, sources = cached_answer_tuple(message)
    log_chat_to_csv(message, answer, sources)
    return ChatResponse(answer=answer, sources=sources)

# Endpoint to serve test results for dashboard
@app.get("/test-results")
async def get_test_results():
    """Serve all automation testing results from timestamped report directories"""
    try:
        automation_dir = Path(__file__).parent.parent / "automation_testing"
        reports_dir = automation_dir / "reports"
        
        if not reports_dir.exists():
            return {"error": "Reports directory not found", "message": "Run automation tests first", "test_runs": []}
        
        test_runs = []
        
        # Scan all test run directories
        for test_dir in sorted(reports_dir.iterdir(), reverse=True):  # Most recent first
            if not test_dir.is_dir():
                continue
                
            report_path = test_dir / "report.json"
            preds_path = test_dir / "preds.jsonl"
            gold_path = test_dir / "gold.jsonl"
            
            if not report_path.exists():
                continue  # Skip directories without report files
            
            try:
                # Load main report data
                with open(report_path, 'r') as f:
                    report_data = json.load(f)
                
                test_run = {
                    'run_id': test_dir.name,
                    'summary': report_data.get('summary', {}),
                    'total_questions': len(report_data.get('per_question', [])),
                    'files_present': {
                        'report': report_path.exists(),
                        'predictions': preds_path.exists(),
                        'gold': gold_path.exists()
                    }
                }
                
                # Add detailed predictions if requested
                predictions_data = None
                if preds_path.exists() and gold_path.exists():
                    try:
                        # Load predictions
                        predictions = []
                        with open(preds_path, 'r') as f:
                            for line in f:
                                predictions.append(json.loads(line.strip()))
                        
                        # Load gold standard
                        gold_data = {}
                        with open(gold_path, 'r') as f:
                            for line in f:
                                item = json.loads(line.strip())
                                gold_data[item['id']] = item
                        
                        # Combine predictions with gold standard and per-question metrics
                        combined_predictions = []
                        for pred in predictions:
                            pred_id = pred['id']
                            if pred_id in gold_data:
                                gold_item = gold_data[pred_id]
                                
                                # Find matching per-question metrics from report
                                per_question_metrics = next(
                                    (m for m in report_data.get("per_question", []) if m["id"] == pred_id), 
                                    {}
                                )
                                
                                combined_predictions.append({
                                    'id': pred_id,
                                    'category': pred_id.split('-')[0],  # AS, DR, GR, GA
                                    'query': gold_item['query'],
                                    'model_answer': pred['model_answer'],
                                    'reference_answer': gold_item['reference_answer'],
                                    'nuggets': gold_item['nuggets'],
                                    'retrieved_ids': pred['retrieved_ids'],
                                    'gold_passages': gold_item['gold_passages'],
                                    'url': gold_item['url'],
                                    'metrics': {
                                        'nugget_precision': per_question_metrics.get('nugget_precision', 0),
                                        'nugget_recall': per_question_metrics.get('nugget_recall', 0),
                                        'nugget_f1': per_question_metrics.get('nugget_f1', 0),
                                        'sbert_cosine': per_question_metrics.get('sbert_cosine', 0),
                                        'bertscore_f1': per_question_metrics.get('bertscore_f1', 0),
                                        'recall@1': per_question_metrics.get('recall@1', 0),
                                        'recall@3': per_question_metrics.get('recall@3', 0),
                                        'recall@5': per_question_metrics.get('recall@5', 0),
                                        'ndcg@1': per_question_metrics.get('ndcg@1', 0),
                                        'ndcg@3': per_question_metrics.get('ndcg@3', 0),
                                        'ndcg@5': per_question_metrics.get('ndcg@5', 0)
                                    }
                                })
                        
                        # Add predictions metadata
                        pred_stat_info = os.stat(preds_path)
                        predictions_data = {
                            'predictions': combined_predictions,
                            'total_questions': len(combined_predictions),
                            'categories': {
                                'AS': len([p for p in combined_predictions if p['category'] == 'AS']),
                                'DR': len([p for p in combined_predictions if p['category'] == 'DR']),
                                'GR': len([p for p in combined_predictions if p['category'] == 'GR']),
                                'GA': len([p for p in combined_predictions if p['category'] == 'GA'])
                            },
                            'predictions_timestamp': datetime.fromtimestamp(pred_stat_info.st_mtime).isoformat()
                        }
                    except Exception as pred_error:
                        print(f"Error loading predictions for {test_dir.name}: {pred_error}")
                        # Continue without predictions if there's an error
                
                # Add predictions data to test run
                if predictions_data:
                    test_run['predictions_data'] = predictions_data
                    
                test_runs.append(test_run)
                
            except Exception as e:
                print(f"Error loading test run {test_dir.name}: {e}")
                continue
        
        return {
            'test_runs': test_runs,
            'total_runs': len(test_runs),
            'latest_run': test_runs[0] if test_runs else None
        }
    except Exception as e:
        return {"error": "Failed to load test results", "message": str(e), "test_runs": []}

# Endpoint to get specific test run details
@app.get("/test-results/{run_id}")
async def get_test_run_details(run_id: str):
    """Get detailed results for a specific test run"""
    try:
        automation_dir = Path(__file__).parent.parent / "automation_testing"
        reports_dir = automation_dir / "reports"
        test_dir = reports_dir / run_id
        
        if not test_dir.exists() or not test_dir.is_dir():
            return {"error": "Test run not found", "message": f"No test run found with ID: {run_id}"}
        
        report_path = test_dir / "report.json"
        preds_path = test_dir / "preds.jsonl"
        gold_path = test_dir / "gold.jsonl"
        
        if not report_path.exists():
            return {"error": "Report not found", "message": f"No report.json found for test run: {run_id}"}
        
        # Load the full report data
        with open(report_path, 'r') as f:
            report_data = json.load(f)
        
        result = {
            'run_id': run_id,
            'summary': report_data.get('summary', {}),
            'per_question': report_data.get('per_question', []),
            'files_present': {
                'report': report_path.exists(),
                'predictions': preds_path.exists(),
                'gold': gold_path.exists()
            }
        }
        
        # Add detailed predictions if available
        if preds_path.exists() and gold_path.exists():
            try:
                # Load predictions
                predictions = []
                with open(preds_path, 'r') as f:
                    for line in f:
                        predictions.append(json.loads(line.strip()))
                
                # Load gold standard
                gold_data = {}
                with open(gold_path, 'r') as f:
                    for line in f:
                        item = json.loads(line.strip())
                        gold_data[item['id']] = item
                
                # Combine predictions with gold standard and per-question metrics
                combined_predictions = []
                for pred in predictions:
                    pred_id = pred['id']
                    if pred_id in gold_data:
                        gold_item = gold_data[pred_id]
                        
                        # Find matching per-question metrics from report
                        per_question_metrics = next(
                            (m for m in report_data.get("per_question", []) if m["id"] == pred_id), 
                            {}
                        )
                        
                        combined_predictions.append({
                            'id': pred_id,
                            'category': pred_id.split('-')[0],  # AS, DR, GR, GA
                            'query': gold_item['query'],
                            'model_answer': pred['model_answer'],
                            'reference_answer': gold_item['reference_answer'],
                            'nuggets': gold_item['nuggets'],
                            'retrieved_ids': pred['retrieved_ids'],
                            'gold_passages': gold_item['gold_passages'],
                            'url': gold_item['url'],
                            'metrics': {
                                'nugget_precision': per_question_metrics.get('nugget_precision', 0),
                                'nugget_recall': per_question_metrics.get('nugget_recall', 0),
                                'nugget_f1': per_question_metrics.get('nugget_f1', 0),
                                'sbert_cosine': per_question_metrics.get('sbert_cosine', 0),
                                'bertscore_f1': per_question_metrics.get('bertscore_f1', 0),
                                'recall@1': per_question_metrics.get('recall@1', 0),
                                'recall@3': per_question_metrics.get('recall@3', 0),
                                'recall@5': per_question_metrics.get('recall@5', 0),
                                'ndcg@1': per_question_metrics.get('ndcg@1', 0),
                                'ndcg@3': per_question_metrics.get('ndcg@3', 0),
                                'ndcg@5': per_question_metrics.get('ndcg@5', 0)
                            }
                        })
                
                # Add predictions metadata
                pred_stat_info = os.stat(preds_path)
                predictions_data = {
                    'predictions': combined_predictions,
                    'total_questions': len(combined_predictions),
                    'categories': {
                        'AS': len([p for p in combined_predictions if p['category'] == 'AS']),
                        'DR': len([p for p in combined_predictions if p['category'] == 'DR']),
                        'GR': len([p for p in combined_predictions if p['category'] == 'GR']),
                        'GA': len([p for p in combined_predictions if p['category'] == 'GA'])
                    },
                    'predictions_timestamp': datetime.fromtimestamp(pred_stat_info.st_mtime).isoformat()
                }
                
                result['predictions_data'] = predictions_data
                
            except Exception as pred_error:
                print(f"Error loading predictions for {run_id}: {pred_error}")
                result['predictions_error'] = str(pred_error)
        
        return result
        
    except Exception as e:
        return {"error": "Failed to load test run details", "message": str(e)}

# Endpoint to run new tests
@app.post("/run-tests")
async def run_tests():
    """Run the automation testing suite and return status"""
    try:
        import subprocess
        import sys
        
        # Path to the run_tests.py script
        automation_dir = Path(__file__).parent.parent / "automation_testing"
        run_tests_script = automation_dir / "run_tests.py"
        
        if not run_tests_script.exists():
            return {"error": "Test runner not found", "message": "run_tests.py script is missing"}
        
        # Run the test script
        result = subprocess.run(
            [sys.executable, str(run_tests_script)],
            cwd=str(automation_dir),
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            return {
                "status": "success",
                "message": "Tests completed successfully",
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        else:
            return {
                "status": "error", 
                "message": "Tests failed",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
    except subprocess.TimeoutExpired:
        return {"error": "Test execution timeout", "message": "Tests took longer than 5 minutes to complete"}
    except Exception as e:
        return {"error": "Failed to run tests", "message": str(e)}

# Serve dashboard HTML file for /dashboard route
@app.get("/dashboard")
async def serve_dashboard():
    """Serve the dashboard HTML file"""
    from fastapi.responses import FileResponse
    dashboard_path = Path(__file__).parent.parent / "frontend" / "out" / "dashboard.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path, media_type="text/html")
    else:
        return {"error": "Dashboard not found", "message": "Build the frontend first"}

# Mount static files at root after all API routes
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../frontend/out'))
if os.path.isdir(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
    print("Mounted frontend from:", frontend_path)

# Load scraped JSONs from the scraper/ folder
filenames = [
    "course_descriptions.json",
    "degree_requirements.json",
    "academic_standards.json",
    "graduation.json",
    "graduation_grading.json",
]
for name in filenames:
    path = DATA_DIR / name
    if path.exists():
        load_json_file(str(path))
    else:
        print(f"WARNING: {path} not found, skipping.")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=True)