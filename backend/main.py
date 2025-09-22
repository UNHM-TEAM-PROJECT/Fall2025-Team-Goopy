import os
import json
import numpy as np
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional, List

from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pydantic import BaseModel, Field

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

# -----------------------------
# FastAPI backend app
# -----------------------------
app = FastAPI()

PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:8003/t3/")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[PUBLIC_URL],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Retrieval stores
# -----------------------------
chunks_embeddings: Optional[np.ndarray] = None
chunk_texts: List[str] = []
chunk_sources: List[dict] = []
chunk_ids: List[str] = []  # stable IDs for retrieval metrics (e.g., "degree-requirements#12")

# -----------------------------
# Models
# -----------------------------
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=-1,  # CPU
)

# -----------------------------
# Helpers
# -----------------------------
def _derive_doc_id_from_json(data: dict, path: str) -> str:
    """Use URL slug when available; otherwise the filename stem."""
    url = (data or {}).get("url", "")
    if url:
        try:
            p = urlparse(url)
            last = Path(p.path).name or (Path(p.path).parts[-1] if p.path else "")
            if last:
                return last
        except Exception:
            pass
    return Path(path).stem

def load_json_file(path: str):
    """Load a scraped JSON file into global embeddings/texts/sources/ids."""
    global chunks_embeddings, chunk_texts, chunk_sources, chunk_ids

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    doc_id = _derive_doc_id_from_json(data, path)

    new_texts: List[str] = []
    new_sources: List[dict] = []
    new_ids: List[str] = []
    local_idx = 0

    def recurse_sections(sections, parent_title=""):
        nonlocal local_idx
        for sec in sections:
            title = sec.get("title", "")
            full_title = f"{parent_title} > {title}" if parent_title else title

            # plain text
            for t in sec.get("text", []):
                t = (t or "").strip()
                if not t:
                    continue
                new_texts.append(t)
                new_sources.append({"title": full_title, "url": data.get("url", "")})
                new_ids.append(f"{doc_id}#{local_idx}")
                local_idx += 1

            # links (label + URL)
            for link in sec.get("links", []):
                label = (link or {}).get("label")
                url = (link or {}).get("url")
                if label and url:
                    new_texts.append(f"Courses: {label}")
                    new_sources.append({"title": label, "url": url})
                    new_ids.append(f"{doc_id}#{local_idx}")
                    local_idx += 1

            if "subsections" in sec:
                recurse_sections(sec["subsections"], parent_title=full_title)

    recurse_sections(data.get("sections", []))

    if not new_texts:
        print(f"WARNING: no text found in {path}")
        return

    new_embeds = embed_model.encode(new_texts, convert_to_numpy=True)
    if chunks_embeddings is None:
        chunks_embeddings = new_embeds
    else:
        chunks_embeddings = np.vstack([chunks_embeddings, new_embeds])

    chunk_texts.extend(new_texts)
    chunk_sources.extend(new_sources)
    chunk_ids.extend(new_ids)

    print(f"Loaded {len(new_texts)} chunks from {path} as doc '{doc_id}'")

def get_top_chunks(question: str, top_k: int = 3):
    """Return top_k chunks as dicts with id/text/source/score."""
    if chunks_embeddings is None or len(chunks_embeddings) == 0:
        return []
    qv = embed_model.encode([question], convert_to_numpy=True)[0]
    denom = (np.linalg.norm(chunks_embeddings, axis=1) * np.linalg.norm(qv) + 1e-10)
    scores = np.dot(chunks_embeddings, qv) / denom
    idxs = scores.argsort()[-top_k:][::-1]
    return [
        {
            "id": chunk_ids[i],
            "text": chunk_texts[i],
            "source": chunk_sources[i],
            "score": float(scores[i]),
        }
        for i in idxs
    ]

def _answer_question(question: str):
    top_chunks = get_top_chunks(question, top_k=3)
    context = " ".join([c["text"] for c in top_chunks])

    prompt = (
        "Answer the question ONLY using the provided context. "
        "If the answer cannot be found, say you don't know. "
        "If the context does not mention the degree, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    try:
        result = qa_pipeline(prompt, max_new_tokens=128)
        answer = result[0]["generated_text"].strip()

        seen = set()
        sources = []
        for c in top_chunks:
            src = c["source"]
            key = (src["title"], src.get("url"))
            if key not in seen:
                seen.add(key)
                sources.append({"title": src["title"], "url": src.get("url", "")})

        retrieved_ids = [c["id"] for c in top_chunks]
        return answer, sources, retrieved_ids
    except Exception as e:
        return f"ERROR running local model: {e}", [], []

@lru_cache(maxsize=128)
def cached_answer_tuple(question_str: str):
    return _answer_question(question_str)


# Schemas

class ChatRequest(BaseModel):
    message: str
    history: Optional[List[str]] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[dict] = Field(default_factory=list)
    retrieved_ids: List[str] = Field(default_factory=list)


# API

@app.post("/t3/chat", response_model=ChatResponse)
async def answer_question(req: ChatRequest):
    message = req.message if not isinstance(req.message, list) else " ".join(req.message)
    answer, sources, retrieved_ids = cached_answer_tuple(message)
    return ChatResponse(answer=answer, sources=sources, retrieved_ids=retrieved_ids)


# Static frontend (optional)
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../frontend/out'))
if os.path.isdir(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
    print("Mounted frontend from:", frontend_path)

# Preload all catalog pages

CATALOG_FILES = [
    "academic_standards.json",   # Academic Standards
    "degree_requirements.json",  # Degree Requirements
    "graduation_grading.json",   # Grading
    "graduation.json",           # Graduation
    "course_descriptions.json",  # (optional helper set)
]

def _candidate_paths(fname: str) -> List[Path]:
    here = Path(__file__).parent
    return [
        here / ".." / "scrape" / fname,
        here / "scrape" / fname,
        here / ".." / fname,
        here / fname,
    ]

def preload_content():
    loaded: set[str] = set()
    for name in CATALOG_FILES:
        for p in _candidate_paths(name):
            p = p.resolve()
            if p.exists() and p.suffix == ".json" and str(p) not in loaded:
                try:
                    load_json_file(str(p))
                    loaded.add(str(p))
                    break  # stop after first found location for this name
                except Exception as e:
                    print(f"ERROR loading {p}: {e}")

    # Optional: load any other JSONs in ../scrape/ not already loaded
    scrape_dir = (Path(__file__).parent / ".." / "scrape").resolve()
    if scrape_dir.is_dir():
        for p in scrape_dir.glob("*.json"):
            rp = str(p.resolve())
            if rp not in loaded:
                try:
                    load_json_file(rp)
                    loaded.add(rp)
                except Exception as e:
                    print(f"ERROR loading {rp}: {e}")

preload_content()


# Run server

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=True)
