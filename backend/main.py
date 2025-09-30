import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from pydantic import BaseModel

import json
from functools import lru_cache
from backend.text_fragments import build_text_fragment_url, choose_snippet, is_synthetic_label
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from fastapi.staticfiles import StaticFiles
import os

from backend.hierarchy import compute_tier

import re
from urllib.parse import urlparse

import yaml

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

CFG = {}  # global config

def load_retrieval_cfg():
    """Load retrieval behavior from YAML so nothing is hard-coded in code."""
    global CFG
    cfg_path = Path(__file__).resolve().parent / "config" / "retrieval.yaml"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            CFG = yaml.safe_load(f) or {}
    else:
        CFG = {}
    # defaults if keys missing
    CFG.setdefault("policy_terms", [])
    CFG.setdefault("tier_boosts", {1:1.35, 2:1.10, 3:1.0, 4:1.0})
    CFG.setdefault("intent", {"course_keywords":[], "degree_keywords":[], "course_code_regex": r"\b[A-Z]{3,5}\s?\d{3}\b"})
    CFG.setdefault("nudges", {"policy_acadreg_url": 1.15})
    CFG.setdefault("guarantees", {"ensure_tier1_on_policy": True})
    CFG.setdefault("tier4_gate", {"use_embedding": True, "min_title_sim": 0.42, "min_alt_sim": 0.38})


# FastAPI backend app
app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/debug/tier-counts")
def tier_counts():
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for m in chunk_meta:
        t = (m or {}).get("tier")
        if t in counts:
            counts[t] += 1
    return {"tier1": counts[1], "tier2": counts[2], "tier3": counts[3], "tier4": counts[4], "total": len(chunk_meta)}


load_retrieval_cfg()

# Allow CORS for frontend
PUBLIC_URL = os.getenv("PUBLIC_URL", "http://localhost:8003/")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[PUBLIC_URL],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# for chunking text
chunks_embeddings = None
chunk_texts = []
chunk_sources = []

chunk_meta = [] 



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
    """Load a JSON/NDJSON file into global embeddings/texts/sources/meta, defensively."""
    global chunks_embeddings, chunk_texts, chunk_sources, chunk_meta

    def _iter_pages(obj):
        """
        Yield tuples: (page_title, page_url, sections_list)
        Supports:
          - List[dict]  (each dict is a page)
          - Dict with 'pages' key
          - NDJSON lines
        Skips any non-dict items gracefully.
        """
        # Case A: list at top level
        if isinstance(obj, list):
            for rec in obj:
                if not isinstance(rec, dict):
                    continue  # skip strings or other types
                yield rec.get("page_title", "") or rec.get("title", ""), rec.get("page_url", ""), rec.get("sections", [])
            return

        # Case B: dict at top level
        if isinstance(obj, dict):
            # 1) pages array
            if isinstance(obj.get("pages"), list):
                for rec in obj["pages"]:
                    if not isinstance(rec, dict):
                        continue
                    yield rec.get("page_title", "") or rec.get("title", ""), rec.get("page_url", ""), rec.get("sections", [])
                return
            # 2) single page dict (has sections)
            if "sections" in obj:
                yield obj.get("page_title", "") or obj.get("title", ""), obj.get("page_url", ""), obj.get("sections", [])
                return
            # 3) unknown dict shape -> nothing
            return

        # Case C: NDJSON (newline-delimited) fallback
        # If we got a string here, try line-by-line parsing
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(rec, dict):
                        yield rec.get("page_title", "") or rec.get("title", ""), rec.get("page_url", ""), rec.get("sections", [])
        except Exception:
            return

    # ---- read the file once ----
    with open(path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            # If not valid JSON as a whole, fall back to NDJSON iteration
            data = None

    new_texts, new_sources, new_meta = [], [], []

    def add_piece(text_str: str, title_str: str, url_str: str):
        if not text_str:
            return
        new_texts.append(text_str)
        new_sources.append({"title": title_str, "url": url_str})
        new_meta.append(compute_tier(url_str, title_str))

    # Iterate pages robustly
    if data is not None:
        page_iter = _iter_pages(data)
    else:
        page_iter = _iter_pages("")  # triggers NDJSON branch

    page_count = 0
    for page_title, page_url, sections in page_iter:
        page_count += 1
        if not isinstance(sections, list):
            # Some scrapers store a single dict instead of a list
            sections = [sections] if isinstance(sections, dict) else []

        for sub in sections:
            if not isinstance(sub, dict):
                continue
            sec_title = sub.get("title", "")
            full_title = f"{page_title} – {sec_title}" if sec_title else (page_title or sec_title)
            sec_url = sub.get("page_url", "") or page_url

            # paragraphs
            for p in sub.get("text", []) or []:
                if isinstance(p, str):
                    add_piece(p, full_title, sec_url)

            # bullet lists
            for li in sub.get("lists", []) or []:
                if isinstance(li, list):
                    for item in li:
                        if isinstance(item, str):
                            add_piece(item, full_title, sec_url)

            # links
            for link in sub.get("links", []) or []:
                if isinstance(link, dict):
                    label = link.get("label")
                    link_url = link.get("url")
                    if label and link_url:
                        add_piece(f"Courses: {label}", label, link_url)

    if new_texts:
        new_embeds = embed_model.encode(new_texts, convert_to_numpy=True)
        if chunks_embeddings is None:
            chunks_embeddings = new_embeds
        else:
            chunks_embeddings = np.vstack([chunks_embeddings, new_embeds])

        chunk_texts.extend(new_texts)
        chunk_sources.extend(new_sources)
        chunk_meta.extend(new_meta)

    # 🔍 Debug summary
    print(f"[loader] Pages parsed: {page_count}")
    print(f"[loader] New chunks: {len(new_texts)}  |  Total chunks: {len(chunk_texts)}")
    if new_meta:
        print(f"[loader] Example meta: {new_meta[-1]}")


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


def _program_intent(query: str) -> bool:
    q = (query or "")
    ql = q.lower()
    intent = CFG.get("intent", {})
    course_kw = intent.get("course_keywords", [])
    degree_kw = intent.get("degree_keywords", [])
    code_rx = intent.get("course_code_regex", r"\b[A-Z]{3,5}\s?\d{3}\b")
    course_code = re.search(code_rx, q)
    return any(k in ql for k in (course_kw + degree_kw)) or bool(course_code)


def _tier_boost(tier: int) -> float:
    return float(CFG.get("tier_boosts", {}).get(tier, 1.0))


POLICY_TERMS = tuple(CFG.get("policy_terms", []))


def _is_acad_reg_url(url: str) -> bool:
    return isinstance(url, str) and "/graduate/academic-regulations-degree-requirements/" in url


def _title_for_sim(src: dict) -> str:

    # Build a compact string for similarity: page title + last two URL path segments.
    # This helps match things like 'MS Information Technology' even if the title varies.
    
    title = (src.get("title") or "").strip()
    url = (src.get("url") or "")
    path = urlparse(url).path if url else ""
    segs = [s for s in path.split("/") if s]
    tail = " ".join(segs[-2:]) if segs else ""
    return (title + " " + tail).strip()


def _tier4_is_relevant_embed(query: str, idx: int) -> bool:
    
    # Embedding-based Tier-4 gate: allow a program page only if the query is 
    # semantically similar to the page's title/URL (threshold from YAML config).
    
    gate = CFG.get("tier4_gate", {})
    if not gate.get("use_embedding", True):
        return True  # if disabled in config, allow all Tier-4

    src = chunk_sources[idx] if idx < len(chunk_sources) else {}
    cand = _title_for_sim(src)
    if not cand:
        return False

    q_vec, c_vec = embed_model.encode([query, cand], convert_to_numpy=True)
    sim = float(np.dot(q_vec, c_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(c_vec) + 1e-8))

    thresh = float(gate.get("min_title_sim", 0.42))  # tune in backend/config/retrieval.yaml
    return sim >= thresh


def _search_chunks(query: str, topn: int = 40, k: int = 5):
    """
    Hierarchy-aware retrieval.

    Inputs:
      query: user question
      topn: number of initial candidates by raw embedding similarity
      k: number of final chunks to return

    Policy (from config):
      - Tiers 3 & 4 are blocked unless _program_intent(query) is True.
      - Tier boosts favor Tier 1 > Tier 2 > Tier 3/4.
      - For policy-looking queries (CFG.policy_terms), nudge Academic Regulations URLs.
      - If the filter removes all candidates, fall back to allowed tiers (never return empty).
      - If enabled (CFG.guarantees.ensure_tier1_on_policy), guarantee at least one Tier-1 on policy queries.
      - Tier-4 candidates must be semantically relevant to the named program (embedding threshold in CFG.tier4_gate).

    Returns:
      final_indices: list[int] of length ≤ k
      retrieval_path: list[dict] for transparency/testing:
        {rank, idx, score, title, url, tier, tier_name}
    """

    if chunks_embeddings is None or not chunk_texts:
        return [], []

    # 1) Encode query and compute cosine similarities
    q_vec = embed_model.encode([query], convert_to_numpy=True)[0]
    denom = (np.linalg.norm(chunks_embeddings, axis=1) * np.linalg.norm(q_vec)) + 1e-8
    sims = (chunks_embeddings @ q_vec) / denom

    # 2) Take Top-N by raw similarity
    cand_idxs = np.argsort(-sims)[:topn].tolist()

    # 3) Intent + policy flags
    q_lower = (query or "").lower()
    allow_program = _program_intent(query)  # unlocks tiers 3 & 4
    looks_policy = any(term in q_lower for term in POLICY_TERMS)
    

    # 4) Filter: drop tiers 3 & 4 unless allowed; for Tier-4 also require relevance
    filtered = []
    for i in cand_idxs:
        meta_i = chunk_meta[i] if i < len(chunk_meta) else {}
        tier = meta_i.get("tier", 2)

        # Rule: Tier 3 & 4 only when program/course intent is explicit
        if (tier == 3 or tier == 4) and not allow_program:
            continue

        # Extra rule for Tier 4: must be semantically similar to the named program (embedding-based)
        if tier == 4 and allow_program:
            if not _tier4_is_relevant_embed(query, i):
                continue

        filtered.append(i)

    # 4b) Safe fallback if we filtered everything out
    if not filtered:
        allowed_tiers = {1, 2} if not allow_program else {1, 2, 3, 4}
        filtered = [
            i for i in range(len(chunk_meta))
            if ((chunk_meta[i] or {}).get("tier") in allowed_tiers)
        ]
        if not filtered:  # extreme edge case
            filtered = list(range(len(chunk_meta)))

    # 5) Rescore with tier boosts (+ optional policy URL nudge)
    policy_nudge = float(CFG.get("nudges", {}).get("policy_acadreg_url", 1.15))
    rescored = []
    for i in filtered:
        meta_i = chunk_meta[i] if i < len(chunk_meta) else {}
        src_i  = chunk_sources[i] if i < len(chunk_sources) else {}
        tier = meta_i.get("tier", 2)

        base = float(sims[i]) * _tier_boost(tier)  # e.g., 1.35 for Tier 1, 1.10 for Tier 2
        nudge = 1.0
        if looks_policy and _is_acad_reg_url(src_i.get("url", "")):
            nudge = policy_nudge

        rescored.append((i, base * nudge))

    rescored.sort(key=lambda x: x[1], reverse=True)
    ordered = [i for i, _ in rescored]

    # 6) For policy questions, ensure at least one Tier-1 (global injection if needed & enabled)
    final: list[int] = []
    if looks_policy and bool(CFG.get("guarantees", {}).get("ensure_tier1_on_policy", True)):
        has_tier1 = any((chunk_meta[i] or {}).get("tier") == 1 for i in ordered[:k])
        if not has_tier1:
            best_t1_idx = -1
            best_t1_score = -1.0
            for i in range(len(chunk_meta)):
                meta_i = chunk_meta[i] or {}
                if meta_i.get("tier") == 1:
                    sc = float(sims[i]) * _tier_boost(1) * policy_nudge
                    if sc > best_t1_score:
                        best_t1_score = sc
                        best_t1_idx = i
            if best_t1_idx != -1:
                final.append(best_t1_idx)

    # 7) Fill remaining slots from the ordered list
    for i in ordered:
        if len(final) >= k:
            break
        if i not in final:
            final.append(i)

    final = final[:k]

    # 8) Build retrieval_path for transparency/testing
    retrieval_path = []
    for rank, i in enumerate(final, start=1):
        src = chunk_sources[i] if i < len(chunk_sources) else {}
        meta = chunk_meta[i] if i < len(chunk_meta) else {}
        retrieval_path.append({
            "rank": rank,
            "idx": i,
            "score": round(float(sims[i]), 6),
            "title": src.get("title"),
            "url": src.get("url"),
            "tier": meta.get("tier"),
            "tier_name": meta.get("tier_name"),
        })

    return final, retrieval_path

def get_top_chunks_policy(question: str, top_k: int = 5):
    """
    Hierarchy-aware retrieval that returns the SAME shape as get_top_chunks:
    list of (chunk_text, chunk_source) tuples.
    """
    idxs, _ = _search_chunks(question, topn=40, k=top_k)
    return [(chunk_texts[i], chunk_sources[i]) for i in idxs]

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
    top_chunks = get_top_chunks_policy(question, top_k=5)  # hierarchy-aware
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

def cached_answer_with_path(message: str):
    # 1) Use your existing cache for answer+sources
    answer, sources = cached_answer_tuple(message)

    # 2) Compute retrieval path from the same query
    _, retrieval_path = _search_chunks(message, topn=40, k=5)

    return answer, sources, retrieval_path

# FastAPI request model
class ChatRequest(BaseModel):
    message: str
    history: list[str] = None

# models.py or near your FastAPI code
from pydantic import BaseModel
from typing import List, Dict, Any

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    retrieval_path: List[Dict[str, Any]]  # NEW: [{rank, idx, score, title, url, tier, tier_name}, ...]

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

    answer, sources, retrieval_path = cached_answer_with_path(message)

    log_chat_to_csv(message, answer, sources)
    return ChatResponse(answer=answer, sources=sources, retrieval_path=retrieval_path)

# Mount static files at root after all API routes
frontend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../frontend/out'))
if os.path.isdir(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
    print("Mounted frontend from:", frontend_path)

# Load scraped JSONs from the scraper/ folder
filenames = [
    "unh_catalog.json",
]
for name in filenames:
    path = DATA_DIR / name
    if path.exists():
        load_json_file(str(path))
    else:
        print(f"WARNING: {path} not found, skipping.")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8003, reload=True)