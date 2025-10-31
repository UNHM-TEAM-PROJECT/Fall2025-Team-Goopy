import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
from models.ml_models import get_qa_pipeline
from services.chunk_service import build_context_from_indices
from services.retrieval_service import search_chunks
from services.beam_search import generate_with_beam_search
from text_fragments import build_text_fragment_url, choose_snippet, is_synthetic_label
from utils.course_utils import extract_course_fallbacks
from config.settings import get_config

UNKNOWN = "I don't have that information."

def _wrap_sources_with_text_fragments(
    sources_with_passages: List[Tuple[str, Dict]],
    question: str
) -> List[Dict]:
    wrapped = []
    for passage, src in sources_with_passages:
        url = src.get("url", "")
        if not url or is_synthetic_label(passage):
            wrapped.append({**src, "url": url})
            continue
        
        snippet = choose_snippet(passage, hint=question, max_chars=160)
        wrapped.append({
            **src,
            "url": build_text_fragment_url(url, text=snippet) if snippet else url
        })
    return wrapped

def _extract_best_credits(
    chunks: List[Tuple[str, Dict]]
) -> Optional[Tuple[str, Dict, str]]:
    credit_rx = re.compile(
        r"(?:(?:minimum|at least|a total(?: of)?|total(?: of)?)\s+)?(\d{1,3})\s*(?:credit|credits|cr)\b",
        re.IGNORECASE,
    )
    
    best: Optional[Tuple[str, Dict, str, int]] = None
    
    for text, src in chunks:
        for m in credit_rx.finditer(text or ""):
            num = m.group(1)
            span_text = text[max(0, m.start()-60): m.end()+60]
            
            weight = 1
            if re.search(r"\bminimum\b|\brequired\b|\btotal\b", span_text, re.I):
                weight += 2
            
            try:
                n = int(num)
                if 6 <= n <= 90:
                    weight += 1
            except Exception:
                pass
            
            ans = f"{num}"
            cand = (text, src, ans, weight)
            
            if best is None or cand[3] > best[3]:
                best = cand
    
    if best:
        return (best[0], best[1], best[2])
    return None

def _extract_gre_requirement(
    question: str,
    chunks: List[Tuple[str, Dict]]
) -> Optional[Tuple[str, Dict, str]]:
    qn = question.lower()
    if not any(tok in qn for tok in ["gre", "g r e", "gmat", "g m a t", "test score", "test scores"]):
        return None
    
    for text, src in chunks:
        t = (text or "").lower()
        if "gre" in t or "gmat" in t or "test score" in t or "test scores" in t:
            if re.search(r"\bnot required\b|\bno gre\b|\bwaived\b|\bno (?:gmat|gre) required\b", t):
                return (text, src, "No")
            if re.search(r"\brequired\b|\bmust submit\b|\bofficial scores\b", t):
                return (text, src, "Yes")
    
    return None

def get_prompt(question: str, context: str) -> str:
    return (
        "Using ONLY the provided context, write a concise explanation in exactly 2–3 complete sentences.\n"
        "Mention requirements, deadlines, or procedures if they are present.\n"
        f"If the context is insufficient, output exactly: {UNKNOWN}\n"
        "Do not include assumptions, examples, or general knowledge beyond the context.\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        f"Detailed explanation:"
    )

def _answer_question(question: str,) -> Tuple[str, List[str], List[Dict]]:
    qa_pipeline = get_qa_pipeline()
    cfg = get_config()

    topn_cfg = cfg.get("search", {})
    topn_local = int(topn_cfg.get("topn_default", 40))
    # honor YAML k
    k_local = int(cfg.get("retrieval_sizes", {}).get("k", cfg.get("k", 5)))
    idxs, retrieval_path = search_chunks(question, topn=topn_local, k=k_local)
    top_chunks, context = build_context_from_indices(idxs)

    if not idxs:
        return "I couldn't find relevant information in the catalog.", [], retrieval_path, None
    
    # generate answer using beam search with fallback
    qa_pipeline = get_qa_pipeline()
    cfg = get_config()
    beam_cfg = cfg.get("beam_search", {})
    use_beam = beam_cfg.get("enabled", True)
    
    if use_beam:
        answer = generate_with_beam_search(qa_pipeline, get_prompt(question, context), question, context)
        if answer is None:
            # Beam search failed, use fallback
            result = qa_pipeline(
                get_prompt(question, context),
                max_new_tokens=128,
                repetition_penalty=1.2,
                no_repeat_ngram_size=2,
            )
            answer = result[0]["generated_text"].strip()
    else:
        # Beam search disabled, use single generation
        result = qa_pipeline(
            get_prompt(question, context),
            max_new_tokens=128,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
        )
        answer = result[0]["generated_text"].strip()
    # apply fallbacks
    def _looks_idk(a: str) -> bool:
        return (a or "").strip() == UNKNOWN

    enriched_sources = _wrap_sources_with_text_fragments(top_chunks, question)

    # Simple fallbacks without intent detection
    # degree credits fallback
    qn_lower = (question or "").lower()
    if any(tok in qn_lower for tok in ["credits required", "how many credits", "total credits", "credit requirement"]):
        hit = _extract_best_credits(top_chunks)
        if hit:
            _, src, num = hit
            if _looks_idk(answer) or not re.search(r"\b\d{1,3}\b", answer):
                answer = f"{num}."
    
    # course fallbacks (simple pattern matching)
    if re.search(r"\b[A-Z]{2,4}\s*\d{3,4}\b", question):
        cf = extract_course_fallbacks(top_chunks)
        need_help = _looks_idk(answer) or \
                   (not re.search(r"credits|prereq|grade", answer, re.I))
        if need_help and any(cf.values()):
            parts = []
            if cf["credits"]:
                parts.append(f"Credits: {cf['credits']}")
            if cf["prereqs"]:
                parts.append(f"Prerequisite(s): {cf['prereqs']}")
            if cf["grademode"]:
                parts.append(f"Grade Mode: {cf['grademode']}")
            if parts:
                answer = ". ".join(parts) + "."
    
    # build sources
    enriched_all = enriched_sources + [src for _, src in top_chunks[3:]]
    seen = set()
    citation_lines = []
    for src in enriched_all:
        key = (src.get("title"), src.get("url"))
        if key in seen:
            continue
        seen.add(key)
        line = f"- {src.get('title', 'Source')}"
        if src.get("url"):
            line += f" ({src['url']})"
        citation_lines.append(line)
    return answer, citation_lines, retrieval_path, context

@lru_cache(maxsize=128)
def _cached_answer_core(cache_key: str) -> Tuple[str, List[str], List[Dict]]:
    return _answer_question(cache_key)

def cached_answer_with_path(
    message: str,
    alias_url: Optional[str] = None,
    intent_key: Optional[str] = None,
    course_norm: Optional[str] = None,
) -> Tuple[str, List[str], List[Dict]]:
    # Simplified: ignore broken parameters, just cache by message
    return _cached_answer_core(message)