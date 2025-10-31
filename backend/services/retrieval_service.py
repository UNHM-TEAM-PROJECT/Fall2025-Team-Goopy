import re
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse
import numpy as np
from config.settings import get_config, get_policy_terms
from models.ml_models import get_embed_model
from services.chunk_service import get_chunks_data, get_chunk_norms
from services.intent_service import is_admissions_url, is_degree_requirements_url, has_admissions_terms, has_policy_terms

def _tier_boost(tier: int) -> float:
    cfg = get_config()
    return float(cfg.get("tier_boosts", {}).get(tier, 1.0))

def _is_acad_reg_url(url: str) -> bool:
    return isinstance(url, str) and "/graduate/academic-regulations-degree-requirements/" in url

def _title_for_sim(src: Dict[str, Any]) -> str:
    title = (src.get("title") or "").strip()
    url = src.get("url") or ""
    path = urlparse(url).path if url else ""
    segs = [s for s in path.split("/") if s]
    tail = " ".join(segs[-2:]) if segs else ""
    return (title + " " + tail).strip()

def _tier4_is_relevant_embed(query: str, idx: int) -> bool:
    cfg = get_config()
    gate = cfg.get("tier4_gate", {})
    if not gate.get("use_embedding", True):
        return True
    _, _, chunk_sources, _ = get_chunks_data()
    if idx >= len(chunk_sources):
        return False
    cand = _title_for_sim(chunk_sources[idx])
    if not cand:
        return False
    embed_model = get_embed_model()
    q_vec, c_vec = embed_model.encode([query, cand], convert_to_numpy=True)
    sim = float(np.dot(q_vec, c_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(c_vec) + 1e-8))
    thresh = float(gate.get("min_title_sim", 0.42))
    return sim >= thresh

def search_chunks(
    query: str,
    topn: int = 40,
    k: int = 5
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """
    Simplified search without broken intent/program detection.
    Uses only the query text and basic tier filtering.
    """
    cfg = get_config()
    policy_terms = get_policy_terms()
    embed_model = get_embed_model()
    chunks_embeddings, chunk_texts, chunk_sources, chunk_meta = get_chunks_data()
    if chunks_embeddings is None or not chunk_texts:
        return [], []

    # encode query
    q_vec = embed_model.encode([query], convert_to_numpy=True)[0]
    chunk_norms = get_chunk_norms()
    query_norm = np.linalg.norm(q_vec)

    valid_chunks = chunk_norms > 1e-8
    sims = np.zeros(len(chunks_embeddings))

    if query_norm > 1e-8:
        sims[valid_chunks] = (
            (chunks_embeddings[valid_chunks] @ q_vec) /
            (chunk_norms[valid_chunks] * query_norm)
        )
    
    # get top candidates
    cand_idxs = np.argsort(-sims)[:topn * 2].tolist()
    
    q_lower = (query or "").lower()
    looks_policy = any(term in q_lower for term in policy_terms) or has_policy_terms(q_lower)
    looks_admissions = any(tok in q_lower for tok in [
        "admission", "admissions", "apply", "gre", "gmat", "test score", "test scores", "toefl", "ielts"
    ])
    
    # Simple course code detection (without broken course_norm param)
    course_code_pattern = re.compile(r'\b[A-Z]{2,4}\s*\d{3,4}\b')
    has_course_code = bool(course_code_pattern.search(query))

    # extract query terms
    query_terms = set(re.findall(r'\b\w+\b', q_lower))
    
    filtered: List[int] = []
    for i in cand_idxs:
        if i >= len(chunk_texts):
            continue

        chunk_text_lower = chunk_texts[i].lower()
        meta_i = chunk_meta[i] if i < len(chunk_meta) else {}
        tier = meta_i.get("tier", 2)

        # Skip program pages (Tier 4) for course queries
        if has_course_code and tier == 4:
            continue

        # Policy queries: filter Tier 3/4
        if looks_policy:
            if tier == 3:
                if not has_policy_terms(chunk_text_lower):
                    continue
            if tier == 4:
                # Skip program pages for policy queries unless they have policy terms
                if not has_policy_terms(chunk_text_lower):
                    continue
        
        # Admissions queries: filter Tier 3
        if looks_admissions:
            if tier == 3 and not has_admissions_terms(chunk_text_lower):
                continue
        
        # Basic relevance check
        term_matches = len(query_terms.intersection(set(re.findall(r'\b\w+\b', chunk_text_lower))))
        if term_matches == 0 and sims[i] < 0.1:
            continue
                
        # Tier filtering: allow most tiers but check relevance for Tier 4
        if tier == 4:
            if not _tier4_is_relevant_embed(query, i):
                continue
        
        filtered.append(i)

    # fallback if no results
    if not filtered:
        filtered = list(range(len(chunk_meta)))

    # rescore with bonuses
    policy_nudge = float(cfg.get("nudges", {}).get("policy_acadreg_url", 1.15))

    rescored = []
    for i in filtered:
        meta_i = chunk_meta[i] if i < len(chunk_meta) else {}
        src_i = chunk_sources[i] if i < len(chunk_sources) else {}
        tier = meta_i.get("tier", 2)
        base = float(sims[i]) * _tier_boost(tier)

        # policy nudge
        nudge = policy_nudge if looks_policy and _is_acad_reg_url(src_i.get("url", "")) else 1.0

        # admissions bonus
        admissions_bonus = 1.0
        if looks_admissions:
            url_i = (src_i.get("url") or "")
            txt_i = (chunk_texts[i] or "").lower()
            if is_admissions_url(url_i):
                admissions_bonus *= 1.6
            if has_admissions_terms(txt_i):
                admissions_bonus *= 1.25
            if is_degree_requirements_url(url_i):
                admissions_bonus *= 0.85

        rescored.append((i, base * nudge * admissions_bonus))

    rescored.sort(key=lambda x: x[1], reverse=True)
    ordered = [i for i, _ in rescored]

    # Prioritize Tier 1 for policy queries
    if looks_policy:
        lead_t1 = None
        for i in ordered:
            if (chunk_meta[i] or {}).get("tier") == 1:
                lead_t1 = i
                break
        if lead_t1 is not None and ordered and ordered[0] != lead_t1:
            ordered.remove(lead_t1)
            ordered.insert(0, lead_t1)

    final: List[int] = []
    if looks_policy and bool(cfg.get("guarantees", {}).get("ensure_tier1_on_policy", True)):
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

    for i in ordered:
        if len(final) >= k:
            break
        if i not in final:
            final.append(i)

    final = final[:k]

    retrieval_path = []
    for rank, i in enumerate(final, start=1):
        src = chunk_sources[i] if i < len(chunk_sources) else {}
        meta = chunk_meta[i] if i < len(chunk_meta) else {}
        retrieval_path.append(
            {
                "rank": rank,
                "idx": i,
                "score": round(float(sims[i]), 6),
                "title": src.get("title"),
                "url": src.get("url"),
                "tier": meta.get("tier"),
                "tier_name": meta.get("tier_name"),
                "text": chunk_texts[i] if i < len(chunk_texts) else ""
            }
        )
    return final, retrieval_path