import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from hierarchy import compute_tier
from models.ml_models import get_embed_model
from utils.program_utils import build_program_index
from config.settings import get_config

# paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "scraper"
CACHE_PATH = DATA_DIR / "chunks_cache.pkl"

# global data structures
chunks_embeddings: Optional[np.ndarray] = None
chunk_texts: List[str] = []
chunk_sources: List[Dict[str, Any]] = []
chunk_meta: List[Dict[str, Any]] = []
CHUNK_NORMS: Optional[np.ndarray] = None

def build_context_from_indices(idxs: List[int]) -> Tuple[List[Tuple[str, Dict[str, Any]]], str]:
    _, chunk_texts, chunk_sources, _ = get_chunks_data()
    if not idxs:
        return [], ""
    top_chunks = [(chunk_texts[i], chunk_sources[i]) for i in idxs if i < len(chunk_texts)]
    parts = []
    for text, source in top_chunks:
        title = source.get("title", "Source")
        title = title.split(" - ")[-1] if " - " in title else title
        parts.append(f"{title}: {text}")
    return top_chunks, "\n\n".join(parts)

def _compute_meta_from_source(src: Dict[str, Any]) -> Dict[str, Any]:
    return compute_tier(src.get("url", ""), src.get("title", ""))

def save_chunks_cache() -> None:
    global chunks_embeddings, chunk_texts, chunk_sources, chunk_meta
    if not chunk_texts or chunks_embeddings is None:
        return
    
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(
            {
                "texts": chunk_texts,
                "sources": chunk_sources,
                "embeddings": chunks_embeddings,
                "meta": chunk_meta,
                "norms": CHUNK_NORMS
            },
            f,
        )
    
    print(f"Saved {len(chunk_texts)} chunks to cache")


def load_chunks_cache() -> bool:
    global chunk_texts, chunk_sources, chunks_embeddings, chunk_meta, CHUNK_NORMS
    if not CACHE_PATH.exists():
        return False
    
    with open(CACHE_PATH, "rb") as f:
        data = pickle.load(f)
    
    chunk_texts = data.get("texts", [])
    chunk_sources = data.get("sources", [])
    chunks_embeddings = data.get("embeddings")
    CHUNK_NORMS = data.get("norms")
    cached_meta = data.get("meta")
    
    if cached_meta:
        chunk_meta = cached_meta
    else:
        # recompute metadata if not in cache
        chunk_meta = [_compute_meta_from_source(src) for src in chunk_sources]
    
    if chunks_embeddings is None or not chunk_texts:
        chunk_meta = []
        chunk_sources = []
        chunk_texts = []
        return False
    if CHUNK_NORMS is None and chunks_embeddings is not None:
        CHUNK_NORMS = np.linalg.norm(chunks_embeddings, axis=1)
    print(f"Loaded {len(chunk_texts)} chunks from cache.")
    return True


def load_json_file(path: str) -> None:
    global chunks_embeddings, chunk_texts, chunk_sources, chunk_meta
    
    cfg = get_config()
    embed_model = get_embed_model()
    page_count = 0
    new_texts: List[str] = []
    new_sources: List[Dict[str, Any]] = []
    new_meta: List[Dict[str, Any]] = []
    
    def add_chunk_with_context(text: str, title: str, url: str) -> None:
        """Add a chunk with contextual header prepended."""
        # Only add if at least 3 words and either contains sentence-ending punctuation or is long enough
        word_count = len(text.split())
        has_sentence_punct = any(p in text for p in ".!?")
        min_length = 30
        if word_count >= 3 and (has_sentence_punct or len(text) >= min_length):
            # Prepend contextual header to improve retrieval
            enable_headers = cfg.get("chunking", {}).get("enable_contextual_headers", True)
            if enable_headers:
                contextual_chunk = f"Section: {title}\n\n{text}"
            else:
                contextual_chunk = text
            new_texts.append(contextual_chunk)
            src = {"title": title, "url": url}
            new_sources.append(src)
            new_meta.append(_compute_meta_from_source(src))
    
    def add_chunk(text: str, title: str, url: str) -> None:
        """Wrapper for backward compatibility."""
        add_chunk_with_context(text, title, url)
    
    def create_overlapping_chunks(items: List[str], title: str, url: str, chunk_size: int = 3, overlap: int = 1) -> None:
        """
        Create overlapping chunks from a list of text items.
        
        Args:
            items: List of text items (sentences, paragraphs, or list items)
            title: Section title for context
            url: Source URL
            chunk_size: Number of items per chunk
            overlap: Number of items to overlap between chunks
        """
        if not items:
            return
        
        # If we have few items, just combine them
        if len(items) <= chunk_size:
            combined = " ".join(items)
            add_chunk(combined, title, url)
            return
        
        # Create overlapping chunks
        for i in range(0, len(items), chunk_size - overlap):
            chunk_items = items[i:i + chunk_size]
            if chunk_items:
                combined = " ".join(chunk_items)
                add_chunk(combined, title, url)
            
            # Stop if we've reached the end
            if i + chunk_size >= len(items):
                break
    
    def process_section(
        section: Dict[str, Any],
        parent_title: str = "",
        base_url: str = ""
    ) -> None:
        nonlocal page_count
        
        if not isinstance(section, dict):
            return
        
        page_count += 1
        
        title = section.get("title", "").strip()
        url = section.get("page_url", base_url).strip()
        
        # build hierarchical title
        full_title = (
            f"{parent_title} - {title}" if parent_title and title
            else (title or parent_title)
        )
        
        # process text content
        texts = section.get("text", [])
        if texts:
            # Collect valid text items
            valid_texts = []
            for text_item in texts:
                if isinstance(text_item, str):
                    stripped = text_item.strip()
                    if stripped:
                        valid_texts.append(stripped)
            
            # Create overlapping chunks from text items
            if valid_texts:
                enable_overlap = cfg.get("chunking", {}).get("enable_overlap", True)
                if enable_overlap:
                    text_chunk_size = cfg.get("chunking", {}).get("text_chunk_size", 3)
                    text_overlap = cfg.get("chunking", {}).get("text_overlap", 1)
                    create_overlapping_chunks(valid_texts, full_title, url, 
                                            chunk_size=text_chunk_size, overlap=text_overlap)
                else:
                    # Original behavior: add each text item separately
                    for text in valid_texts:
                        add_chunk(text, full_title, url)
        
        # process lists (filter out short navigational items)
        lists = section.get("lists", [])
        if lists:
            for list_group in lists:
                if isinstance(list_group, list):
                    valid_items = []
                    for item in list_group:
                        if isinstance(item, str):
                            stripped = item.strip()
                            if stripped:
                                valid_items.append(stripped)
                    
                    # Create overlapping chunks from list items
                    if valid_items:
                        enable_overlap = cfg.get("chunking", {}).get("enable_overlap", True)
                        if enable_overlap:
                            list_chunk_size = cfg.get("chunking", {}).get("list_chunk_size", 5)
                            list_overlap = cfg.get("chunking", {}).get("list_overlap", 2)
                            create_overlapping_chunks(valid_items, full_title, url,
                                                    chunk_size=list_chunk_size, overlap=list_overlap)
                        else:
                            # Original behavior: add each list item separately
                            for item in valid_items:
                                add_chunk(item, full_title, url)
        
        # process subsections recursively
        subsections = section.get("subsections", [])
        if subsections:
            for subsection in subsections:
                process_section(subsection, full_title, url)
    
    # load JSON file
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"ERROR: Could not parse JSON from {path}")
        return
    
    # process the root structure
    if isinstance(data, dict):
        pages = data.get("pages", [])
        if isinstance(pages, list):
            for page in pages:
                if isinstance(page, dict):
                    page_title = page.get("page_title", "")
                    page_url = page.get("page_url", "")
                    sections = page.get("sections", [])
                    if isinstance(sections, list):
                        for section in sections:
                            process_section(section, page_title, page_url)
    
    # embed and add new chunks
    if new_texts:
        new_embeds = embed_model.encode(new_texts, convert_to_numpy=True)
        
        if chunks_embeddings is None:
            chunks_embeddings = new_embeds
        else:
            chunks_embeddings = np.vstack([chunks_embeddings, new_embeds])
        
        chunk_texts.extend(new_texts)
        chunk_sources.extend(new_sources)
        chunk_meta.extend(new_meta)
        global CHUNK_NORMS
        if chunks_embeddings is not None:
            CHUNK_NORMS = np.linalg.norm(chunks_embeddings, axis=1)
        print(f"Loaded {len(new_texts)} chunks from {path}")
    else:
        print(f"WARNING: No text found in {path}")
    print(f"[loader] Pages parsed: {page_count}")
    print(f"[loader] New chunks: {len(new_texts)}  |  Total chunks: {len(chunk_texts)}")


def load_catalog(path: Path) -> None:
    if path.exists():
        load_json_file(str(path))
    else:
        print(f"WARNING: {path} not found, skipping.")


def load_initial_data() -> None:
    loaded_from_cache = load_chunks_cache()
    
    if not loaded_from_cache:
        print("Loading data from source files...")
        filenames = ["unh_catalog.json"]
        for name in filenames:
            load_catalog(DATA_DIR / name)
        save_chunks_cache()
    
    # build program index for fuzzy matching
    build_program_index(chunk_sources, chunk_meta)


def get_chunks_data() -> tuple:
    return chunks_embeddings, chunk_texts, chunk_sources, chunk_meta

def get_tier_counts() -> Dict[int, int]:
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for meta in chunk_meta:
        tier = (meta or {}).get("tier")
        if tier in counts:
            counts[tier] += 1
    return counts

def get_chunk_norms() -> Optional[np.ndarray]:
    """Get the precomputed chunk norms, computing them if needed."""
    global CHUNK_NORMS, chunks_embeddings
    if CHUNK_NORMS is None and chunks_embeddings is not None:
        CHUNK_NORMS = np.linalg.norm(chunks_embeddings, axis=1)
    return CHUNK_NORMS