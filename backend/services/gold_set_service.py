import json
from pathlib import Path
from typing import List, Dict, Optional, Set
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

class GoldSetManager:
    
    def __init__(self, gold_file_path: str = "../automation_testing/gold.jsonl"):
        self.gold_file_path = Path(gold_file_path)
        self.gold_entries: List[Dict] = []
        self.gold_questions: Set[str] = set()
        self.gold_answers: Dict[str, str] = {}
        self.gold_urls: Dict[str, str] = {}
        self.gold_embeddings: Optional[np.ndarray] = None
        self.embed_model: Optional[SentenceTransformer] = None
        
        # load gold set
        self._load_gold_set()
    
    def _load_gold_set(self):
        if not self.gold_file_path.exists():
            print(f"Warning: Gold set file not found at {self.gold_file_path}")
            return
        
        with open(self.gold_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    self.gold_entries.append(entry)
                    
                    # index by question
                    query = entry.get('query', '').lower()
                    self.gold_questions.add(query)
                    
                    # map question to answer
                    self.gold_answers[query] = entry.get('reference_answer', '')
                    
                    # map question to URL
                    self.gold_urls[query] = entry.get('url', '')
        
        print(f"Loaded {len(self.gold_entries)} gold Q&A pairs")
    
    def get_gold_documents(self) -> List[Document]:
        documents = []
        
        for entry in self.gold_entries:
            # create a document from Q&A pair
            content = f"Question: {entry.get('query', '')}\n\nAnswer: {entry.get('reference_answer', '')}"
            
            # add nuggets for additional context
            nuggets = entry.get('nuggets', [])
            if nuggets:
                content += f"\n\nKey Points: {'; '.join(nuggets)}"
            
            doc = Document(
                page_content=content,
                metadata={
                    'source': 'gold_set',
                    'gold_id': entry.get('id', ''),
                    'url': entry.get('url', ''),
                    'tier': 0,  # Highest priority tier
                    'is_gold': True,
                    'original_query': entry.get('query', ''),
                    'gold_passages': entry.get('gold_passages', [])
                }
            )
            documents.append(doc)
        
        return documents
    
    def compute_gold_embeddings(self, embed_model: SentenceTransformer):
        self.embed_model = embed_model
        
        queries = [entry.get('query', '') for entry in self.gold_entries]
        self.gold_embeddings = embed_model.encode(queries, convert_to_numpy=True)
        
        print(f"Computed embeddings for {len(queries)} gold questions")
    
    def find_matching_gold_entry(self, query: str, threshold: float = 0.85) -> Optional[Dict]:
        if self.gold_embeddings is None or self.embed_model is None:
            return None
        
        # compute query embedding
        query_embedding = self.embed_model.encode([query], convert_to_numpy=True)[0]
        
        # compute similarities
        similarities = np.dot(self.gold_embeddings, query_embedding) / (
            np.linalg.norm(self.gold_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # find best match
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        
        if best_score >= threshold:
            entry = self.gold_entries[best_idx].copy()
            entry['match_score'] = best_score
            return entry
        
        return None
    
    def is_gold_url(self, url: str) -> bool:
        return any(url in gold_url for gold_url in self.gold_urls.values())
    
    def get_gold_boost_for_chunk(
        self, 
        chunk_text: str, 
        chunk_metadata: Dict,
        query: str
    ) -> float:
        boost = 1.0
        
        # check if chunk is from gold set
        if chunk_metadata.get('is_gold', False):
            boost *= 2.5  # Strong boost for gold chunks
        
        # check if chunk URL matches gold URLs
        chunk_url = chunk_metadata.get('url', '')
        if self.is_gold_url(chunk_url):
            boost *= 1.5  # Moderate boost for gold URLs
        
        # check if chunk contains gold answer text
        query_lower = query.lower()
        if query_lower in self.gold_questions:
            gold_answer = self.gold_answers.get(query_lower, '')
            if gold_answer and gold_answer.lower() in chunk_text.lower():
                boost *= 1.8  # Boost for chunks containing gold answers
        
        return boost
    
    def get_direct_answer(self, query: str, threshold: float = 0.85) -> Optional[str]:
        match = self.find_matching_gold_entry(query, threshold)
        if match:
            return match.get('reference_answer')
        return None
    
    def get_statistics(self) -> Dict:
        return {
            'total_entries': len(self.gold_entries),
            'total_questions': len(self.gold_questions),
            'categories': self._get_categories(),
            'has_embeddings': self.gold_embeddings is not None
        }
    
    def _get_categories(self) -> Dict[str, int]:
        categories = {}
        for entry in self.gold_entries:
            category = entry.get('id', '').split(':')[0]
            categories[category] = categories.get(category, 0) + 1
        return categories

# blobal instance
_gold_manager: Optional[GoldSetManager] = None

def get_gold_manager() -> GoldSetManager:
    global _gold_manager
    if _gold_manager is None:
        _gold_manager = GoldSetManager()
    return _gold_manager

def initialize_gold_set(embed_model: SentenceTransformer):
    manager = get_gold_manager()
    manager.compute_gold_embeddings(embed_model)
    return manager