import re
from typing import Optional
from config.settings import get_config

def score_answer_candidate(answer: str, question: str, context: str) -> float:
    """
    Score an answer candidate based on quality heuristics.
    Higher scores indicate better answers.
    """
    from services.qa_service import UNKNOWN
    
    # Heavily penalize UNKNOWN responses
    if answer.strip() == UNKNOWN:
        return -100.0
    
    score = 0.0
    
    # Basic length check - penalize very short answers
    word_count = len(answer.split())
    if word_count < 3:
        score -= 5.0
    
    # Reward context relevance
    context_terms = set(re.findall(r'\b\w{3,}\b', context.lower()))
    answer_terms = set(re.findall(r'\b\w{3,}\b', answer.lower()))
    if context_terms and answer_terms:
        overlap = len(context_terms & answer_terms)
        score += overlap * 0.1
    
    return score

def generate_with_beam_search(
    qa_pipeline,
    prompt: str,
    question: str,
    context: str,
    beam_width: Optional[int] = None,
    num_candidates: Optional[int] = None
) -> Optional[str]:
    """
    Generate answer using beam search to explore multiple candidate answers.
    Returns the highest-scoring answer, or None if beam search fails.
    
    Args:
        qa_pipeline: The QA model pipeline
        prompt: The formatted prompt for generation
        question: The original question (for scoring)
        context: The context (for scoring)
        beam_width: Number of beams to explore (uses config if None)
        num_candidates: Number of candidates to generate (uses config if None)
    
    Returns:
        Best answer string, or None if beam search fails
    """
    cfg = get_config()
    beam_cfg = cfg.get("beam_search", {})
    
    # Use provided params or fall back to config
    beam_width = beam_width or beam_cfg.get("beam_width", 5)
    num_candidates = num_candidates or beam_cfg.get("num_candidates", 3)
    num_candidates = min(num_candidates, beam_width)
    
    try:
        results = qa_pipeline(
            prompt,
            max_new_tokens=128,
            num_beams=beam_width,
            num_return_sequences=num_candidates,
            repetition_penalty=1.2,
            no_repeat_ngram_size=2,
            early_stopping=True,
            diversity_penalty=beam_cfg.get("diversity_penalty", 0.3),
        )
        
        # Score all candidates
        candidates = []
        for result in results:
            answer = result["generated_text"].strip()
            score = score_answer_candidate(answer, question, context)
            candidates.append((answer, score))
        
        # Return the highest-scoring candidate
        if candidates:
            best_answer, _ = max(candidates, key=lambda x: x[1])
            return best_answer
        
    except Exception:
        # Return None to signal failure - caller will handle fallback
        return None
    
    return None
