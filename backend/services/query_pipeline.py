"""
Simplified pipeline for preprocessing and retrieval logic.
"""
from services.qa_service import cached_answer_with_path

def process_question_for_retrieval(incoming_message):
    """
    NOTE: Intent detection, program level detection, and program alias matching 
    have been removed as they were broken and degrading retrieval quality.
    
    Args:
        incoming_message: The user's question (string or list of strings)
    
    Returns:
        dict with keys: answer, sources, retrieval_path, context
    """
    # handle list messages
    if isinstance(incoming_message, list):
        incoming_message = " ".join(incoming_message)

    # Simple retrieval without any scoping or intent-based modifications
    answer, sources, retrieval_path, context = cached_answer_with_path(incoming_message)

    return dict(
        answer=answer,
        sources=sources,
        retrieval_path=retrieval_path,
        context=context,
    )

