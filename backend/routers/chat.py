from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from general_responses import get_generic_response
from models.api_models import ChatRequest, ChatResponse
from services.session_service import get_session, update_session, clear_session, clear_all_sessions, push_history, now_iso
from utils.logging_utils import log_chat_interaction
import importlib

router = APIRouter()

# New helper to wrap available query pipeline functions and normalize output
def _process_question(question: str) -> dict:
    qp = importlib.import_module("services.query_pipeline")
    # Try preferred function
    if hasattr(qp, "cached_answer_with_path"):
        out = qp.cached_answer_with_path(question)
        if isinstance(out, dict):
            return {
                "answer": out.get("answer") or out.get("response") or "",
                "sources": out.get("sources") or [],
                "retrieval_path": out.get("retrieval_path") or out.get("path") or [],
                "transformed_query": out.get("transformed_query") or question,
            }
        if isinstance(out, tuple):
            answer = out[0] if len(out) > 0 else ""
            sources = out[1] if len(out) > 1 else []
            retrieval_path = out[2] if len(out) > 2 else []
            transformed_query = out[3] if len(out) > 3 else question
            return {
                "answer": answer,
                "sources": sources,
                "retrieval_path": retrieval_path,
                "transformed_query": transformed_query,
            }
        # Fallback if unexpected type
        return {
            "answer": str(out),
            "sources": [],
            "retrieval_path": [],
            "transformed_query": question,
        }
    # Fallback to _answer_question
    if hasattr(qp, "_answer_question"):
        answer = qp._answer_question(question)
        return {
            "answer": answer,
            "sources": [],
            "retrieval_path": [],
            "transformed_query": question,
        }
    raise RuntimeError("No suitable query function found in services.query_pipeline")

@router.post("/chat", response_model=ChatResponse)
async def answer_question(request: ChatRequest, x_session_id: Optional[str] = Header(default=None)):
    if not x_session_id:
        raise HTTPException(
            status_code=400,
            detail="Please include X-Session-Id header"
        )
    incoming_message = request.message if not isinstance(request.message, list) else " ".join(request.message)
    
    # Check for generic responses first
    resp = get_generic_response(incoming_message)
    if resp:
        return ChatResponse(answer=resp, sources=[], retrieval_path=[])
    
    # Process question
    result = _process_question(incoming_message)
    answer = result["answer"]
    
    # Update session with last question
    update_session(
        x_session_id,
        last_question=incoming_message,
        last_answer=answer
    )
    
    # Store simplified history
    try:
        push_history(
            x_session_id,
            {
                "timestamp": now_iso(),
                "question": incoming_message,
                "answer": answer,
                "retrieval_path": result["retrieval_path"],
            },
        )
    except Exception:
        pass
    
    log_chat_interaction(incoming_message, answer, result["sources"])
    return ChatResponse(
        answer=answer, 
        sources=result["sources"], 
        retrieval_path=result["retrieval_path"],
        transformed_query=result["transformed_query"]
    )

@router.post("/reset-session")
def reset_one_session(x_session_id: Optional[str] = Header(default=None)):
    if not x_session_id:
        raise HTTPException(
            status_code=400,
            detail="Please include X-Session-Id header"
        )
    if clear_session(x_session_id):
        return {"status": "session_cleared", "session_id": x_session_id}
    return {"status": "no_session_to_clear", "session_id": x_session_id}

@router.post("/reset")
async def reset_all_sessions():
    clear_all_sessions()
    return {"status": "cleared"}