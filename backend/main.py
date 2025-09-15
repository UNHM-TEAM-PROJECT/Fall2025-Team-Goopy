from fastapi import FastAPI
import uvicorn
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import json
from functools import lru_cache

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# FastAPI backend app
app = FastAPI()

# for chunking text
chunks_embeddings = None
chunk_texts = []
chunk_sources = []

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

        # build citations line by line
        seen = set()
        citation_lines = []
        for _, src in top_chunks:
            key = (src["title"], src.get("url"))
            if key not in seen:
                seen.add(key)
                line = f"- {src['title']}"
                if src.get("url"):
                    line += f" ({src['url']})"
                citation_lines.append(line)

        citations = "\n".join(citation_lines)
        return f"{answer}\n\nSources:\n{citations}"
    except Exception as e:
        return f"ERROR running local model: {e}"

# cached wrapper for answers
@lru_cache(maxsize=128)
def cached_answer_str(question_str):
    return _answer_question(question_str)

# FastAPI request model
class ChatRequest(BaseModel):
    message: str | list[str]
    history: list[str] | None = None

# FastAPI response model
class ChatResponse(BaseModel):
    answer: str

# FastAPI chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def answer_question(request: ChatRequest):
    message = request.message
    if isinstance(message, list):
        message = " ".join(message)
    answer = cached_answer_str(message)
    return ChatResponse(answer=answer)

if __name__ == "__main__":
    # Load data files and run server
    load_json_file("course_descriptions.json")
    load_json_file("degree_requirements.json")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
