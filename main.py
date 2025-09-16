import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import json
from functools import lru_cache

chunks_embeddings = None
chunk_texts = []
chunk_sources = []

embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=-1
)

def load_json_file(path):
    global chunks_embeddings, chunk_texts, chunk_sources

    with open(path, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    new_texts = []
    new_sources = []

    def recurse_sections(sections, parent_title="", page_url=""):
        for sec in sections:
            title = sec.get("title", "")
            full_title = f"{parent_title} > {title}" if parent_title else title

            for t in sec.get("text", []):
                new_texts.append(t)
                new_sources.append({"title": full_title, "url": page_url})

            for link in sec.get("links", []):
                label = link.get("label")
                url = link.get("url")
                if label and url:
                    new_texts.append(f"Courses: {label}")
                    new_sources.append({"title": label, "url": url})

            if "subsections" in sec:
                recurse_sections(sec["subsections"], parent_title=full_title, page_url=page_url)

    for page_data in data_list:
        recurse_sections(page_data.get("sections", []), page_url=page_data.get("url", ""))

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

def get_top_chunks(question, top_k=3):
    if chunks_embeddings is None or len(chunks_embeddings) == 0:
        return []
    question_vec = embed_model.encode([question], convert_to_numpy=True)[0]
    scores = np.dot(chunks_embeddings, question_vec) / (
        np.linalg.norm(chunks_embeddings, axis=1) * np.linalg.norm(question_vec) + 1e-10
    )
    top_indices = scores.argsort()[-top_k:][::-1]
    return [(chunk_texts[i], chunk_sources[i]) for i in top_indices]

def _answer_question(question):
    top_chunks = get_top_chunks(question)
    context = " ".join([text for text, _ in top_chunks])

    prompt = (
        "Answer the question ONLY using the provided context. "
        "If the answer cannot be found, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    try:
        result = qa_pipeline(prompt, max_new_tokens=128)
        answer = result[0]["generated_text"].strip()

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

@lru_cache(maxsize=128)
def cached_answer_str(question_str):
    return _answer_question(question_str)

def answer_question(message, history=None):
    if isinstance(message, list):
        message = " ".join(message)
    return cached_answer_str(message)

# --- Gradio UI ---
unh_blue = "#003366"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    with gr.Row(elem_id="header"):
        gr.HTML(
            f"""
            <div style="background-color: {unh_blue}; color: white; padding:20px; border-radius:8px; text-align:center;">
                <h1>UNH Graduate Catalog Chatbot</h1>
                <p>Ask questions about programs, courses, and policies from the UNH Graduate Catalog</p>
            </div>
            """
        )

    chatbot = gr.ChatInterface(fn=answer_question, type="messages")

    with gr.Row(elem_id="footer"):
        gr.Markdown(
            f"""
            <div style="text-align:center; padding:10px; font-size:14px; color:#555;">
                <hr style="margin:10px 0;">
                <p>Built for the <strong>University of New Hampshire</strong> Graduate Catalog project</p>
            </div>
            """
        )

if __name__ == "__main__":
    load_json_file("unh_catalog.json")
    demo.launch()
