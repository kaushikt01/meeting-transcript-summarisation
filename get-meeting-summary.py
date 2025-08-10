
import os
import re
import json
from typing import List
from tqdm import tqdm

import requests
import numpy as np
import faiss
import spacy
import pytextrank
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
TRANSCRIPT_PATH = "meeting_transcript.txt"
OUTPUT_SUMMARY = "Complete summary.txt"
CHUNK_SIZE = 400     # words per chunk (tune)
CHUNK_OVERLAP = 50   # words overlap between chunks
EMBED_MODEL = "all-MiniLM-L6-v2"  # sentence-transformer
FAISS_INDEX_PATH = "faiss_index.idx"
METADATA_PATH = "chunks_metadata.json"

# Local LLM settings (LM Studio default base url / OpenAI-compatible)
LOCAL_LLM_BASE = os.getenv("LOCAL_LLM_BASE", "http://127.0.0.1:1234/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")   # replace with your model id loaded in LM Studio

TOP_K = 6  # number of chunks to retrieve for RAG
# ----------------------------

# ---------- UTIL: read transcript ----------
with open(TRANSCRIPT_PATH, "r", encoding="utf-8") as f:
    transcript = f.read()

# ---------- spaCy preprocessing ----------
nlp = spacy.load("en_core_web_sm")
# add pytextrank for keyword extraction (used later)
nlp.add_pipe("textrank")

def preprocess_text(text: str) -> str:
    doc = nlp(text)
    cleaned = " ".join(
        token.lemma_.lower() for token in doc
        if not token.is_stop and not token.is_punct and token.lemma_.strip()
    )
    return cleaned

# ---------- chunking ----------
def chunk_text_words(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

# create top-level cleaned transcript and chunks
cleaned_transcript = preprocess_text(transcript)
chunks = chunk_text_words(cleaned_transcript)

# ---------- embedding model ----------
embedder = SentenceTransformer(EMBED_MODEL)  # downloads if not present

# compute embeddings for all chunks
print(f"[+] Computing embeddings for {len(chunks)} chunks...")
embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

# ---------- FAISS index (Flat L2) ----------
dim = embeddings.shape[1]
if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH):
    # load existing
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    print("[+] Loaded FAISS index and metadata from disk.")
else:
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    metadata = {"chunks": chunks}
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f)
    print("[+] Built FAISS index and saved to disk.")

# ---------- Retrieval ----------
def retrieve_top_k(query: str, k: int = TOP_K) -> List[str]:
    q_emb = embedder.encode([query], convert_to_numpy=True)
    distances, idxs = index.search(q_emb, k)
    idxs = idxs[0].tolist()
    result_chunks = []
    for i in idxs:
        if i < 0 or i >= len(metadata["chunks"]):
            continue
        result_chunks.append(metadata["chunks"][i])
    return result_chunks

# ---------- LLM call (OpenAI compatibility) ----------
def call_local_llm(prompt: str, system: str = None, max_tokens: int = 512, temperature: float = 0.2):
    """
    Calls OpenAI-compatible /v1/chat/completions endpoint at LOCAL_LLM_BASE.
    LM Studio supports OpenAI-compatible endpoints (see docs).
    """
    url = f"{LOCAL_LLM_BASE}/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": LLM_MODEL,
        "messages": []
    }
    if system:
        payload["messages"].append({"role": "system", "content": system})
    payload["messages"].append({"role": "user", "content": prompt})
    payload["max_tokens"] = max_tokens
    payload["temperature"] = temperature

    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    r.raise_for_status()
    resp = r.json()
    # OpenAI-compatible responses often have choices[0].message.content
    if "choices" in resp and len(resp["choices"]) > 0:
        # support both chat and completion style
        choice = resp["choices"][0]
        if "message" in choice and "content" in choice["message"]:
            return choice["message"]["content"]
        if "text" in choice:
            return choice["text"]
    # fallback full json
    return json.dumps(resp)

# ---------- RAG summarization prompt ----------
RAG_SYSTEM_PROMPT = (
    "You are an expert meeting summarizer. Produce a concise, structured summary suitable for busy managers. "
    "Output in Markdown with sections: Key Points (bullet list), Detailed Insight (short paragraphs), and Keywords (comma-separated). "
    "If the context is partial or missing details, note that you used retrieved excerpts and may have truncated context."
)

def rag_summarize(query: str, top_k: int = TOP_K) -> str:
    retrieved = retrieve_top_k(query, top_k)
    context = "\n\n---\n\n".join(retrieved)
    prompt = (
        f"CONTEXT (retrieved transcript chunks):\n{context}\n\n"
        f"USER QUERY: {query}\n\n"
        "Now produce the summary as requested in the system prompt."
    )
    return call_local_llm(prompt, system=RAG_SYSTEM_PROMPT, max_tokens=600, temperature=0.1)

# ---------- Combined flow: meeting-level summary ----------
print("[+] Running RAG summarization for the whole meeting...")
meeting_query = "Summarize the meeting transcript. Give Key Points, Detailed Insight, and Keywords."
meeting_summary = rag_summarize(meeting_query, top_k=TOP_K)

with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as f:
    f.write(meeting_summary)
print(f"[+] Wrote meeting summary to {OUTPUT_SUMMARY}")

# ---------- Per-person summaries ----------
# extract speaker lines from original transcript using regex pattern used earlier
pattern = r"\[(\d{2}:\d{2}:\d{2})\]\s*(.*?):\s*\"(.*?)\""  # pattern used in earlier code
matches = re.findall(pattern, transcript)
person_map = {}
for ts, name, msg in matches:
    person_map.setdefault(name.strip(), []).append(msg.strip())

print(f"[+] Found {len(person_map)} speakers; generating per-person summaries...")
for name, msgs in person_map.items():
    joined = " ".join(msgs)
    cleaned_person_text = preprocess_text(joined)
    # temporary per-person chunks and embeddings (we can reuse the global embedder)
    person_chunks = chunk_text_words(cleaned_person_text)
    # compute person embeddings and simple in-memory index for retrieval
    if len(person_chunks) == 0:
        continue
    person_embs = embedder.encode(person_chunks, convert_to_numpy=True)
    # build tiny FAISS (in-memory)
    p_index = faiss.IndexFlatL2(person_embs.shape[1])
    p_index.add(person_embs)
    # quick local retrieve function
    def person_retrieve(query, k=4):
        q_emb = embedder.encode([query], convert_to_numpy=True)
        d, ids = p_index.search(q_emb, k)
        return [person_chunks[i] for i in ids[0] if i >= 0 and i < len(person_chunks)]
    person_context = "\n\n---\n\n".join(person_retrieve(f"Summarize the contributions of {name}", k=4))
    prompt = (
        f"CONTEXT (person {name}):\n{person_context}\n\n"
        f"USER QUERY: Please summarize contributions and action items for {name}. Provide Key Points, Detailed Insight and Keywords."
    )
    out = call_local_llm(prompt, system=RAG_SYSTEM_PROMPT, max_tokens=400, temperature=0.0)
    fname = f"{name}.txt"
    with open(fname, "w", encoding="utf-8") as of:
        of.write(out)
    print(f"  - Wrote per-person summary: {fname}")

print("[+] Done.")
