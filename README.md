Here’s a **README.md** for your new RAG-based meeting summarizer.
It’s written so anyone can run it from scratch, with clear install, usage, and configuration details.

---

````markdown
# 📝 RAG-Based Meeting Summarizer

A Python pipeline to **summarize meeting transcripts** using **Retrieval-Augmented Generation (RAG)** with:
- **SentenceTransformers** for semantic embeddings
- **FAISS** for fast vector similarity search
- **Local LLM** (via LM Studio / OpenAI-compatible API) for high-quality summarization
- **spaCy + PyTextRank** for keyword extraction
- **Markdown-structured summaries** (Key Points, Detailed Insight, Keywords)
- **Per-person summaries** with individual contributions

---

## 🚀 Features

- **Full meeting summary** from large transcripts without token limit issues
- **Per-speaker summaries** based on retrieved, semantically relevant chunks
- **RAG approach**: retrieves only relevant transcript parts before sending to the LLM
- **Configurable**: chunk size, top-k retrieval, LLM endpoint, temperature, etc.
- **Local-first**: no cloud APIs required (works entirely offline if you have a local LLM)

---

## 📦 Installation

### 1. Clone and enter the project
```bash
git clone https://github.com/yourusername/rag-meeting-summarizer.git
cd rag-meeting-summarizer
````

### 2. Install dependencies

```bash
pip install -U pip setuptools wheel
pip install spacy pytextrank sentence-transformers faiss-cpu requests tqdm transformers
python -m spacy download en_core_web_sm
```

**Dependencies**

* [spaCy](https://spacy.io/) – NLP preprocessing
* [pytextrank](https://github.com/DerwenAI/pytextrank) – keyword extraction
* [sentence-transformers](https://www.sbert.net/) – embeddings
* [FAISS](https://github.com/facebookresearch/faiss) – vector search
* [transformers](https://huggingface.co/docs/transformers) – optional model loading
* [requests](https://docs.python-requests.org/) – API calls to local LLM
* [tqdm](https://tqdm.github.io/) – progress bars

---

## ⚙️ Configuration

Edit the **top section** of `rag_meeting_summary.py`:

```python
TRANSCRIPT_PATH = "meeting_transcript.txt"    # Input file
OUTPUT_SUMMARY = "Complete summary.txt"       # Final meeting summary
CHUNK_SIZE = 400                              # Words per chunk
CHUNK_OVERLAP = 50                            # Overlap between chunks
EMBED_MODEL = "all-MiniLM-L6-v2"              # SentenceTransformer model
TOP_K = 6                                     # Retrieved chunks for RAG

LOCAL_LLM_BASE = "http://127.0.0.1:1234/v1"   # LM Studio API base URL
LLM_MODEL = "gpt-4o-mini"                     # Model name loaded in LM Studio
```

---

## 📄 Input Format

The script expects **`meeting_transcript.txt`** in this format:

```
[00:00:05] Alice: "Hello everyone, let's start the meeting."
[00:00:10] Bob: "Today we will discuss the project status."
...
```

The regex pattern in the code extracts speaker names and messages automatically.

---

## ▶️ Usage

1. Make sure your **local LLM server is running** (LM Studio / Ollama / etc.) and listening on `LOCAL_LLM_BASE`.
2. Place your transcript in `meeting_transcript.txt`.
3. Run:

```bash
python rag_meeting_summary.py
```

---

## 📤 Output

* **Complete summary**: `Complete summary.txt`
  Markdown format with:

  * Key Points (bullet list)
  * Detailed Insight (paragraphs)
  * Keywords (comma-separated)

* **Per-speaker summaries**:
  Each speaker gets a `{SpeakerName}.txt` file.

Example:

```
Complete summary.txt
Alice.txt
Bob.txt
```

---

## 🔧 How It Works

1. **Preprocessing**:

   * Lemmatization
   * Stopword removal
   * Punctuation filtering

2. **Chunking**: Splits transcript into overlapping word chunks.

3. **Embedding**: Converts chunks into vector representations with `sentence-transformers`.

4. **Indexing**: Stores embeddings in a FAISS index for fast similarity search.

5. **Retrieval**: Selects top-k most relevant chunks for the summarization query.

6. **LLM Summarization**: Sends retrieved context to your **local LLM** for generation.

7. **Keyword Extraction**: Optionally extract keywords from summaries with PyTextRank.

---

## 📚 References

* [SentenceTransformers Docs](https://www.sbert.net/docs/)
* [FAISS GitHub](https://github.com/facebookresearch/faiss)
* [spaCy Docs](https://spacy.io/usage)
* [PyTextRank GitHub](https://github.com/DerwenAI/pytextrank)
* [LM Studio Docs](https://lmstudio.ai/docs)

---
