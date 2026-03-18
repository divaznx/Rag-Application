# PDF RAG — Ask Your Document

A full-stack RAG application using LangChain, ChromaDB, OpenAI GPT-4o, and Gradio.

## Stack
| Layer | Tool |
|---|---|
| PDF Parsing | `PyPDFLoader` (LangChain) |
| Text Splitting | `RecursiveCharacterTextSplitter` |
| Embeddings | `text-embedding-3-small` (OpenAI) |
| Vector Store | ChromaDB (local) |
| LLM | GPT-4o (OpenAI) |
| Chain | `RetrievalQA` (LangChain) |
| UI | Gradio |

## Setup

```bash
# 1. Clone / navigate to project
cd pdf-rag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your OpenAI key
cp .env.example .env
# Edit .env and add your key

# 5. Run
python app.py
```

Then open http://localhost:7860 in your browser.

## Project Structure

```
pdf-rag/
├── app.py           # Gradio UI — all frontend logic
├── rag.py           # RAG core: load, index, retrieve, answer
├── requirements.txt
├── .env.example
└── vectorstore/     # ChromaDB persists here (auto-created)
```

## How it works

```
PDF
 └─► PyPDFLoader (pages)
       └─► RecursiveCharacterTextSplitter (1000-char chunks, 150 overlap)
             └─► OpenAI Embeddings (text-embedding-3-small)
                   └─► ChromaDB (stored locally)
                         └─► RetrievalQA chain
                               ├─► User question → embed → top-4 chunks
                               └─► GPT-4o answers from chunks only
```

## Key design decisions

- **Strict RAG prompt** — GPT-4o is instructed to answer only from retrieved context.
  If the answer isn't in the document, it says so (no hallucination).
- **Source citations** — Every answer shows which pages the answer came from.
- **Chunk overlap (150)** — Prevents answers from being cut off at chunk boundaries.
- **K=4 retrieval** — Fetches top-4 most similar chunks per question, balancing
  context richness vs. prompt length.
