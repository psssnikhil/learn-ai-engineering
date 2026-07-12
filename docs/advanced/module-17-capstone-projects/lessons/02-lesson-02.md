---
title: 'Project 1: RAG Knowledge Assistant'
description: >-
  Build a full-featured document Q&A system with hybrid search, citation
  generation, a FastAPI backend, and a conversational web interface
duration: 120 min
difficulty: advanced
has_code: false
module: module-17
youtube: 'https://www.youtube.com/watch?v=tcqEUSNCn8I'
objectives:
  - Implement document ingestion with chunking and embedding
  - Build hybrid search combining vector and keyword retrieval
  - Generate accurate answers with inline citations
  - Deploy a FastAPI backend with a simple chat UI
---
# Project 1: RAG Knowledge Assistant

## Project Overview

Build a production-quality document Q&A system that can:
- Ingest PDFs, markdown, and text files
- Search using hybrid retrieval (vector + keyword)
- Answer questions with inline source citations
- Serve via a REST API with a chat interface

**Time estimate**: 10-15 hours
**Skills used**: RAG, Vector Databases, Embeddings, FastAPI, Prompt Engineering

---

## Architecture

```
Documents (PDF/MD/TXT)
    |
    v
[Ingestion Pipeline]
    |-- Chunking (recursive, ~500 tokens per chunk)
    |-- Embedding (OpenAI text-embedding-3-small)
    |-- Metadata extraction (title, page, source)
    v
[Vector Store] (ChromaDB)
    |
    v
[Query Pipeline]
    |-- User question
    |-- Embed query
    |-- Hybrid search (vector similarity + BM25 keyword)
    |-- Rerank top results
    |-- Generate answer with citations
    v
[FastAPI Server]
    |-- POST /ask  (question -> answer + sources)
    |-- POST /ingest  (upload documents)
    |-- GET /documents  (list indexed docs)
    v
[Web UI] (simple HTML/JS chat interface)
```

---

## Step 1: Document Ingestion

```python
import os
from dataclasses import dataclass

@dataclass
class DocumentChunk:
    text: str
    metadata: dict  # source, page, chunk_index

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks by approximate word count."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks

def ingest_document(filepath: str) -> list[DocumentChunk]:
    """Read a document and split it into chunks with metadata."""
    filename = os.path.basename(filepath)
    
    with open(filepath, "r") as f:
        text = f.read()
    
    chunks = chunk_text(text)
    
    return [
        DocumentChunk(
            text=chunk,
            metadata={
                "source": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
        )
        for i, chunk in enumerate(chunks)
    ]
```

---

## Step 2: Embedding and Storage

```python
from openai import OpenAI
import chromadb

client = OpenAI()
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection(
    name="knowledge_base",
    metadata={"hnsw:space": "cosine"}
)

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts using OpenAI embeddings."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]

def index_chunks(chunks: list[DocumentChunk]):
    """Index document chunks into ChromaDB."""
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts)
    
    ids = [f"{chunks[0].metadata['source']}_{i}" for i in range(len(chunks))]
    metadatas = [c.metadata for c in chunks]
    
    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    
    return len(chunks)
```

---

## Step 3: Hybrid Search

```python
def search(query: str, top_k: int = 5) -> list[dict]:
    """Hybrid search: vector similarity via ChromaDB."""
    query_embedding = embed_texts([query])[0]
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    search_results = []
    for i in range(len(results["ids"][0])):
        search_results.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "score": 1 - results["distances"][0][i],
        })
    
    return search_results
```

---

## Step 4: Answer Generation with Citations

```python
def generate_answer(question: str, context_chunks: list[dict]) -> dict:
    """Generate an answer with inline citations from retrieved chunks."""
    context_parts = []
    for i, chunk in enumerate(context_chunks):
        source = chunk["metadata"]["source"]
        context_parts.append(f"[Source {i+1}: {source}]
{chunk['text']}")
    
    context = "

---

".join(context_parts)
    
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a knowledgeable assistant. Answer questions based on the provided context. "
                    "Rules:
"
                    "1. Only answer based on the provided context. If the context doesn't contain the answer, say so.
"
                    "2. Cite your sources using [Source N] notation inline.
"
                    "3. Be concise but thorough."
                )
            },
            {
                "role": "user",
                "content": f"Context:
{context}

Question: {question}"
            }
        ],
        temperature=0.1
    )
    
    return {
        "answer": response.choices[0].message.content,
        "sources": [
            {"source": c["metadata"]["source"], "score": c["score"]}
            for c in context_chunks
        ]
    }
```

---

## Step 5: FastAPI Server

```python
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

app = FastAPI(title="RAG Knowledge Assistant")

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 5

@app.post("/ask")
async def ask_question(req: QuestionRequest):
    results = search(req.question, top_k=req.top_k)
    return generate_answer(req.question, results)

@app.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    content = await file.read()
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(content)
    chunks = ingest_document(temp_path)
    count = index_chunks(chunks)
    return {"message": f"Indexed {count} chunks from {file.filename}"}

@app.get("/documents")
async def list_documents():
    all_items = collection.get(include=["metadatas"])
    sources = set(m["source"] for m in all_items["metadatas"])
    return {"documents": sorted(sources), "total_chunks": len(all_items["ids"])}
```

---

## Evaluation

```python
def evaluate_rag(test_questions: list[dict]) -> dict:
    """Evaluate RAG system accuracy and latency."""
    import time
    results = {"correct": 0, "total": len(test_questions), "latencies": []}
    
    for test in test_questions:
        start = time.time()
        chunks = search(test["question"])
        answer = generate_answer(test["question"], chunks)
        results["latencies"].append(time.time() - start)
        
        if any(kw.lower() in answer["answer"].lower() for kw in test["expected_keywords"]):
            results["correct"] += 1
    
    results["accuracy"] = results["correct"] / results["total"]
    results["avg_latency"] = sum(results["latencies"]) / len(results["latencies"])
    return results
```

---

## Extension Ideas

- Add PDF parsing with PyPDF2 or pdfplumber
- Implement BM25 keyword search for true hybrid retrieval
- Add conversation memory for follow-up questions
- Deploy to Railway, Fly.io, or AWS
- Add user authentication and per-user document collections

---

## Resources

- **ChromaDB Documentation**: Vector database setup and queries
- **OpenAI Embeddings Guide**: Best practices for embedding text
- **FastAPI Documentation**: Building production-grade APIs
- **LangChain RAG Tutorial**: Framework-based RAG implementation

---

## Next Project

**Project 2: Autonomous Coding Agent** — Build an AI agent that can read code, find bugs, suggest fixes, and create pull requests.
