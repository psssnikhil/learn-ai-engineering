---
title: Building a Basic RAG System
description: >-
  Build an end-to-end RAG pipeline — ingest documents, embed chunks, retrieve
  context, and generate grounded answers
duration: 40 min
difficulty: intermediate
has_code: false
youtube: 'https://www.youtube.com/watch?v=sVcwVQRHIc8'
---
## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Build a complete RAG pipeline from scratch | 40 min | Intermediate |
| Ingest, chunk, and embed documents | | |
| Retrieve relevant context for a query | | |
| Generate grounded LLM responses | | |

---

## Architecture Overview

Every RAG system follows the same four stages:

```
Documents → Chunk → Embed → Store
                              ↓
User Query → Embed → Search → Top-K chunks → LLM → Answer
```

> **Watch First:** [Build a RAG System from Scratch](https://www.youtube.com/watch?v=sVcwVQRHIc8) — walkthrough of each stage.

---

## Step 1: Document Ingestion

Start with clean text. Strip boilerplate, normalize whitespace, and attach metadata.

```python
def ingest_document(text: str, doc_id: str, title: str) -> dict:
    return {
        "doc_id": doc_id,
        "title": title,
        "text": " ".join(text.split()),  # normalize whitespace
        "source": "internal_kb",
    }
```

---

## Step 2: Chunking

Split documents into retrieval-sized pieces (typically 300–800 tokens with 10–20% overlap).

```python
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
```

---

## Step 3: Embed and Store

```python
import chromadb
from openai import OpenAI

client = OpenAI()
chroma = chromadb.PersistentClient(path="./rag_store")
collection = chroma.get_or_create_collection("docs")

def embed(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding

def index_chunks(chunks: list[str], doc_id: str):
    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[f"{doc_id}_{i}"],
            documents=[chunk],
            embeddings=[embed(chunk)],
            metadatas=[{"doc_id": doc_id, "chunk_index": i}],
        )
```

---

## Step 4: Retrieve and Generate

```python
def retrieve(query: str, n_results: int = 4) -> list[str]:
    results = collection.query(
        query_embeddings=[embed(query)],
        n_results=n_results,
    )
    return results["documents"][0]

def answer(query: str) -> str:
    context_chunks = retrieve(query)
    context = "

---

".join(context_chunks)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the provided context. "
                    "If the answer is not in the context, say you don't know."
                ),
            },
            {
                "role": "user",
                "content": f"Context:
{context}

Question: {query}",
            },
        ],
    )
    return response.choices[0].message.content
```

---

## Complete Pipeline Class

```python
class RAGSystem:
    def __init__(self, collection_name: str = "docs"):
        self.client = OpenAI()
        self.chroma = chromadb.PersistentClient(path="./rag_store")
        self.collection = self.chroma.get_or_create_collection(collection_name)

    def add_document(self, text: str, doc_id: str):
        for i, chunk in enumerate(chunk_text(text)):
            self.collection.add(
                ids=[f"{doc_id}_{i}"],
                documents=[chunk],
                embeddings=[self._embed(chunk)],
                metadatas=[{"doc_id": doc_id}],
            )

    def _embed(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    def query(self, question: str) -> str:
        return answer(question)
```

> **Tip:** Log retrieved chunk IDs with every answer. When users report bad responses, you can inspect exactly what context the model saw.

---

## Common Pitfalls

| Problem | Symptom | Fix |
|---------|---------|-----|
| Chunks too large | Vague retrieval | Reduce chunk size |
| No overlap | Lost context at boundaries | Add 10–20% overlap |
| No grounding instruction | Hallucinations | Enforce "answer from context only" |
| Stale index | Wrong answers after doc updates | Re-index on content changes |

---

## Recommended Videos

- [Build a RAG System from Scratch](https://www.youtube.com/watch?v=sVcwVQRHIc8)
- [RAG with LangChain](https://www.youtube.com/watch?v=tcqEUSNCn8I)

---

## Additional Resources

- [ChromaDB Docs](https://docs.trychroma.com/)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [OpenAI Cookbook: RAG](https://cookbook.openai.com/examples/question_answering_using_embeddings)
