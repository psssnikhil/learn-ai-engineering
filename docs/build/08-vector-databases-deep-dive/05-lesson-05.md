---
title: ChromaDB and Open-Source Vector Databases
description: >-
  Build vector search applications using ChromaDB, Weaviate, and other
  open-source alternatives to managed services
duration: 40 min
difficulty: intermediate
has_code: false
---
# ChromaDB and Open-Source Vector Databases

## Learning Objectives

By the end of this lesson, you will be able to:
- Set up and use ChromaDB for local vector search
- Compare open-source vector database options (Chroma, Weaviate, Qdrant, Milvus)
- Build a complete document search application with ChromaDB
- Understand when to choose open-source vs managed solutions

---

## ChromaDB: The Developer-Friendly Vector Database

ChromaDB is an open-source embedding database designed for simplicity. It runs locally, handles embeddings automatically, and needs minimal setup.

### Quick Start

```python
import chromadb

# Create a persistent client (saves to disk)
client = chromadb.PersistentClient(path="./chroma_db")

# Create a collection
collection = client.get_or_create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}  # distance metric
)

# Add documents -- ChromaDB generates embeddings automatically
collection.add(
    documents=[
        "RAG combines retrieval with generation for grounded answers",
        "Vector databases store embeddings for similarity search",
        "Fine-tuning customizes models for specific tasks",
        "Prompt engineering optimizes LLM inputs for better outputs",
    ],
    metadatas=[
        {"category": "rag", "difficulty": "intermediate"},
        {"category": "vector-db", "difficulty": "beginner"},
        {"category": "training", "difficulty": "advanced"},
        {"category": "prompting", "difficulty": "beginner"},
    ],
    ids=["doc-1", "doc-2", "doc-3", "doc-4"]
)

# Query
results = collection.query(
    query_texts=["How do I search documents by meaning?"],
    n_results=3
)

for doc, distance in zip(results["documents"][0], results["distances"][0]):
    print(f"[{distance:.3f}] {doc}")
```

---

## Using Custom Embedding Models

```python
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Use OpenAI embeddings instead of default
openai_ef = OpenAIEmbeddingFunction(
    api_key="your-api-key",
    model_name="text-embedding-3-small"
)

collection = client.get_or_create_collection(
    name="openai-docs",
    embedding_function=openai_ef
)

# Or use Sentence Transformers (free, local)
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

local_ef = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="local-docs",
    embedding_function=local_ef
)
```

---

## Filtered Search in ChromaDB

```python
# Filter by metadata
results = collection.query(
    query_texts=["beginner AI concepts"],
    n_results=5,
    where={"difficulty": "beginner"}
)

# Combine metadata and document content filters
results = collection.query(
    query_texts=["search techniques"],
    n_results=5,
    where={"category": {"$in": ["rag", "vector-db"]}},
    where_document={"$contains": "retrieval"}
)
```

---

## Open-Source Vector Database Comparison

| Feature | ChromaDB | Weaviate | Qdrant | Milvus |
|---------|----------|----------|--------|--------|
| **Best for** | Prototyping, small-medium scale | Full-featured production | High performance | Enterprise scale |
| **Language** | Python | Go | Rust | Go/C++ |
| **Hosting** | Local / Cloud | Self-host / Cloud | Self-host / Cloud | Self-host / Cloud |
| **Auto embeddings** | Yes | Yes | No | No |
| **Hybrid search** | No | Yes (BM25 + vector) | Yes | Yes |
| **Max vectors** | Millions | Billions | Billions | Billions |
| **Setup complexity** | Very low | Medium | Low | High |

---

## Building a Document Search App

```python
import chromadb
from pathlib import Path

def build_document_search(docs_dir: str) -> chromadb.Collection:
    """Index text files from a directory into ChromaDB."""
    client = chromadb.PersistentClient(path="./search_db")
    collection = client.get_or_create_collection("file-search")

    documents, metadatas, ids = [], [], []
    for filepath in Path(docs_dir).glob("*.txt"):
        text = filepath.read_text()
        documents.append(text)
        metadatas.append({"filename": filepath.name, "size": len(text)})
        ids.append(filepath.stem)

    collection.upsert(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    return collection

def search(collection, query: str, n: int = 3):
    """Search the document collection."""
    results = collection.query(query_texts=[query], n_results=n)
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        print(f"
--- {meta['filename']} (score: {1-dist:.3f}) ---")
        print(doc[:200] + "..." if len(doc) > 200 else doc)
```

---

## Key Takeaways

- ChromaDB is the fastest way to get started with vector search -- it handles embeddings and storage automatically
- Open-source databases offer flexibility and avoid vendor lock-in
- ChromaDB is ideal for prototypes and small-to-medium workloads
- Weaviate and Qdrant are better choices for production deployments needing hybrid search or high throughput
- Choose based on your scale, team expertise, and whether you need managed hosting

## Resources

- [ChromaDB Documentation](https://docs.trychroma.com/) -- Official getting started guide
- [YouTube: ChromaDB Tutorial](https://www.youtube.com/watch?v=QSW2L8dkaZk) -- Building semantic search with Chroma
- [Weaviate Docs](https://weaviate.io/developers/weaviate) -- Weaviate getting started
- [Qdrant Documentation](https://qdrant.tech/documentation/) -- Qdrant setup and usage

---

Next: Vector Database Schema Design
