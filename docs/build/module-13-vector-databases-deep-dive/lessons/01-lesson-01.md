---
title: Introduction to Vector Databases
description: >-
  Understand what vector databases are, how they work, and why they are
  essential for modern AI applications like RAG and semantic search
duration: 50 min
difficulty: intermediate
has_code: true
module: module-13
---
# Introduction to Vector Databases

## Prerequisites

Before this lesson you should be comfortable with:

- **Module 09, Lesson 02 — Vector Databases & Embeddings** — you know what embeddings are, how cosine similarity works, and why RAG needs a retrieval index
- **Basic RAG pipeline** — chunk → embed → store → query (Module 09, Lesson 01)
- **Python** — enough to read and run the code examples below

This module goes *deeper* than the Module 09 overview. We assume you have built a basic Chroma index; here we focus on production architecture, ANN trade-offs, and database selection at scale.

---

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Articulate why keyword databases fail for semantic search | 10 min | Intermediate |
| Trace a query through embedding → ANN search → metadata return | 15 min | Intermediate |
| Compare managed vs self-hosted vector databases for production | 10 min | Intermediate |
| Run a side-by-side semantic vs keyword search experiment | 15 min | Intermediate |

---

## Intuition First: The Filing Cabinet vs the Map

Imagine two ways to find a document in a library.

**Keyword filing cabinet:** Every document is filed alphabetically by exact words. Ask for "running shoes" and you get documents containing those exact tokens. "Jogging sneakers" sits in a different drawer — invisible to your search.

**Semantic map:** Every document is pinned to a coordinate on a giant map where *nearby pins mean similar meaning*. You place a pin for your query ("comfortable shoes for long walks") and walk to the nearest neighbors. "Cushioned walking shoes," "daily commuter sneakers," and "orthopedic footwear" all cluster nearby — even without shared keywords.

A **vector database** is that map. It stores coordinates (embeddings) and finds nearest neighbors in milliseconds, even when the map has a billion pins.

The crucial insight from Module 09 still holds: **the LLM reasons; the vector database remembers where things are**. This module teaches you to build and operate the map itself.

---

## Why Vector Databases Exist

Traditional databases excel at exact matches and structured filters. Vector databases excel at *similarity* — finding items whose meaning is close to a query, not whose tokens overlap.

```python
# Traditional database: exact match
SELECT * FROM products WHERE name ILIKE '%running shoes%'
# Finds: "Nike Running Shoes Pro"
# Misses: "jogging sneakers", "athletic footwear", "trail runners"

# Vector database: semantic similarity
results = vector_db.search(
    query_embedding=embed("running shoes"),
    top_k=5
)
# Finds all of the above — searches by MEANING, not keywords
```

### The Core Pipeline

```
User Query: "comfortable shoes for long walks"
                    |
                    v
            ┌──────────────┐
            |  Embedding    |  Convert text → vector
            |  Model        |  [0.23, -0.45, 0.67, ...]
            └──────┬───────┘
                   v
            ┌──────────────┐
            | Vector DB     |  ANN search → nearest vectors
            | (HNSW / IVF)  |
            └──────┬───────┘
                   v
            Top results + metadata:
            1. "Cushioned walking shoes with arch support" (0.92)
            2. "Best sneakers for daily commuting" (0.87)
            3. "Orthopedic footwear for extended wear" (0.85)
```

Every vector database record has three parts:

| Component | Purpose | Example |
|-----------|---------|---------|
| **ID** | Unique key for upsert/delete | `annual-report-2024::chunk-0042` |
| **Vector** | Embedding (fixed dimension at index creation) | 1536 floats for `text-embedding-3-small` |
| **Metadata** | Structured fields for filtering and display | `{"text": "...", "source": "...", "tenant_id": "acme"}` |

---

## Embeddings: A Quick Refresher

An **embedding** is a dense numerical vector that captures semantic meaning. Similar concepts produce vectors with small angles between them (high cosine similarity).

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine_sim(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

emb1 = get_embedding("The cat sat on the mat")
emb2 = get_embedding("A kitten rested on the rug")
emb3 = get_embedding("Stock prices rose sharply today")

print(f"cat/kitten: {cosine_sim(emb1, emb2):.3f}")  # ~0.85-0.92
print(f"cat/stocks: {cosine_sim(emb1, emb3):.3f}")  # ~0.15-0.25
```

Module 09 introduced this concept. In **Lesson 2** of this module we go much deeper: model selection, matryoshka dimensions, normalization, and domain-specific benchmarking.

---

## ANN: Why Brute Force Doesn't Scale

Exact nearest-neighbor search compares your query against every stored vector. At 1M vectors × 1536 dimensions, that is ~1 billion floating-point operations per query — too slow for interactive search.

**Approximate Nearest Neighbor (ANN)** algorithms trade a small amount of recall (typically 1–5%) for 10–1000× speed gains.

| Algorithm | Mechanism | Used By | Best For |
|-----------|-----------|---------|----------|
| **Flat (brute force)** | Compare all vectors | Small indexes, eval ground truth | < 50K vectors |
| **HNSW** | Multi-layer navigable graph | Pinecone, Qdrant, Weaviate | Real-time, high recall |
| **IVF** | Cluster partitions, probe nearest | Milvus, pgvector | Large batch queries |
| **IVF + PQ** | IVF + vector compression | Billion-scale | Memory-constrained |
| **DiskANN** | SSD-resident graph | Azure AI Search, Milvus | RAM > dataset size |

```
Exact Search (brute force):     ANN Search (HNSW):
Compare with ALL vectors         Navigate graph layers
O(n) — linear in corpus size    O(log n) — sublinear

1M vectors:   ~100-500ms          1M vectors:   ~1-5ms
10M vectors:  ~1-5s               10M vectors:  ~5-20ms
100M vectors: impractical         100M vectors: ~20-100ms (with tuning)
```

**Lesson 3** covers HNSW parameter tuning (`M`, `ef_construction`, `ef`), IVF `nprobe`, and Product Quantization in detail — with benchmark scripts you can run on your own data.

---

## Hands-On: Semantic vs Keyword Search

Run this experiment to see why vector databases matter for RAG:

```python
"""Compare keyword overlap vs embedding similarity on a tiny corpus."""
from openai import OpenAI
import numpy as np

client = OpenAI()
DOCS = [
    {"id": "d1", "text": "Refunds available within 30 days of purchase"},
    {"id": "d2", "text": "Standard shipping takes 5-7 business days"},
    {"id": "d3", "text": "How to return an item you no longer want"},
    {"id": "d4", "text": "Expedited delivery options and costs"},
]
QUERY = "Can I send back something I bought last week?"

def embed(text: str) -> list[float]:
    r = client.embeddings.create(model="text-embedding-3-small", input=text)
    return r.data[0].embedding

def cosine(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Keyword search: count overlapping tokens
query_tokens = set(QUERY.lower().split())
keyword_scores = [
    (doc["id"], len(query_tokens & set(doc["text"].lower().split())))
    for doc in DOCS
]
keyword_scores.sort(key=lambda x: x[1], reverse=True)

# Vector search
query_vec = embed(QUERY)
doc_vecs = [(doc["id"], embed(doc["text"])) for doc in DOCS]
vector_scores = [(doc_id, cosine(query_vec, vec)) for doc_id, vec in doc_vecs]
vector_scores.sort(key=lambda x: x[1], reverse=True)

print("Keyword top-2:", keyword_scores[:2])
print("Vector top-2:", vector_scores[:2])
# Vector search finds d3 (return policy) even though query shares zero keywords
```

The vector search retrieves `d3` ("How to return an item...") because it understands *intent*, not token overlap. This is the retrieval foundation every RAG system depends on.

---

## Database Landscape: Choosing Your Stack

| Database | Type | Best For | Key Feature |
|----------|------|----------|-------------|
| **Pinecone** | Managed cloud | Production at scale, zero ops | Serverless, namespaces, metadata filters |
| **Qdrant** | Self-hosted / cloud | High performance, Rust speed | Rich filtering, quantization, sharding |
| **Weaviate** | Self-hosted / cloud | Hybrid search (BM25 + vector) | Built-in vectorization, GraphQL API |
| **Milvus** | Self-hosted / cloud | Billion-scale, GPU | IVF+PQ, DiskANN, distributed |
| **Chroma** | Embedded / local | Prototyping, dev | In-process, auto-embeddings |
| **pgvector** | PostgreSQL extension | Existing Postgres stack | No new infra, SQL + vectors |

### Decision Matrix

```
How many vectors?
  ├─ < 100K        → Chroma or pgvector (simple, cheap)
  ├─ 100K - 10M    → Qdrant, Pinecone, or Weaviate
  └─ > 10M         → Milvus, Pinecone pods, or Qdrant cluster

Do you need hybrid search (keyword + semantic)?
  ├─ Yes           → Weaviate, Qdrant, Elasticsearch + dense vectors
  └─ No            → Any ANN-capable database

Who operates it?
  ├─ You want zero ops → Pinecone, Weaviate Cloud, Qdrant Cloud
  └─ You have infra team → Qdrant, Milvus, pgvector on RDS
```

**Lessons 4–5** walk through Pinecone and Chroma/Qdrant/Weaviate in production depth.

---

## Quick Start: Chroma and Pinecone

### Chroma (Local Prototype)

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection(
    name="my_documents",
    metadata={"hnsw:space": "cosine"}
)

collection.add(
    documents=[
        "Python is great for data science and machine learning",
        "JavaScript powers modern web applications",
        "Rust provides memory safety without garbage collection",
    ],
    ids=["doc1", "doc2", "doc3"]
)

results = collection.query(
    query_texts=["Which language is best for AI?"],
    n_results=2
)
print(results["documents"])
```

### Pinecone (Production Cloud)

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")
pc.create_index(
    name="my-index",
    dimension=1536,
    metric="cosine",
    spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
)
index = pc.Index("my-index")

index.upsert(vectors=[{
    "id": "doc1",
    "values": get_embedding("Python for data science"),
    "metadata": {"category": "programming"}
}])

results = index.query(
    vector=get_embedding("AI programming language"),
    top_k=3,
    filter={"category": "programming"},
    include_metadata=True
)
```

---

## Vector Databases in AI Applications

| Use Case | How Vectors Help | Typical Scale |
|----------|-----------------|---------------|
| **RAG** | Retrieve relevant chunks before LLM generation | 10K–100M chunks |
| **Semantic search** | Find documents/products by meaning | 1K–10M items |
| **Recommendations** | "Users who liked X also liked Y" via embedding similarity | 1M–1B items |
| **Anomaly detection** | Flag outliers far from any cluster | Streaming |
| **Multimodal search** | Image + text in shared embedding space | 100K–10M assets |

---

## Failure Modes & Common Misconceptions

**Misconception 1: Vector search replaces keyword search.**
Vector search misses exact identifiers — error codes, SKUs, legal citations. Production systems almost always combine both (Lesson 8: Hybrid Search).

**Misconception 2: Any embedding model works for your domain.**
General models underperform on specialized corpora (legal, medical, code). Always benchmark Recall@5 on your own labeled queries before choosing a model.

**Misconception 3: Higher similarity score = always relevant.**
Scores are relative to your corpus distribution. A 0.72 score might be excellent in a sparse domain and mediocre in a dense one. Calibrate thresholds on eval data.

**Misconception 4: Vector DBs are the source of truth.**
They are a *search index*, not a database of record. Always store original documents externally and treat the vector index as rebuildable.

!!! warning "Dimension mismatch is fatal"
    If you create an index with `dimension=1536` but later switch to a 3072-dimension model, every query will fail or return garbage. Version your embedding model and re-index atomically when changing models.

---

## Production Notes

- **Latency budget:** Embedding (~20ms) + ANN search (~1–50ms) + metadata fetch (~1–5ms) = typically 30–100ms added to every RAG query.
- **Index metric must match embedding training:** OpenAI embeddings use cosine; normalize vectors if using dot product.
- **Metadata filters reduce search space:** Pre-filtering by `tenant_id` or `category` before ANN search improves both latency and relevance (Lesson 6).
- **Monitor recall, not just latency:** A fast index that misses relevant documents is worse than a slower one with higher recall. Lesson 9 covers evaluation methodology.

---

## Key Takeaways

- Vector databases store embeddings and find nearest neighbors by meaning, not keywords.
- ANN algorithms (HNSW, IVF, PQ) make billion-scale search feasible with 1–5% recall trade-off.
- Choose your database based on scale, hybrid search needs, and operational capacity.
- The vector index is rebuildable — your document store is the source of truth.
- This module goes deeper than Module 09: expect HNSW tuning, multi-tenant sharding, hybrid benchmarks, and production eval harnesses in upcoming lessons.

---

## Resources

- [Pinecone: What is a Vector Database?](https://www.pinecone.io/learn/vector-database/) — Visual ANN explanations
- [ANN Benchmarks](https://ann-benchmarks.com/) — Algorithm comparison on standard datasets
- [Module 09 Lesson 02](../../module-09-rag-retrieval-augmented-generation/lessons/02-vector-databases.md) — Prerequisite overview

---

## Next Lesson

**[Lesson 2: Embeddings and Vector Representations](02-lesson-02.md)** — Deep dive into embedding models, matryoshka dimensions, normalization, and domain benchmarking beyond the Module 09 overview.
