---
title: Introduction to Vector Databases
description: >-
  Understand what vector databases are, how they work, and why they are
  essential for modern AI applications like RAG and semantic search
duration: 40 min
difficulty: intermediate
has_code: false
---
# Introduction to Vector Databases

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand what vectors and embeddings are | 40 min | Intermediate |
| Learn why traditional databases fall short for AI | | |
| Explore how vector databases store and search data | | |
| Compare popular vector database solutions | | |

---

## Why Vector Databases?

Traditional databases search by exact matches or keyword patterns. But AI applications need to search by **meaning**.

```python
# Traditional database: exact match
SELECT * FROM products WHERE name = 'running shoes'
# Finds: "running shoes"
# Misses: "jogging sneakers", "athletic footwear", "trail runners"

# Vector database: semantic similarity
results = vector_db.search(
    query_embedding=embed("running shoes"),
    top_k=5
)
# Finds: "running shoes", "jogging sneakers", "athletic footwear",
#         "trail runners", "marathon trainers"
# Searches by MEANING, not keywords!
```

### The Core Problem Vector Databases Solve

```
User Query: "comfortable shoes for long walks"
                    |
                    v
            ┌──────────────┐
            |  Embedding    |  Convert text to a vector
            |  Model        |  [0.23, -0.45, 0.67, ...]
            └──────┬───────┘
                   v
            ┌──────────────┐
            | Vector DB     |  Find nearest vectors
            | (ANN Search)  |  (= most similar meanings)
            └──────┬───────┘
                   v
            Top results:
            1. "Cushioned walking shoes with arch support" (0.92 similarity)
            2. "Best sneakers for daily commuting" (0.87 similarity)
            3. "Orthopedic footwear for extended wear" (0.85 similarity)
```

---

## What Are Embeddings?

An **embedding** is a list of numbers (a vector) that represents the meaning of text, images, or other data. Similar items have similar vectors.

```python
from openai import OpenAI
client = OpenAI()

# Generate embeddings
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding  # Returns list of 1536 floats

# Similar sentences produce similar vectors
emb1 = get_embedding("The cat sat on the mat")
emb2 = get_embedding("A kitten rested on the rug")
emb3 = get_embedding("Stock prices rose sharply today")

# Cosine similarity
from numpy import dot
from numpy.linalg import norm

def cosine_sim(a, b):
    return dot(a, b) / (norm(a) * norm(b))

print(cosine_sim(emb1, emb2))  # ~0.89 (very similar - both about cats on surfaces)
print(cosine_sim(emb1, emb3))  # ~0.21 (very different - unrelated topics)
```

---

## How Vector Databases Work

### Storage

Vector databases store each record as:
- A **vector** (the embedding)
- **Metadata** (any additional fields: text, category, date, etc.)
- An **ID** for retrieval

```python
# What gets stored in a vector database
record = {
    "id": "doc-001",
    "vector": [0.23, -0.45, 0.67, ...],  # 1536 dimensions
    "metadata": {
        "text": "Python is a versatile programming language...",
        "source": "python-docs",
        "category": "programming",
        "date": "2025-01-15"
    }
}
```

### Search: Approximate Nearest Neighbors (ANN)

Exact search across millions of vectors would be too slow. Vector databases use **ANN algorithms** that trade a tiny bit of accuracy for massive speed gains.

| Algorithm | Used By | How It Works |
|-----------|---------|-------------|
| **HNSW** (Hierarchical Navigable Small World) | Pinecone, Qdrant, Weaviate | Graph-based: builds navigable graph layers |
| **IVF** (Inverted File Index) | Milvus, pgvector | Partition-based: clusters vectors, searches relevant clusters |
| **ScaNN** | Google Vertex AI | Quantization + partitioning for speed |

```
Exact Search (brute force):     ANN Search (HNSW):
Compare with ALL vectors         Navigate graph to neighborhood
O(n) - slow at scale            O(log n) - fast at any scale

1M vectors: ~100ms               1M vectors: ~1ms
10M vectors: ~1000ms             10M vectors: ~5ms
```

---

## Popular Vector Databases Compared

| Database | Type | Best For | Key Feature |
|----------|------|----------|-------------|
| **Pinecone** | Managed cloud | Production at scale | Fully managed, zero ops |
| **Chroma** | Embedded / local | Prototyping, small apps | Simple API, runs in-process |
| **Qdrant** | Self-hosted / cloud | Production with control | Rich filtering, Rust-based |
| **Weaviate** | Self-hosted / cloud | Multimodal search | Built-in vectorization |
| **Milvus** | Self-hosted / cloud | Large-scale deployments | GPU-accelerated search |
| **pgvector** | PostgreSQL extension | Existing Postgres users | No new infrastructure |

### Quick Start with Chroma (Local)

```python
import chromadb

# Create a client (in-memory for prototyping)
client = chromadb.Client()

# Create a collection
collection = client.create_collection(
    name="my_documents",
    metadata={"hnsw:space": "cosine"}  # Use cosine similarity
)

# Add documents (Chroma auto-generates embeddings)
collection.add(
    documents=[
        "Python is great for data science and machine learning",
        "JavaScript powers modern web applications",
        "Rust provides memory safety without garbage collection",
        "Go excels at building concurrent network services",
    ],
    ids=["doc1", "doc2", "doc3", "doc4"]
)

# Search by meaning
results = collection.query(
    query_texts=["Which language is best for AI?"],
    n_results=2
)
print(results["documents"])
# [["Python is great for data science and machine learning",
#   "Rust provides memory safety without garbage collection"]]
```

### Quick Start with Pinecone (Cloud)

```python
from pinecone import Pinecone

pc = Pinecone(api_key="your-api-key")

# Create an index
pc.create_index(
    name="my-index",
    dimension=1536,  # Must match your embedding model
    metric="cosine",
    spec={"serverless": {"cloud": "aws", "region": "us-east-1"}}
)

index = pc.Index("my-index")

# Upsert vectors with metadata
index.upsert(vectors=[
    {
        "id": "doc1",
        "values": get_embedding("Python for data science"),
        "metadata": {"category": "programming", "language": "python"}
    },
    {
        "id": "doc2",
        "values": get_embedding("JavaScript web development"),
        "metadata": {"category": "programming", "language": "javascript"}
    }
])

# Query with metadata filtering
results = index.query(
    vector=get_embedding("AI programming language"),
    top_k=3,
    filter={"category": "programming"},
    include_metadata=True
)
```

---

## Vector Databases in AI Applications

### 1. RAG (Retrieval Augmented Generation)

The most common use case: give LLMs access to your data.

```
User question -> Embed question -> Search vector DB
    -> Get relevant documents -> Pass to LLM as context
    -> LLM generates grounded answer
```

### 2. Semantic Search

Search by meaning across documents, products, support tickets, etc.

### 3. Recommendation Systems

Find items similar to what a user liked based on embedding similarity.

### 4. Anomaly Detection

Detect outliers by finding data points far from any cluster in embedding space.

---

## Key Takeaways

- Vector databases store and search high-dimensional embeddings
- They enable search by meaning (semantic similarity) rather than keywords
- ANN algorithms like HNSW make billion-scale search fast
- Choose your database based on scale, hosting preference, and features
- Vector databases are the backbone of RAG, semantic search, and recommendations

---

## Next Lesson

**Lesson 2: Indexing Strategies and Performance** - Learn how to optimize your vector database for speed, accuracy, and cost.
