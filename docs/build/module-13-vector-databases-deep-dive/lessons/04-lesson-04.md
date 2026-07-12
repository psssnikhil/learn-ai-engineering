---
title: Working with Pinecone
description: >-
  Learn to build and query a production vector database using Pinecone's managed
  service
duration: 40 min
difficulty: intermediate
has_code: false
module: module-13
---
# Working with Pinecone

## Learning Objectives

By the end of this lesson, you will be able to:
- Set up a Pinecone index and understand its configuration options
- Insert, query, and delete vectors with metadata
- Implement filtered similarity search
- Understand Pinecone's architecture and pricing model

---

## What is Pinecone?

Pinecone is a fully managed vector database designed for production AI applications. It handles indexing, scaling, and infrastructure so you can focus on your application logic.

### Why Pinecone?

| Feature | Pinecone | Self-hosted (Faiss/Milvus) |
|---------|----------|---------------------------|
| Setup time | Minutes | Hours to days |
| Scaling | Automatic | Manual |
| Maintenance | None | You manage it |
| Filtered search | Built-in | Custom implementation |
| Cost | Pay-per-use | Server costs |

---

## Setting Up Pinecone

```python
from pinecone import Pinecone, ServerlessSpec

# Initialize client
pc = Pinecone(api_key="your-api-key")

# Create an index
pc.create_index(
    name="my-knowledge-base",
    dimension=1536,            # Match your embedding model
    metric="cosine",           # cosine, euclidean, or dotproduct
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# Connect to the index
index = pc.Index("my-knowledge-base")
```

---

## Inserting Vectors (Upsert)

```python
from openai import OpenAI

openai_client = OpenAI()

def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        input=text, model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Prepare documents with metadata
documents = [
    {
        "id": "doc-1",
        "text": "RAG systems combine retrieval with generation for accurate answers",
        "metadata": {"category": "rag", "source": "textbook", "year": 2024}
    },
    {
        "id": "doc-2",
        "text": "Fine-tuning adapts pre-trained models to specific domains",
        "metadata": {"category": "fine-tuning", "source": "blog", "year": 2024}
    },
    {
        "id": "doc-3",
        "text": "Vector databases enable semantic search over embeddings",
        "metadata": {"category": "vector-db", "source": "docs", "year": 2023}
    },
]

# Upsert vectors with metadata
vectors_to_upsert = []
for doc in documents:
    embedding = get_embedding(doc["text"])
    vectors_to_upsert.append({
        "id": doc["id"],
        "values": embedding,
        "metadata": {**doc["metadata"], "text": doc["text"]}
    })

index.upsert(vectors=vectors_to_upsert)
print(f"Upserted {len(vectors_to_upsert)} vectors")
```

---

## Querying Vectors

```python
# Simple similarity search
query = "How do retrieval systems work?"
query_embedding = get_embedding(query)

results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True
)

for match in results["matches"]:
    print(f"Score: {match['score']:.3f} | {match['metadata']['text']}")
```

### Filtered Queries

```python
# Search only within a specific category
results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True,
    filter={
        "category": {"$eq": "rag"},
        "year": {"$gte": 2024}
    }
)
```

### Pinecone Filter Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `$eq` | Equals | `{"category": {"$eq": "rag"}}` |
| `$ne` | Not equals | `{"source": {"$ne": "blog"}}` |
| `$gt`, `$gte` | Greater than | `{"year": {"$gte": 2024}}` |
| `$lt`, `$lte` | Less than | `{"year": {"$lt": 2023}}` |
| `$in` | In list | `{"category": {"$in": ["rag", "vector-db"]}}` |

---

## Batch Operations

```python
# Batch upsert for large datasets
def batch_upsert(index, vectors, batch_size=100):
    """Upsert vectors in batches to avoid API limits."""
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
    print(f"Upserted {len(vectors)} vectors in {len(vectors)//batch_size + 1} batches")

# Delete vectors
index.delete(ids=["doc-1", "doc-2"])

# Delete by metadata filter
index.delete(filter={"category": {"$eq": "outdated"}})

# Get index statistics
stats = index.describe_index_stats()
print(f"Total vectors: {stats['total_vector_count']}")
```

---

## Key Takeaways

- Pinecone provides a managed vector database that handles scaling and infrastructure automatically
- Upsert combines insert and update -- same ID overwrites the existing vector
- Metadata filters enable combining semantic search with structured queries
- Batch operations are essential for loading large datasets efficiently
- Choose the metric (cosine, euclidean, dotproduct) based on your embedding model

## Resources

- [Pinecone Documentation](https://docs.pinecone.io/) -- Official guides and API reference
- [YouTube: Pinecone in 10 Minutes](https://www.youtube.com/watch?v=dRUIGgNBvVk) -- Quick start tutorial by James Briggs
- [Pinecone Examples](https://github.com/pinecone-io/examples) -- Official example notebooks

---

Next: Working with ChromaDB and Open-Source Alternatives
