---
title: Vector Databases & Embeddings
description: Master embeddings and vector similarity search - the foundation of RAG
duration: 35 min
difficulty: intermediate
has_code: false
module: module-09
youtube: 'https://www.youtube.com/watch?v=klTvEwg3oJ4'
---
# Vector Databases & Embeddings

## What are Embeddings?

**Embeddings**: Convert text to numbers (vectors) that capture meaning

```
"cat" → [0.2, -0.5, 0.8, ..., 0.3]  (1536 dimensions)
"dog" → [0.3, -0.4, 0.7, ..., 0.2]  (similar to cat!)
"car" → [-0.8, 0.9, -0.2, ..., 0.1] (very different)
```

**Key property**: Similar meanings = Similar vectors

## Creating Embeddings

```python
from openai import OpenAI

client = OpenAI()

def embed(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",  # 1536 dimensions
        input=text
    )
    return response.data[0].embedding

# Create embeddings
cat_vec = embed("cat")
dog_vec = embed("dog")
car_vec = embed("car")

print(f"Embedding dimension: {len(cat_vec)}")  # 1536
```

## Similarity Search

**Cosine similarity**: Measure how similar two vectors are

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Compare
cat_dog = cosine_similarity(cat_vec, dog_vec)  # 0.85 (similar!)
cat_car = cosine_similarity(cat_vec, car_vec)  # 0.15 (different!)

print(f"cat-dog similarity: {cat_dog}")
print(f"cat-car similarity: {cat_car}")
```

## Vector Databases

**Purpose**: Store and search millions of embeddings efficiently

**Popular options**:
- **Pinecone**: Managed, easy to use
- **Weaviate**: Open source, GraphQL
- **Qdrant**: Fast, Rust-based
- **ChromaDB**: Simple, local
- **Milvus**: Enterprise-grade
- **pgvector**: PostgreSQL extension

## ChromaDB Example

```python
import chromadb

# Initialize
client = chromadb.Client()
collection = client.create_collection("documents")

# Add documents (auto-embeds!)
collection.add(
    documents=[
        "The cat sat on the mat",
        "The dog played in the park",
        "Python is a programming language"
    ],
    ids=["doc1", "doc2", "doc3"]
)

# Query
results = collection.query(
    query_texts=["Tell me about animals"],
    n_results=2
)

print(results['documents'])
# Returns: ["The cat sat on the mat", "The dog played in the park"]
```

## Pinecone Example

```python
from pinecone import Pinecone

# Initialize
pc = Pinecone(api_key="your-key")
index = pc.Index("my-index")

# Upsert vectors
index.upsert(vectors=[
    ("doc1", cat_vec, {"text": "The cat sat on the mat"}),
    ("doc2", dog_vec, {"text": "The dog played"}),
])

# Query
query_vec = embed("animals")
results = index.query(
    vector=query_vec,
    top_k=2,
    include_metadata=True
)

for match in results['matches']:
    print(f"Score: {match['score']}, Text: {match['metadata']['text']}")
```

## Indexing Strategies

### 1. HNSW (Hierarchical Navigable Small World)
- Fast approximate search
- Most popular
- O(log n) complexity

### 2. IVF (Inverted File Index)
- Cluster vectors
- Good for large datasets

### 3. Flat (Exact Search)
- 100% accurate
- Slow for large datasets

## Embedding Models Comparison

| Model | Dimensions | Max Tokens | Cost |
|-------|-----------|------------|------|
| text-embedding-3-small | 1536 | 8191 | $0.02/1M |
| text-embedding-3-large | 3072 | 8191 | $0.13/1M |
| Cohere embed-v3 | 1024 | 512 | $0.10/1M |
| Sentence-Transformers | 384-768 | 512 | Free (self-host) |

---

## 📹 Recommended Videos

- [Vector Databases Simply Explained](https://www.youtube.com/watch?v=dN0lsF2cvm4) — Fireship quick overview
- [What are Vector Embeddings?](https://www.youtube.com/watch?v=ySus5ZS0b94) — Weaviate visual explanation
- [Pinecone Vector Database Tutorial](https://www.youtube.com/watch?v=Ff6P_JFJwEk) — James Briggs hands-on walkthrough

---

## 📚 Additional Resources

- [What is a Vector Database?](https://www.pinecone.io/learn/vector-database/) — Pinecone learning center
- [Vector Similarity Search](https://weaviate.io/developers/weaviate/concepts/vector-index) — Weaviate concepts guide
- [Choosing a Vector Database](https://thenewstack.io/comparing-vector-databases-a-hands-on-review/) — The New Stack hands-on comparison

---
