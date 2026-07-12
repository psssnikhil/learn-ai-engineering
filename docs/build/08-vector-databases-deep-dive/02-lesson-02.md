---
title: Embeddings and Vector Representations
description: >-
  Understand how text, images, and other data are converted into vector
  embeddings for similarity search
duration: 30 min
difficulty: intermediate
has_code: false
objectives:
  - Explain what an embedding is and why it enables similarity search
  - Generate text embeddings using the OpenAI API
  - Calculate cosine similarity between embedding vectors
  - Choose the right embedding model for your use case
---
# Embeddings and Vector Representations

## Learning Objectives

By the end of this lesson, you will be able to:
- Explain how embedding models convert text into numerical vectors
- Generate embeddings using the OpenAI and open-source models
- Calculate similarity between vectors using cosine similarity
- Understand the trade-offs between embedding models

---

## What Are Embeddings?

An embedding is a dense numerical vector that captures the semantic meaning of data. Similar concepts produce vectors that are close together in high-dimensional space.

```
"king"   → [0.21, -0.45, 0.87, 0.12, ...]   (1536 dimensions)
"queen"  → [0.19, -0.42, 0.85, 0.14, ...]   (similar!)
"banana" → [0.78,  0.33, 0.05, -0.91, ...]   (very different)
```

This means we can find related content by finding vectors that are close to each other -- without keyword matching.

---

## Generating Embeddings

### Using OpenAI

```python
from openai import OpenAI

client = OpenAI()

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Generate an embedding vector for the given text."""
    response = client.embeddings.create(
        input=text,
        model=model,
    )
    return response.data[0].embedding

# Generate embeddings
embedding = get_embedding("What is a vector database?")
print(f"Dimensions: {len(embedding)}")  # 1536 for text-embedding-3-small
```

### Using Sentence Transformers (Open Source)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [
    "What is a vector database?",
    "How do embeddings work?",
    "The weather is sunny today",
]

embeddings = model.encode(texts)
print(f"Shape: {embeddings.shape}")  # (3, 384)
```

---

## Measuring Similarity

### Cosine Similarity

The most common similarity metric for embeddings. It measures the angle between two vectors, ignoring magnitude.

```python
import numpy as np

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Compare sentences
emb_1 = get_embedding("How to train a neural network")
emb_2 = get_embedding("Training deep learning models")
emb_3 = get_embedding("Best pizza recipes")

print(cosine_similarity(emb_1, emb_2))  # ~0.89 (very similar)
print(cosine_similarity(emb_1, emb_3))  # ~0.15 (very different)
```

### Other Distance Metrics

| Metric | Formula | Best For |
|--------|---------|----------|
| **Cosine similarity** | cos(a, b) | Text similarity (most common) |
| **Euclidean (L2)** | sqrt(sum((a-b)^2)) | When magnitude matters |
| **Dot product** | sum(a*b) | Normalized vectors, speed |
| **Manhattan (L1)** | sum(abs(a-b)) | Sparse, high-dimensional data |

---

## Embedding Model Comparison

| Model | Dimensions | Speed | Quality | Cost |
|-------|-----------|-------|---------|------|
| **text-embedding-3-small** (OpenAI) | 1536 | Fast | Good | $0.02/1M tokens |
| **text-embedding-3-large** (OpenAI) | 3072 | Medium | Best | $0.13/1M tokens |
| **all-MiniLM-L6-v2** (open source) | 384 | Very fast | Good | Free |
| **BGE-large-en** (BAAI) | 1024 | Medium | Very good | Free |
| **Cohere embed-v3** | 1024 | Fast | Very good | $0.10/1M tokens |

---

## Batch Embedding for Efficiency

```python
def batch_embed(texts: list[str], model: str = "text-embedding-3-small",
                batch_size: int = 100) -> list[list[float]]:
    """Embed texts in batches for efficiency."""
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(input=batch, model=model)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings

# Embed 1000 documents efficiently
# embeddings = batch_embed(documents)
```

---

## Key Takeaways

- Embeddings convert text into numerical vectors that capture semantic meaning
- Similar texts produce vectors that are close together in vector space
- Cosine similarity is the standard metric for comparing text embeddings
- OpenAI's text-embedding-3-small offers a good balance of quality, speed, and cost
- Open-source models like MiniLM are free and work well for many use cases

## Resources

- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings) -- Official documentation
- [Sentence Transformers](https://www.sbert.net/) -- Open-source embedding models
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) -- Benchmark for embedding models

---

Next: Indexing Strategies
