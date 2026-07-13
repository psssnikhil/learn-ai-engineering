---
title: Embeddings and Vector Representations
description: >-
  Understand how text, images, and other data are converted into vector
  embeddings for similarity search
duration: 45 min
difficulty: intermediate
has_code: true
module: module-13
---
# Embeddings and Vector Representations

## Prerequisites

Before this lesson you should be comfortable with:

- **Module 09, Lesson 02 — Vector Databases & Embeddings** — cosine similarity, basic embedding generation, Chroma quick start
- **Module 13, Lesson 01** — vector database architecture and ANN overview
- **NumPy basics** — dot products, vector norms

Module 09 taught you *what* embeddings are. This lesson teaches you *how to choose, tune, and benchmark* embedding models for production — including matryoshka dimensions, normalization, batching economics, and domain-specific evaluation.

---

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Compare embedding models by quality, cost, and dimension | 10 min | Intermediate |
| Generate and batch embeddings efficiently at scale | 10 min | Intermediate |
| Apply matryoshka dimension reduction without re-embedding | 10 min | Intermediate |
| Run a domain benchmark script to pick the right model | 15 min | Intermediate |

---

## Intuition First: Translation to Coordinates

Think of an embedding model as a **universal translator** that converts any sentence into GPS coordinates on a semantic globe. The translator was trained on billions of text pairs — "these two sentences mean similar things, place them close together; these two are unrelated, place them far apart."

Once everything is on the globe, finding similar content is geometry: measure the angle between your query coordinate and every stored coordinate. No keyword matching required.

The catch: different translators (embedding models) use different map projections. OpenAI's `text-embedding-3-large` uses 3072 dimensions — a high-resolution map. `all-MiniLM-L6-v2` uses 384 dimensions — a compressed tourist map. Both work, but the high-resolution map captures finer distinctions at higher storage cost.

Your job in production is picking the right translator for your domain and budget, then keeping it stable across index and query time.

---

## What Are Embeddings?

An embedding is a fixed-length dense vector that encodes semantic meaning. Similar texts produce vectors with high cosine similarity (small angle between them).

```
"How to configure HNSW ef parameter"  → [0.12, -0.34, 0.87, ..., 0.05]  (1536d)
"Tuning HNSW search accuracy"         → [0.11, -0.31, 0.85, ..., 0.07]  (similar!)
"Best pizza recipes in Naples"        → [0.78,  0.33, 0.05, ..., -0.91] (different)
```

The individual numbers have no human-readable meaning. Value comes entirely from **relative position** in the space.

---

## Generating Embeddings

### OpenAI (Production Default)

```python
from openai import OpenAI

client = OpenAI()

def get_embedding(text: str, model: str = "text-embedding-3-small") -> list[float]:
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

embedding = get_embedding("What is a vector database?")
print(f"Dimensions: {len(embedding)}")  # 1536 for text-embedding-3-small
```

### Sentence Transformers (Open Source, Local)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [
    "What is a vector database?",
    "How do embeddings work?",
    "The weather is sunny today",
]
embeddings = model.encode(texts, normalize_embeddings=True)
print(f"Shape: {embeddings.shape}")  # (3, 384)
```

### Batch Embedding for Scale

Never embed one document at a time in production. Batch API calls reduce latency and cost:

```python
def batch_embed(texts: list[str], model: str = "text-embedding-3-small",
                batch_size: int = 100) -> list[list[float]]:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(input=batch, model=model)
        all_embeddings.extend(item.embedding for item in response.data)
    return all_embeddings

# 10,000 docs at batch_size=100 → 100 API calls instead of 10,000
```

| Batch Size | OpenAI Rate Limit Consideration | Typical Throughput |
|------------|--------------------------------|-------------------|
| 50 | Safe for Tier 1 accounts | ~500 docs/sec |
| 100 | Default for most pipelines | ~800 docs/sec |
| 500 | Requires higher tier | ~2000 docs/sec |

---

## Measuring Similarity

### Cosine Similarity (Standard for Text)

Cosine similarity measures the angle between two vectors, ignoring magnitude. Range: -1 to +1.

```python
import numpy as np

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

emb_1 = get_embedding("How to train a neural network")
emb_2 = get_embedding("Training deep learning models")
emb_3 = get_embedding("Best pizza recipes")

print(f"ML pair:  {cosine_similarity(emb_1, emb_2):.3f}")  # ~0.85-0.92
print(f"ML/pizza: {cosine_similarity(emb_1, emb_3):.3f}")  # ~0.10-0.20
```

### Distance Metrics and Index Compatibility

| Metric | Formula | When to Use | Index Setting |
|--------|---------|-------------|---------------|
| **Cosine** | cos(θ) = a·b / (‖a‖‖b‖) | Text embeddings (default) | `metric="cosine"` |
| **Dot product** | a·b | Pre-normalized vectors (fastest) | `metric="dotproduct"` |
| **Euclidean (L2)** | ‖a - b‖ | When magnitude carries signal | `metric="euclidean"` |

!!! note "Normalize for dot product"
    If you use dot product as your index metric, L2-normalize all vectors at ingest and query time. For unit vectors, dot product equals cosine similarity but computes faster on some hardware.

```python
def normalize(v: list[float]) -> list[float]:
    arr = np.array(v)
    return (arr / np.linalg.norm(arr)).tolist()
```

---

## Embedding Model Comparison

| Model | Dimensions | MTEB Score (avg) | Cost | Best For |
|-------|-----------|-----------------|------|----------|
| **text-embedding-3-small** | 1536 (or 256–1536) | ~62.3 | $0.02/1M tokens | Default production choice |
| **text-embedding-3-large** | 3072 (or 256–3072) | ~64.6 | $0.13/1M tokens | High-precision domains |
| **Cohere embed-v3** | 1024 | ~64.5 | $0.10/1M tokens | Multilingual, reranking pairs |
| **all-MiniLM-L6-v2** | 384 | ~56.3 | Free (local) | Prototyping, edge deployment |
| **BGE-large-en-v1.5** | 1024 | ~63.5 | Free (local) | Self-hosted production |
| **voyage-3** | 1024 | ~65.5 | $0.06/1M tokens | Code and technical docs |

Scores from [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — always re-benchmark on *your* data.

---

## Matryoshka Embeddings: Dimension Reduction Without Re-Embedding

OpenAI's `text-embedding-3-*` models support **Matryoshka Representation Learning (MRL)**: the first N dimensions of a larger embedding retain most of the semantic information.

```python
def truncate_embedding(embedding: list[float], dimensions: int) -> list[float]:
    """Use first N dimensions — valid for text-embedding-3 models."""
    truncated = embedding[:dimensions]
    # Re-normalize after truncation for cosine search
    arr = np.array(truncated)
    return (arr / np.linalg.norm(arr)).tolist()

full = get_embedding("Refund policy for enterprise customers", model="text-embedding-3-small")
compact = truncate_embedding(full, dimensions=256)

print(f"Full: {len(full)}d, Compact: {len(compact)}d")
# 256d uses 6× less storage with ~2-4% recall loss on most domains
```

| Dimensions | Storage per Vector | Typical Recall@5 Loss vs Full |
|------------|-------------------|------------------------------|
| 1536 (full) | 6,144 bytes | Baseline |
| 512 | 2,048 bytes | ~1% |
| 256 | 1,024 bytes | ~2-4% |
| 128 | 512 bytes | ~5-8% |

This is one of the highest-ROI optimizations in production: cut memory and index cost 6× with minimal quality loss.

---

## Hands-On: Domain Benchmark Script

Don't trust leaderboard scores. Run this benchmark on 20–50 labeled query-document pairs from your domain:

```python
"""Benchmark embedding models on your labeled eval set."""
import numpy as np
from openai import OpenAI

client = OpenAI()

EVAL_SET = [
    {
        "query": "How do I return a defective product?",
        "relevant_ids": {"doc-returns-policy", "doc-warranty-claims"},
    },
    {
        "query": "HNSW ef parameter tuning",
        "relevant_ids": {"doc-hnsw-guide", "doc-index-tuning"},
    },
    # Add 20-50 pairs from your domain
]

CORPUS = {
    "doc-returns-policy": "Returns accepted within 30 days...",
    "doc-warranty-claims": "Defective items covered under warranty...",
    "doc-hnsw-guide": "HNSW ef controls search-time accuracy...",
    "doc-index-tuning": "Increase ef for higher recall at query time...",
    "doc-shipping": "Standard delivery takes 5-7 business days...",
}

MODELS = ["text-embedding-3-small", "text-embedding-3-large"]

def embed(text: str, model: str) -> np.ndarray:
    r = client.embeddings.create(input=text, model=model)
    return np.array(r.data[0].embedding)

def recall_at_k(query_vec, corpus_vecs: dict, relevant: set, k: int = 5) -> float:
    scores = [
        (doc_id, float(np.dot(query_vec, vec) / (np.linalg.norm(query_vec) * np.linalg.norm(vec))))
        for doc_id, vec in corpus_vecs.items()
    ]
    scores.sort(key=lambda x: x[1], reverse=True)
    retrieved = {doc_id for doc_id, _ in scores[:k]}
    return len(retrieved & relevant) / len(relevant) if relevant else 0.0

for model in MODELS:
    corpus_vecs = {doc_id: embed(text, model) for doc_id, text in CORPUS.items()}
    recalls = []
    for item in EVAL_SET:
        q_vec = embed(item["query"], model)
        recalls.append(recall_at_k(q_vec, corpus_vecs, item["relevant_ids"], k=3))
    print(f"{model}: mean recall@3 = {np.mean(recalls):.3f}")
```

Run this before every embedding model change. A 15% recall improvement from `3-small` → `3-large` may or may not justify 6.5× the cost for your specific corpus.

---

## Failure Modes

**Index/query model mismatch.** Embedding the corpus with `text-embedding-3-small` but queries with `text-embedding-3-large` destroys retrieval quality. Lock model name in config and validate at startup.

**Forgetting normalization.** Mixing normalized and unnormalized vectors in a dot-product index produces random rankings. Normalize consistently or use cosine metric.

**Embedding stale content.** If documents update but embeddings don't, search returns outdated chunks. Tie embedding version to content hash (Lesson 10 covers change detection).

**Over-long inputs.** Most models truncate at 8192 tokens. Long documents must be chunked *before* embedding — never embed a full 50-page PDF as one vector.

**Language mismatch.** English-trained models underperform on multilingual corpora. Use Cohere embed-v3 or multilingual-e5 for non-English content.

---

## Production Notes

- **Cache embeddings by content hash:** Identical chunks across documents should not be re-embedded. SHA-256 of text → cached vector.
- **Version your embedding model in metadata:** Store `embedding_model: "text-embedding-3-small-v1"` on every vector record for audit and re-index planning.
- **Rate limit handling:** Wrap batch embed calls with exponential backoff. A 429 during bulk ingest should pause, not crash the pipeline.
- **Cost tracking:** At $0.02/1M tokens, embedding 1M chunks (~500 tokens each) costs ~$10. Budget this upfront.

---

## Key Takeaways

- Embeddings map text to coordinates where semantic similarity equals geometric proximity.
- Always batch embed in production; never one-at-a-time API calls.
- Matryoshka truncation (256d from 1536d) cuts storage 6× with ~2-4% recall loss.
- Benchmark on your domain data — leaderboard scores are a starting point, not a decision.
- Lock embedding model version across index and query; mismatch is a silent killer.

---

## Resources

- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [Sentence Transformers](https://www.sbert.net/)

---

## Next Lesson

**[Lesson 3: Indexing Strategies for Vector Search](03-lesson-03.md)** — HNSW tuning, IVF nprobe sweeps, Product Quantization, and benchmark scripts that measure recall vs latency trade-offs.
