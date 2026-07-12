---
title: Hybrid Search
description: >-
  Combine dense embeddings and sparse keyword search for robust retrieval across
  diverse query types
duration: 35 min
difficulty: intermediate
has_code: false
module: module-09
youtube: 'https://www.youtube.com/watch?v=OujMiengFaE'
---
## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand why hybrid search outperforms single-method retrieval | 35 min | Intermediate |
| Implement score fusion techniques | | |
| Configure hybrid search in vector databases | | |
| Tune alpha weights for your domain | | |

---

## The Problem with Single-Method Search

Neither dense nor sparse retrieval alone covers all real-world queries.

| Query | Dense only | Sparse only |
|-------|-----------|-------------|
| "How do I reset my password?" | Works well | May miss paraphrases |
| "Error ECONNREFUSED port 5432" | May miss exact code | Works well |
| "vacation policy 2024" | Good | Good |
| "that thing from last week's email" | Poor | Poor |

**Hybrid search** runs both methods and merges results — covering semantic and lexical matches.

---

## How Hybrid Search Works

```
                    ┌─ Dense search (embeddings) ─→ Ranked list A
User query ────────►│
                    └─ Sparse search (BM25) ───→ Ranked list B
                                    ↓
                            Fusion (RRF / weighted)
                                    ↓
                              Final top-K
```

---

## Reciprocal Rank Fusion (RRF)

RRF is robust because it doesn't require normalizing incompatible score scales.

```python
def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}

    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

**Example:**

```python
dense = ["doc_password", "doc_security", "doc_billing"]
sparse = ["doc_security", "doc_password", "doc_api_errors"]

for doc_id, score in reciprocal_rank_fusion([dense, sparse]):
    print(f"{score:.4f}  {doc_id}")
# doc_password and doc_security both rank high — agreed by both methods
```

---

## Weighted Score Fusion

When both retrievers return comparable scores (0–1 normalized):

```python
def weighted_fusion(
    dense_scores: dict[str, float],
    sparse_scores: dict[str, float],
    alpha: float = 0.5,
) -> list[tuple[str, float]]:
    all_ids = set(dense_scores) | set(sparse_scores)
    combined = {
        doc_id: alpha * dense_scores.get(doc_id, 0) + (1 - alpha) * sparse_scores.get(doc_id, 0)
        for doc_id in all_ids
    }
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)
```

| Alpha | Behavior |
|-------|----------|
| 1.0 | Dense only |
| 0.7 | Dense-heavy (default for most support KBs) |
| 0.5 | Balanced |
| 0.3 | Sparse-heavy (technical docs with codes/IDs) |
| 0.0 | Sparse only |

> **Tip:** Start with α=0.5 and tune using a labeled eval set of 50–100 queries. Track Recall@10 per query type.

---

## Hybrid in Vector Databases

Many vector DBs support hybrid search natively:

**Weaviate:**
```python
# Combines vector + BM25 in a single query
response = client.query.get("Document", ["content"]).with_hybrid(
    query="password reset",
    alpha=0.75,  # 0=BM25, 1=vector
).with_limit(5).do()
```

**Pinecone (sparse-dense vectors):**
```python
# Upload both dense and sparse vectors per document
index.upsert(vectors=[{
    "id": "doc_1",
    "values": dense_embedding,
    "sparse_values": {"indices": [...], "values": [...]},
}])
```

**Elasticsearch + dense_vector:**
```python
# kNN + BM25 in a single hybrid query (ES 8.x+)
```

---

## When to Use Hybrid

| Scenario | Recommendation |
|----------|---------------|
| Customer support FAQ | Hybrid (α ≈ 0.6) |
| API documentation | Hybrid (α ≈ 0.4) — favor exact matches |
| Narrative knowledge base | Dense-heavy (α ≈ 0.8) |
| Product catalog with SKUs | Sparse-heavy (α ≈ 0.3) |

> **Warning:** Hybrid doubles index complexity. If your eval set shows sparse adds <5% Recall@10 improvement, dense-only may be simpler to operate.

---

## Recommended Videos

- [Hybrid Search Explained](https://www.youtube.com/watch?v=OujMiengFaE)
- [Weaviate Hybrid Search](https://www.youtube.com/watch?v=3E_KFXDAeFc)

---

## Additional Resources

- [Weaviate Hybrid Search Docs](https://weaviate.io/developers/weaviate/search/hybrid)
- [Pinecone Hybrid Search](https://www.pinecone.io/learn/hybrid-search-intro/)
- [Elasticsearch Hybrid Search](https://www.elastic.co/search-labs/blog/hybrid-search-elasticsearch)
