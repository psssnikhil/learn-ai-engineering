---
title: Vector Database Evaluation and Testing
description: >-
  Learn to measure and improve the quality of your vector search system with
  recall, precision, and end-to-end evaluation
duration: 35 min
difficulty: intermediate
has_code: false
---
# Vector Database Evaluation and Testing

## Learning Objectives

By the end of this lesson, you will be able to:
- Measure retrieval quality with recall@k and precision@k
- Build evaluation datasets for your vector search system
- Run A/B tests between different configurations
- Monitor search quality in production

---

## Why Evaluate Vector Search?

Changing your embedding model, chunking strategy, or index parameters can dramatically affect search quality. Without measurement, you are guessing.

| What Changed | Potential Impact |
|-------------|-----------------|
| Embedding model upgrade | +15% recall or -10% if mismatched to your data |
| Chunk size 512 → 256 | Better precision, worse context |
| HNSW ef: 50 → 200 | +3% recall, +40ms latency |
| Added metadata filters | May reduce result pool too aggressively |

---

## Key Metrics

### Recall@k

"Of all relevant documents, how many did we retrieve in the top k?"

```python
def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """What fraction of relevant docs appear in the top-k results?"""
    retrieved_top_k = set(retrieved[:k])
    if not relevant:
        return 0.0
    return len(retrieved_top_k & relevant) / len(relevant)

# Example
retrieved = ["doc-3", "doc-1", "doc-7", "doc-5", "doc-2"]
relevant = {"doc-1", "doc-2", "doc-4"}

print(recall_at_k(retrieved, relevant, k=3))  # 1/3 = 0.33
print(recall_at_k(retrieved, relevant, k=5))  # 2/3 = 0.67
```

### Precision@k

"Of the top k results we returned, how many are actually relevant?"

```python
def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """What fraction of top-k results are relevant?"""
    retrieved_top_k = set(retrieved[:k])
    if k == 0:
        return 0.0
    return len(retrieved_top_k & relevant) / k

print(precision_at_k(retrieved, relevant, k=3))  # 1/3 = 0.33
print(precision_at_k(retrieved, relevant, k=5))  # 2/5 = 0.40
```

### Mean Reciprocal Rank (MRR)

"How high does the first relevant result appear?"

```python
def mean_reciprocal_rank(queries_results: list[tuple[list[str], set[str]]]) -> float:
    """Average of 1/rank of first relevant result across queries."""
    rr_sum = 0.0
    for retrieved, relevant in queries_results:
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                rr_sum += 1.0 / rank
                break
    return rr_sum / len(queries_results) if queries_results else 0.0
```

---

## Building an Evaluation Dataset

```python
# An evaluation dataset is a list of (query, expected_relevant_docs) pairs
eval_dataset = [
    {
        "query": "How does attention mechanism work in transformers?",
        "relevant_docs": {"doc-attention-101", "doc-transformer-arch", "doc-self-attention"}
    },
    {
        "query": "Python error handling best practices",
        "relevant_docs": {"doc-python-exceptions", "doc-error-patterns"}
    },
    {
        "query": "How to deploy ML models to production",
        "relevant_docs": {"doc-mlops-deploy", "doc-model-serving", "doc-docker-ml"}
    },
]

def evaluate_search(search_fn, eval_dataset: list[dict], k: int = 5) -> dict:
    """Evaluate a search function against a labeled dataset."""
    recalls, precisions = [], []

    for item in eval_dataset:
        results = search_fn(item["query"], k=k)
        retrieved_ids = [r["id"] for r in results]
        relevant = set(item["relevant_docs"])

        recalls.append(recall_at_k(retrieved_ids, relevant, k))
        precisions.append(precision_at_k(retrieved_ids, relevant, k))

    return {
        f"recall@{k}": sum(recalls) / len(recalls),
        f"precision@{k}": sum(precisions) / len(precisions),
    }
```

---

## A/B Testing Configurations

```python
configs = {
    "baseline": {"model": "text-embedding-3-small", "chunk_size": 512},
    "large_model": {"model": "text-embedding-3-large", "chunk_size": 512},
    "small_chunks": {"model": "text-embedding-3-small", "chunk_size": 256},
}

for name, config in configs.items():
    # Build index with this config
    search_fn = build_search(config)
    metrics = evaluate_search(search_fn, eval_dataset, k=5)
    print(f"{name}: recall@5={metrics['recall@5']:.3f}, precision@5={metrics['precision@5']:.3f}")
```

---

## Production Monitoring

Track these metrics continuously in production:

| Metric | What It Tells You | Alert Threshold |
|--------|-------------------|-----------------|
| **Average similarity score** | Are results getting less relevant? | Drop > 10% from baseline |
| **Zero-result rate** | Are queries failing to match? | > 5% of queries |
| **Latency p95** | Is search slowing down? | > 200ms |
| **Result click-through** | Are users finding results useful? | Drop > 15% from baseline |

---

## Key Takeaways

- Always measure search quality before and after changes to your pipeline
- Recall@k tells you if relevant documents are being found; Precision@k tells you if results are clean
- Build a labeled evaluation dataset early -- even 50 query-document pairs is valuable
- A/B test embedding models, chunk sizes, and index parameters systematically
- Monitor search quality in production to catch degradation early

## Resources

- [YouTube: Evaluating RAG Systems](https://www.youtube.com/watch?v=grfbBQvMlBk) -- End-to-end evaluation walkthrough
- [RAGAS Framework](https://docs.ragas.io/) -- Automated RAG evaluation toolkit
- [BEIR Benchmark](https://github.com/beir-cellar/beir) -- Standard information retrieval benchmark suite

---

Next: Production Vector Database Patterns
