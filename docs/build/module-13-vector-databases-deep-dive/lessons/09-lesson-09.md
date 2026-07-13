---
title: Vector Database Evaluation and Testing
description: >-
  Learn to measure and improve the quality of your vector search system with
  recall, precision, and end-to-end evaluation
duration: 50 min
difficulty: intermediate
has_code: true
module: module-13
---
# Vector Database Evaluation and Testing

## Prerequisites

Before this lesson you should be comfortable with:

- **Module 09, Lesson 08 — RAG Evaluation Metrics** — basic recall/precision concepts
- **Module 13, Lessons 01–08** — indexing, hybrid search, embedding models
- **Python** — running evaluation scripts

You cannot improve what you don't measure. This lesson provides a complete eval methodology: building labeled datasets, measuring ANN recall vs exact search, comparing configurations with A/B tests, and monitoring quality in production.

---

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Compute recall@K, precision@K, MRR, and nDCG | 10 min | Intermediate |
| Build a labeled evaluation dataset for your domain | 10 min | Intermediate |
| Measure ANN index recall against brute-force ground truth | 15 min | Intermediate |
| Run A/B tests comparing embedding models, indexes, and hybrid configs | 15 min | Intermediate |

---

## Intuition First: The Taste Test

You changed the recipe (new embedding model, different chunk size, higher HNSW ef). Does the dish taste better? Without a taste test, you're guessing.

An eval dataset is your panel of judges: 50–200 queries with known correct answers. Every configuration change gets scored against the same panel. Recall@5 tells you "did the right documents appear in the top 5?" MRR tells you "how high was the first correct result?"

Production without eval is flying blind. You won't notice slow degradation until users complain.

---

## Why Evaluate Vector Search?

| Change | Potential Impact | Without Eval |
|--------|-----------------|--------------|
| Embedding model upgrade | +15% recall or -10% if mismatched | Guess which happened |
| Chunk size 512 → 256 | Better precision, worse context | Unknown trade-off |
| HNSW ef: 50 → 200 | +3% recall, +40ms latency | Can't justify latency cost |
| Added metadata filters | May shrink candidate pool too much | Silent recall drop |
| Switched to hybrid (alpha=0.5) | +10-20% on mixed queries | Hope it's better |
| Matryoshka 256d truncation | -3% recall, 6× memory savings | Unknown if acceptable |

---

## Core Metrics

### Recall@K

"Of all relevant documents, what fraction appear in the top K results?"

```python
def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    retrieved_top_k = set(retrieved[:k])
    if not relevant:
        return 0.0
    return len(retrieved_top_k & relevant) / len(relevant)

retrieved = ["doc-3", "doc-1", "doc-7", "doc-5", "doc-2"]
relevant = {"doc-1", "doc-2", "doc-4"}

print(f"Recall@3: {recall_at_k(retrieved, relevant, 3):.2f}")  # 0.33
print(f"Recall@5: {recall_at_k(retrieved, relevant, 5):.2f}")  # 0.67
```

**When to optimize recall:** RAG systems where missing the right document means a wrong or "I don't know" answer. Target: Recall@5 ≥ 0.85 on your eval set.

### Precision@K

"Of the top K results returned, how many are actually relevant?"

```python
def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    retrieved_top_k = set(retrieved[:k])
    if k == 0:
        return 0.0
    return len(retrieved_top_k & relevant) / k

print(f"Precision@3: {precision_at_k(retrieved, relevant, 3):.2f}")  # 0.33
print(f"Precision@5: {precision_at_k(retrieved, relevant, 5):.2f}")  # 0.40
```

**When to optimize precision:** User-facing search UIs where showing irrelevant results erodes trust. Target: Precision@5 ≥ 0.70.

### Mean Reciprocal Rank (MRR)

"How high does the first relevant result appear, on average?"

```python
def mean_reciprocal_rank(queries_results: list[tuple[list[str], set[str]]]) -> float:
    rr_sum = 0.0
    for retrieved, relevant in queries_results:
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                rr_sum += 1.0 / rank
                break
    return rr_sum / len(queries_results) if queries_results else 0.0
```

MRR = 1.0 means the first result is always relevant. MRR = 0.5 means first relevant result averages at rank 2.

### nDCG@K (Normalized Discounted Cumulative Gain)

When relevance is graded (not binary), nDCG accounts for degree of relevance:

```python
import numpy as np

def dcg_at_k(relevances: list[int], k: int) -> float:
    relevances = relevances[:k]
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(relevances))

def ndcg_at_k(retrieved: list[str], relevance_map: dict[str, int], k: int) -> float:
    """relevance_map: doc_id → relevance grade (0, 1, 2, 3)"""
    retrieved_relevances = [relevance_map.get(doc_id, 0) for doc_id in retrieved[:k]]
    ideal_relevances = sorted(relevance_map.values(), reverse=True)

    dcg = dcg_at_k(retrieved_relevances, k)
    idcg = dcg_at_k(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0
```

---

## Building an Evaluation Dataset

### Minimum Viable Eval Set

Start with 30–50 query-document pairs. Quality beats quantity:

```python
eval_dataset = [
    {
        "query": "How does attention mechanism work in transformers?",
        "relevant_docs": {"doc-attention-101", "doc-transformer-arch", "doc-self-attention"},
        "relevance_grades": {
            "doc-attention-101": 3,      # highly relevant
            "doc-transformer-arch": 2,   # relevant
            "doc-self-attention": 2,
        },
    },
    {
        "query": "Python error handling best practices",
        "relevant_docs": {"doc-python-exceptions", "doc-error-patterns"},
    },
    {
        "query": "error code E4041 resolution steps",
        "relevant_docs": {"doc-error-e4041", "doc-troubleshooting-guide"},
    },
]
```

### How to Create Labels

| Method | Effort | Quality | Scale |
|--------|--------|---------|-------|
| **Manual labeling** | High | Best | 30–200 queries |
| **Click logs** | Low (passive) | Good for popular queries | Unlimited |
| **LLM-assisted** | Medium | Moderate (verify samples) | 100–500 queries |
| **Synthetic from docs** | Low | Low-Medium | Bootstrap only |

!!! note "LLM-assisted labeling"
    Ask an LLM "given this query and these 20 chunks, which are relevant?" then manually verify 10% of labels. Fast bootstrap, but always human-verify a sample.

### Eval Dataset Growth Plan

```
Week 1:  30 queries from most common user questions (manual)
Week 4:  50 queries + add failure cases from production logs
Month 3: 100 queries + click-through data from top 20 queries
Month 6: 200 queries + quarterly review of zero-result queries
```

---

## Complete Evaluation Harness

```python
"""Production eval harness for vector search configurations."""
import numpy as np
from dataclasses import dataclass

@dataclass
class EvalResult:
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    latency_p50_ms: float
    latency_p95_ms: float

def evaluate_search_system(
    search_fn,
    eval_dataset: list[dict],
    k: int = 5,
) -> EvalResult:
    recalls, precisions, mrrs, ndcgs = [], [], [], []
    latencies = []

    for item in eval_dataset:
        import time
        t0 = time.perf_counter()
        results = search_fn(item["query"], k=k)
        latencies.append((time.perf_counter() - t0) * 1000)

        retrieved_ids = [r["id"] if isinstance(r, dict) else r for r in results]
        relevant = set(item["relevant_docs"])

        recalls.append(recall_at_k(retrieved_ids, relevant, k))
        precisions.append(precision_at_k(retrieved_ids, relevant, k))

        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant:
                mrrs.append(1.0 / rank)
                break
        else:
            mrrs.append(0.0)

        if "relevance_grades" in item:
            ndcgs.append(ndcg_at_k(retrieved_ids, item["relevance_grades"], k))

    return EvalResult(
        recall_at_k=np.mean(recalls),
        precision_at_k=np.mean(precisions),
        mrr=np.mean(mrrs),
        ndcg_at_k=np.mean(ndcgs) if ndcgs else 0.0,
        latency_p50_ms=np.median(latencies),
        latency_p95_ms=np.percentile(latencies, 95),
    )

def print_eval_report(name: str, result: EvalResult, k: int = 5):
    print(f"\n=== {name} ===")
    print(f"  Recall@{k}:    {result.recall_at_k:.3f}")
    print(f"  Precision@{k}: {result.precision_at_k:.3f}")
    print(f"  MRR:           {result.mrr:.3f}")
    print(f"  nDCG@{k}:      {result.ndcg_at_k:.3f}")
    print(f"  Latency p50:   {result.latency_p50_ms:.1f}ms")
    print(f"  Latency p95:   {result.latency_p95_ms:.1f}ms")
```

---

## Measuring ANN Index Recall

Your ANN index trades recall for speed. Measure exactly how much:

```python
"""Compare ANN index recall against brute-force ground truth."""
import hnswlib
import numpy as np
import time

def measure_ann_recall(data, queries, k=10, ef_values=[10, 20, 50, 100, 200]):
    n, d = data.shape

    # Ground truth
    flat = hnswlib.Index(space="l2", dim=d)
    flat.init_index(max_elements=n)
    flat.add_items(data, np.arange(n))

    ground_truth = []
    for q in queries:
        labels, _ = flat.knn_query(q, k=k)
        ground_truth.append(set(labels[0]))

    # ANN sweep
    hnsw = hnswlib.Index(space="l2", dim=d)
    hnsw.init_index(max_elements=n, ef_construction=200, M=16)
    hnsw.add_items(data, np.arange(n))

    print(f"{'ef':>6} {'ANN Recall@' + str(k):>14} {'p50 ms':>10} {'p95 ms':>10}")
    print("-" * 44)

    for ef in ef_values:
        hnsw.set_ef(ef)
        recalls, lats = [], []
        for i, q in enumerate(queries):
            t0 = time.perf_counter()
            labels, _ = hnsw.knn_query(q, k=k)
            lats.append((time.perf_counter() - t0) * 1000)
            recalls.append(len(set(labels[0]) & ground_truth[i]) / k)
        print(f"{ef:>6} {np.mean(recalls):>14.3f} {np.median(lats):>10.2f} "
              f"{np.percentile(lats, 95):>10.2f}")
```

This separates **embedding quality** (does the right doc have a similar vector?) from **index quality** (does ANN find that vector?). Both must be measured independently.

---

## A/B Testing Configurations

```python
configs = {
    "baseline_small_512": {
        "model": "text-embedding-3-small", "chunk_size": 512, "ef": 50, "hybrid_alpha": None,
    },
    "large_model": {
        "model": "text-embedding-3-large", "chunk_size": 512, "ef": 50, "hybrid_alpha": None,
    },
    "small_chunks": {
        "model": "text-embedding-3-small", "chunk_size": 256, "ef": 50, "hybrid_alpha": None,
    },
    "hybrid": {
        "model": "text-embedding-3-small", "chunk_size": 512, "ef": 50, "hybrid_alpha": 0.5,
    },
    "hybrid_low_alpha": {
        "model": "text-embedding-3-small", "chunk_size": 512, "ef": 50, "hybrid_alpha": 0.25,
    },
}

for name, config in configs.items():
    search_fn = build_search_from_config(config)
    result = evaluate_search_system(search_fn, eval_dataset, k=5)
    print_eval_report(name, result, k=5)
```

Run this before every production change. Document results in a config comparison table for your team.

---

## Production Monitoring

Track these metrics continuously:

| Metric | What It Tells You | Alert Threshold | Dashboard |
|--------|-------------------|-----------------|-----------|
| **Avg top-1 similarity** | Overall retrieval quality | Drop > 10% from baseline | Real-time |
| **Zero-result rate** | Queries matching nothing | > 5% of queries | Daily |
| **Latency p95** | Search slowing down | > 200ms | Real-time |
| **Click-through rate** | Users finding results useful | Drop > 15% | Weekly |
| **Recall@5 (weekly eval)** | Quality drift | Drop > 5% from baseline | Weekly job |
| **Index vector count** | Ingestion health | Unexpected drop | Daily |

### Weekly Eval Job

```python
def weekly_eval_job(search_fn, eval_dataset, baseline_recall: float):
    result = evaluate_search_system(search_fn, eval_dataset, k=5)

    if result.recall_at_k < baseline_recall - 0.05:
        alert(f"RECALL DROP: {result.recall_at_k:.3f} vs baseline {baseline_recall:.3f}")

    if result.latency_p95_ms > 200:
        alert(f"LATENCY SPIKE: p95={result.latency_p95_ms:.0f}ms")

    log_metrics(result)
    return result
```

---

## Failure Modes in Evaluation

**Eval set too small.** 5 queries can't distinguish config changes from noise. Minimum 30; ideal 100+.

**Eval set not representative.** Labeling only easy queries inflates scores. Include hard queries, edge cases, and zero-result queries from production logs.

**Data leakage.** Eval queries appear in the indexed corpus as exact matches. Hold out eval documents or use queries that require semantic (not lexical) matching.

**Optimizing recall alone.** Recall@5 = 0.95 with Precision@5 = 0.20 means you're returning 4 irrelevant results. Balance both metrics.

**Not re-evaluating after corpus changes.** Adding 10K new documents shifts similarity distributions. Re-run eval after major ingestions.

---

## Production Notes

- Build your eval dataset in Week 1 — even 30 labeled queries prevent costly mistakes later.
- Separate embedding eval (brute force) from index eval (ANN parameters) — they have different fixes.
- Run A/B config comparisons before every embedding model or index parameter change.
- Schedule weekly eval jobs against a frozen eval set; alert on recall drops > 5%.
- Log query + top-5 results for failed queries; mine these for eval set growth.

---

## Key Takeaways

- Recall@K measures completeness; Precision@K measures cleanliness; MRR measures first-hit quality.
- Build 30–200 labeled query-document pairs from real user queries and production failures.
- Measure ANN recall separately from embedding quality — different problems, different fixes.
- A/B test every configuration change with the eval harness before deploying.
- Monitor similarity scores, zero-result rate, and latency in production; weekly eval for drift.

---

## Resources

- [RAGAS Framework](https://docs.ragas.io/) — Automated RAG evaluation
- [BEIR Benchmark](https://github.com/beir-cellar/beir) — Standard IR benchmark suite
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — Embedding model benchmarks

---

## Next Lesson

**[Lesson 10: Production Vector Database Patterns](10-lesson-10.md)** — Ingestion pipelines, caching, updates/deletions, cost optimization, and operational best practices.
