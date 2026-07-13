---
title: Indexing Strategies for Vector Search
description: >-
  Learn how vector databases organize and index embeddings for fast approximate
  nearest neighbor search
duration: 55 min
difficulty: intermediate
has_code: true
module: module-13
---
# Indexing Strategies for Vector Search

## Prerequisites

Before this lesson you should be comfortable with:

- **Module 09, Lesson 02** — basic HNSW/IVF/flat index concepts
- **Module 13, Lessons 01–02** — vector database architecture, embeddings, cosine similarity
- **NumPy** — array operations for the Faiss examples below

Module 09 introduced index types at a whiteboard level. This lesson goes hands-on: you will tune HNSW parameters, sweep IVF `nprobe`, compress vectors with Product Quantization, and run a benchmark script that plots recall vs latency for your data.

---

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Explain why brute-force search fails at scale and what ANN trades off | 10 min | Intermediate |
| Build and tune HNSW, IVF, and PQ indexes with Faiss | 20 min | Intermediate |
| Run a parameter sweep benchmark measuring recall@10 vs latency | 15 min | Intermediate |
| Choose the right index type for your scale and latency budget | 10 min | Intermediate |

---

## Intuition First: The City Directory

Imagine finding a restaurant in a city of 10 million buildings.

**Brute force:** Walk to every building and read the sign. Guaranteed to find the best match, but you'll never finish.

**IVF (Inverted File):** Divide the city into 100 neighborhoods. At query time, only search the 5 neighborhoods closest to your starting point. You might miss a great restaurant one neighborhood over — but you're 20× faster.

**HNSW (Hierarchical Navigable Small World):** Build a highway system connecting popular buildings. Start on the express layer (few nodes, long jumps), zoom into local streets (many nodes, short hops). Navigation feels like asking locals for directions — each hop gets you closer.

**Product Quantization (PQ):** Instead of storing full addresses, compress each building to a 4-digit zip code. You lose precision but fit the entire city in your pocket.

Every ANN algorithm makes the same deal: **speed and memory for a small recall loss**. Your job is tuning how much loss is acceptable.

---

## The Scalability Problem

Brute-force search compares the query vector against every stored vector. Complexity is O(n × d) where n = corpus size, d = dimensions.

| Vectors | Dimensions | Distance Computations | Latency (approx) |
|---------|-----------|----------------------|------------------|
| 10,000 | 1536 | 10,000 | ~5 ms |
| 1,000,000 | 1536 | 1,000,000 | ~300–500 ms |
| 10,000,000 | 1536 | 10,000,000 | ~3–5 s |
| 100,000,000 | 1536 | 100,000,000 | ~30–60 s |

At 1M+ vectors, brute force is unusable for interactive search. ANN algorithms target **95–99% recall** (finding the true nearest neighbor in top-K) at **1–50 ms** latency.

---

## HNSW: Deep Dive and Tuning

HNSW (Hierarchical Navigable Small World) builds a multi-layer graph. Upper layers are sparse (long-range connections); lower layers are dense (fine-grained navigation).

### Building an HNSW Index

```python
import hnswlib
import numpy as np

d = 128
n = 100_000
data = np.random.random((n, d)).astype("float32")

index = hnswlib.Index(space="cosine", dim=d)
index.init_index(
    max_elements=n,
    ef_construction=200,  # build-time quality
    M=16,                   # max connections per node
)
index.add_items(data, np.arange(n))
index.set_ef(50)  # search-time quality

query = np.random.random((1, d)).astype("float32")
labels, distances = index.knn_query(query, k=10)
print(f"Top 10 indices: {labels[0]}")
```

### HNSW Tuning Table

| Parameter | Build or Search | Default | Effect of Increasing | Trade-off |
|-----------|----------------|---------|---------------------|-----------|
| **M** | Build | 16 | Better recall, denser graph | +Memory (~M × 8 bytes × n), slower build |
| **ef_construction** | Build | 200 | Better graph quality | Slower index build (linear) |
| **ef** (or `ef_search`) | Search | 50 | Higher recall@K | +Latency (linear) |

### Recommended Starting Points

| Corpus Size | M | ef_construction | ef (search) | Expected Recall@10 |
|-------------|---|-----------------|-------------|-------------------|
| < 100K | 16 | 100 | 32 | ~97% |
| 100K – 1M | 16 | 200 | 64 | ~98% |
| 1M – 10M | 24 | 400 | 128 | ~98.5% |
| > 10M | 32 | 500 | 200 | ~99% |

!!! warning "ef must be ≥ k"
    If you request top-10 results, set `ef ≥ 10`. Best practice: `ef = 2 × k` as a starting point.

### HNSW Failure Modes

- **ef too low:** Recall drops sharply. Symptom: correct documents never appear in top-K despite good embeddings.
- **M too low on large corpora:** Graph becomes disconnected; some regions unreachable. Symptom: high-latency queries that still miss relevant results.
- **Frequent updates on HNSW:** Graph quality degrades with many in-place updates. For high-churn data, consider periodic rebuilds or IVF-based indexes.

---

## IVF: Partition-Based Search

IVF (Inverted File Index) clusters vectors with k-means, then at query time probes only the nearest clusters.

```python
import faiss
import numpy as np

d = 128
n = 100_000
vectors = np.random.random((n, d)).astype("float32")

nlist = 100  # number of clusters
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)

index.train(vectors)  # learns cluster centroids — requires ≥ nlist vectors
index.add(vectors)

index.nprobe = 10  # clusters to search at query time
query = np.random.random((1, d)).astype("float32")
distances, indices = index.search(query, k=10)
print(f"Top 10: {indices[0]}")
```

### IVF Tuning Table

| Parameter | Meaning | Rule of Thumb |
|-----------|---------|---------------|
| **nlist** | Number of clusters | `sqrt(n)` to `4 × sqrt(n)`; e.g., 1000 for 1M vectors |
| **nprobe** | Clusters searched per query | Start at `nlist / 10`; increase until recall target met |

| nlist | nprobe | Recall@10 (typical) | Latency vs Flat |
|-------|--------|---------------------|-----------------|
| 100 | 1 | ~85% | 10× faster |
| 100 | 5 | ~93% | 5× faster |
| 100 | 10 | ~97% | 3× faster |
| 100 | 25 | ~99% | 1.5× faster |
| 100 | 100 | ~100% | ~same as flat |

**When to choose IVF over HNSW:** Batch query workloads (many queries at once), very large corpora (>10M) where HNSW memory is prohibitive, or when you already use Faiss/Milvus infrastructure.

---

## Product Quantization (PQ)

PQ compresses vectors by splitting them into sub-vectors and replacing each with a small codebook index. Memory reduction: 10–32×.

```python
import faiss
import numpy as np

d = 128
n = 100_000
vectors = np.random.random((n, d)).astype("float32")

m = 16       # sub-vectors (must divide d evenly)
nbits = 8    # bits per code

index = faiss.IndexPQ(d, m, nbits)
index.train(vectors)
index.add(vectors)

print(f"Raw:  {n * d * 4 / 1e6:.1f} MB")
print(f"PQ:   {n * m * nbits / 8 / 1e6:.1f} MB")

query = np.random.random((1, d)).astype("float32")
distances, indices = index.search(query, k=10)
```

### PQ Compression Levels

| Config | Bytes/Vector (d=1536) | Compression | Recall@10 Loss |
|--------|------------------------|-------------|----------------|
| Raw float32 | 6,144 | 1× | 0% |
| PQ m=48, 8bit | 48 | 128× | ~3-5% |
| PQ m=96, 8bit | 96 | 64× | ~1-3% |
| Scalar int8 | 1,536 | 4× | ~0.5-1% |

### IVF + PQ: Billion-Scale Standard

For 100M+ vectors, combine IVF partitioning with PQ compression:

```python
nlist = 4096
m = 48
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
index.train(vectors)
index.add(vectors)
index.nprobe = 32
```

This is the default architecture in Milvus and many large-scale deployments.

---

## Benchmark Script: Recall vs Latency Sweep

Run this script to find the optimal HNSW `ef` for your latency budget:

```python
"""Sweep HNSW ef parameter and measure recall@10 vs latency."""
import hnswlib
import numpy as np
import time

np.random.seed(42)
d, n = 128, 50_000
data = np.random.random((n, d)).astype("float32")
queries = np.random.random((100, d)).astype("float32")

# Ground truth via brute force
flat = hnswlib.Index(space="l2", dim=d)
flat.init_index(max_elements=n)
flat.add_items(data, np.arange(n))

ground_truth = []
for q in queries:
    labels, _ = flat.knn_query(q, k=10)
    ground_truth.append(set(labels[0]))

# HNSW index
hnsw = hnswlib.Index(space="l2", dim=d)
hnsw.init_index(max_elements=n, ef_construction=200, M=16)
hnsw.add_items(data, np.arange(n))

print(f"{'ef':>6} {'recall@10':>10} {'p50 ms':>10} {'p95 ms':>10}")
print("-" * 42)

for ef in [10, 20, 50, 100, 200, 400]:
    hnsw.set_ef(ef)
    latencies, recalls = [], []

    for i, q in enumerate(queries):
        t0 = time.perf_counter()
        labels, _ = hnsw.knn_query(q, k=10)
        latencies.append((time.perf_counter() - t0) * 1000)
        recalls.append(len(set(labels[0]) & ground_truth[i]) / 10)

    print(f"{ef:>6} {np.mean(recalls):>10.3f} {np.median(latencies):>10.2f} "
          f"{np.percentile(latencies, 95):>10.2f}")
```

Typical output pattern:

```
    ef  recall@10     p50 ms     p95 ms
------------------------------------------
    10      0.912       0.05       0.08
    50      0.974       0.12       0.18
   100      0.989       0.21       0.31
   200      0.996       0.38       0.55
```

Pick the lowest `ef` that meets your recall target within your latency SLA.

---

## Index Selection Decision Tree

```
How many vectors?
  ├─ < 50K        → Flat (brute force) — exact, simple, no tuning
  ├─ 50K – 1M     → HNSW — best recall/latency for real-time
  ├─ 1M – 100M
  │    ├─ RAM available?  → HNSW (high recall) or IVF+Flat (balanced)
  │    └─ RAM constrained? → IVF+PQ (4-32× memory savings)
  └─ > 100M       → IVF+PQ or DiskANN (SSD-backed)

Query pattern?
  ├─ Real-time single queries  → HNSW
  ├─ Batch analytics           → IVF
  └─ High update rate          → IVF (HNSW degrades with churn)
```

---

## Index Comparison Summary

| Algorithm | Speed | Recall@10 | Memory | Build Time | Best For |
|-----------|-------|-----------|--------|------------|----------|
| **Flat** | Slowest | 100% | Highest | None | < 50K, eval ground truth |
| **HNSW** | Very fast | 97–99.5% | High | Slow | Real-time search |
| **IVF** | Fast | 93–99% | Medium | Medium | Large batch queries |
| **PQ** | Fast | 90–95% | Very low | Medium | Memory-constrained |
| **IVF+PQ** | Very fast | 92–97% | Low | Medium | Billion-scale |
| **DiskANN** | Medium | 95–98% | Disk | Slow | RAM << dataset |

---

## Production Notes

- **Always measure recall on your data.** Synthetic random vectors behave differently from real embeddings. Use Lesson 9's eval harness.
- **Index build is expensive.** HNSW build for 10M vectors can take hours. Plan for offline rebuilds, not inline updates.
- **Match distance metric to index.** Cosine for text embeddings; L2 if vectors aren't normalized.
- **Monitor recall in production.** Track average similarity score of top-1 results; a sudden drop often means index corruption or parameter drift.

---

## Key Takeaways

- ANN trades 1–5% recall for 10–1000× speed — tune parameters to your SLA, not defaults.
- HNSW: tune `M`, `ef_construction` at build time; `ef` at search time. Start with `ef = 2 × k`.
- IVF: tune `nlist ≈ sqrt(n)` and sweep `nprobe` until recall target is met.
- PQ compresses 10–32× with ~2–5% recall loss; combine with IVF for billion-scale.
- Run the benchmark sweep script on your actual embeddings before choosing parameters.

---

## Resources

- [Faiss Wiki — Index Selection](https://github.com/facebookresearch/faiss/wiki)
- [ANN Benchmarks](https://ann-benchmarks.com/)
- [HNSW Paper (Malkov & Yashunin, 2018)](https://arxiv.org/abs/1603.09320)

---

## Next Lesson

**[Lesson 4: Working with Pinecone](04-lesson-04.md)** — Production Pinecone setup: serverless indexes, namespaces, metadata filtering, batch upserts, and cost modeling.
