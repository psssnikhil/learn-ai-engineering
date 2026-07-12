---
title: Indexing Strategies for Vector Search
description: >-
  Learn how vector databases organize and index embeddings for fast approximate
  nearest neighbor search
duration: 40 min
difficulty: intermediate
has_code: false
---
# Indexing Strategies for Vector Search

## Learning Objectives

By the end of this lesson, you will be able to:
- Explain why brute-force search does not scale for vector databases
- Compare key indexing algorithms: IVF, HNSW, and Product Quantization
- Understand the trade-offs between speed, accuracy, and memory usage
- Configure index parameters for your use case

---

## The Scalability Problem

Brute-force search compares your query vector against every vector in the database. This is exact but slow:

| Vectors | Dimensions | Comparisons | Time (approx) |
|---------|-----------|-------------|---------------|
| 10,000 | 1536 | 10,000 | ~5ms |
| 1,000,000 | 1536 | 1,000,000 | ~500ms |
| 100,000,000 | 1536 | 100,000,000 | ~50s |

At scale, you need **Approximate Nearest Neighbor (ANN)** algorithms that trade a small amount of accuracy for massive speed gains.

---

## IVF (Inverted File Index)

IVF partitions the vector space into clusters using k-means, then only searches the closest clusters at query time.

```python
import faiss
import numpy as np

# Create 100,000 random vectors (dimension 128)
d = 128
n = 100_000
vectors = np.random.random((n, d)).astype("float32")

# Build IVF index with 100 clusters
nlist = 100  # number of clusters
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)

# Train the index (learns cluster centroids)
index.train(vectors)
index.add(vectors)

# Search: only probe 10 of 100 clusters
index.nprobe = 10
query = np.random.random((1, d)).astype("float32")
distances, indices = index.search(query, k=5)
print(f"Top 5 nearest neighbors: {indices[0]}")
```

**Key parameters:**
- `nlist`: Number of clusters (more clusters = faster search, needs more training data)
- `nprobe`: Clusters to search at query time (higher = more accurate, slower)

---

## HNSW (Hierarchical Navigable Small World)

HNSW builds a multi-layer graph where each node connects to nearby vectors. Search navigates from coarse upper layers to fine lower layers.

```python
import hnswlib
import numpy as np

d = 128
n = 100_000
data = np.random.random((n, d)).astype("float32")

# Create HNSW index
index = hnswlib.Index(space="cosine", dim=d)
index.init_index(max_elements=n, ef_construction=200, M=16)
index.add_items(data)

# Query
index.set_ef(50)  # search-time accuracy parameter
query = np.random.random((1, d)).astype("float32")
labels, distances = index.knn_query(query, k=5)
print(f"Top 5: {labels[0]}")
```

**Key parameters:**
- `M`: Max connections per node (higher = better recall, more memory)
- `ef_construction`: Build-time quality (higher = better graph, slower build)
- `ef`: Search-time quality (higher = better recall, slower search)

---

## Product Quantization (PQ)

PQ compresses vectors by splitting them into sub-vectors and quantizing each independently. This dramatically reduces memory usage.

```python
import faiss
import numpy as np

d = 128
n = 100_000
vectors = np.random.random((n, d)).astype("float32")

# PQ with 16 sub-quantizers, 8 bits each
m = 16       # number of sub-vectors
nbits = 8    # bits per sub-vector code

index = faiss.IndexPQ(d, m, nbits)
index.train(vectors)
index.add(vectors)

# Memory comparison
print(f"Raw vectors: {n * d * 4 / 1e6:.1f} MB")   # 48.8 MB
print(f"PQ compressed: {n * m / 1e6:.1f} MB")       # 1.6 MB (30x smaller)
```

---

## Index Comparison

| Algorithm | Speed | Recall | Memory | Build Time | Best For |
|-----------|-------|--------|--------|------------|----------|
| **Flat (brute force)** | Slow | 100% | High | None | < 50K vectors |
| **IVF** | Fast | 95-99% | Medium | Medium | Large datasets, batch queries |
| **HNSW** | Very fast | 97-99% | High | Slow | Real-time search |
| **PQ** | Fast | 90-95% | Very low | Medium | Memory-constrained systems |
| **IVF + PQ** | Very fast | 92-97% | Low | Medium | Billion-scale search |

---

## Choosing the Right Index

```
How many vectors do you have?
  ├─ < 50,000 → Use Flat (brute force)
  ├─ 50K - 1M → Use HNSW (best recall/speed)
  ├─ 1M - 100M
  │    ├─ Memory constrained? → Use IVF + PQ
  │    └─ Memory available? → Use HNSW
  └─ > 100M → Use IVF + PQ or ScaNN
```

---

## Key Takeaways

- Brute-force search is exact but does not scale beyond tens of thousands of vectors
- HNSW provides the best recall-to-speed ratio for most use cases
- IVF is efficient for large batch searches with tunable accuracy
- Product Quantization compresses vectors 10-30x for memory-constrained environments
- Combining algorithms (IVF + PQ) enables billion-scale search

## Resources

- [Faiss Wiki](https://github.com/facebookresearch/faiss/wiki) -- Index selection guide and benchmarks
- [Pinecone: What is a Vector Index?](https://www.pinecone.io/learn/vector-database/) -- Visual explanations of ANN algorithms
- [ANN Benchmarks](https://ann-benchmarks.com/) -- Compare algorithm performance across datasets
- [YouTube: HNSW Explained](https://www.youtube.com/watch?v=QvKMwLjdK-s) -- Visual walkthrough of the HNSW algorithm

---

Next: Similarity Search Algorithms
