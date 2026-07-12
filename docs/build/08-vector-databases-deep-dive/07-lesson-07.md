---
title: Scaling Vector Search
description: >-
  Learn strategies for scaling vector databases to handle millions or billions
  of vectors in production
duration: 35 min
difficulty: advanced
has_code: false
---
# Scaling Vector Search

## Learning Objectives

By the end of this lesson, you will be able to:
- Identify bottlenecks in vector search at scale
- Apply sharding and replication strategies
- Optimize memory usage with quantization techniques
- Design for high availability and low latency

---

## Scaling Challenges

As your vector database grows, you hit three walls:

| Challenge | Symptom | Scale Threshold |
|-----------|---------|----------------|
| **Memory** | Index does not fit in RAM | ~10M vectors (1536d) = ~60 GB |
| **Latency** | Search takes > 100ms | Depends on index type and hardware |
| **Throughput** | Cannot handle concurrent queries | ~1000 QPS per node |

---

## Sharding Strategies

Sharding splits your data across multiple nodes to distribute load.

### Horizontal Sharding

Split vectors across nodes, query all shards in parallel, merge results.

```
Query → [Shard 1] → Top 10
      → [Shard 2] → Top 10  → Merge → Global Top 10
      → [Shard 3] → Top 10
```

**Managed databases handle this automatically:**
- Pinecone: Pods auto-shard based on index size
- Weaviate: Built-in sharding with configurable shard count
- Milvus: Automatic sharding across nodes

### Namespace/Partition Sharding

Route queries to specific partitions based on metadata (e.g., by tenant, language, or date range).

```
User query (tenant=acme) → Shard "acme" → Results
User query (tenant=globex) → Shard "globex" → Results
```

This avoids searching irrelevant data entirely.

---

## Memory Optimization

### Quantization Reduces Memory 4-32x

| Technique | Memory per Vector (1536d) | Recall Impact |
|-----------|--------------------------|---------------|
| **Float32 (raw)** | 6,144 bytes | 100% (exact) |
| **Float16** | 3,072 bytes | ~99.9% |
| **Int8 (scalar quantization)** | 1,536 bytes | ~99% |
| **Product Quantization** | 192-384 bytes | 95-98% |
| **Binary Quantization** | 192 bytes | 90-96% |

### Disk-Based Indexes

For datasets that exceed RAM, use disk-backed indexes:

- **Vamana/DiskANN**: Microsoft's disk-optimized ANN algorithm
- **Milvus DiskANN**: Built-in support for SSD-based indexes
- **Qdrant**: Memory-mapped storage for large collections

---

## Replication for High Availability

```
                Write
                  │
          ┌───────┼───────┐
          ▼       ▼       ▼
      [Primary] [Replica] [Replica]
          │       │         │
          └───────┼─────────┘
                  │
              Read queries
              (load balanced)
```

- **Primary**: Handles writes, replicates to followers
- **Replicas**: Handle read queries, provide failover
- Most managed services handle replication automatically

---

## Performance Optimization Checklist

1. **Choose the right index**: HNSW for low-latency, IVF+PQ for memory efficiency
2. **Tune search parameters**: Balance recall vs speed (e.g., HNSW `ef`, IVF `nprobe`)
3. **Pre-filter when possible**: Reduce the search space with metadata filters
4. **Batch queries**: Group multiple searches into single API calls
5. **Cache frequent queries**: Store results for common search patterns
6. **Use appropriate dimensions**: Reduce embedding dimensions if quality allows (OpenAI supports 256-3072)
7. **Monitor and benchmark**: Track p50/p95/p99 latency, recall, and throughput

---

## Key Takeaways

- Most applications do not need to worry about scaling until 1M+ vectors
- Managed services (Pinecone, Weaviate Cloud) handle sharding and replication automatically
- Quantization is the most impactful optimization: 4-32x memory reduction with minimal recall loss
- Partition your data by tenant or domain to avoid searching irrelevant vectors
- Always benchmark with your actual data and query patterns before optimizing

## Resources

- [YouTube: Scaling Vector Databases](https://www.youtube.com/watch?v=09BnQJkWHbg) -- Production scaling strategies
- [Pinecone: Performance Tuning](https://docs.pinecone.io/guides/performance/performance-tuning) -- Official optimization guide
- [DiskANN Paper](https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html) -- Microsoft's disk-based ANN algorithm
- [ANN Benchmarks](https://ann-benchmarks.com/) -- Compare algorithms on standardized workloads

---

Next: Hybrid Search with Vector Databases
