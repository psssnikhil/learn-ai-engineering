---
title: Scaling Vector Search
description: >-
  Learn strategies for scaling vector databases to handle millions or billions
  of vectors in production
duration: 50 min
difficulty: advanced
has_code: true
module: module-13
---
# Scaling Vector Search

## Prerequisites

Before this lesson you should be comfortable with:

- **Module 13, Lessons 01–06** — ANN indexes, HNSW tuning, schema design, multi-tenant patterns
- **Module 13, Lesson 03** — IVF, PQ, benchmark scripts
- **Basic systems concepts** — RAM, SSD, QPS, p95 latency

Most applications don't need scaling until 1M+ vectors. When you hit that wall, this lesson gives you the playbook: sharding strategies, quantization trade-offs, replication, and a performance optimization checklist with tuning tables.

---

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Identify memory, latency, and throughput bottlenecks at scale | 10 min | Advanced |
| Apply horizontal sharding and partition routing strategies | 15 min | Advanced |
| Reduce memory 4–32× with quantization techniques | 10 min | Advanced |
| Design for high availability with replication and failover | 15 min | Advanced |

---

## Intuition First: The City That Outgrew Its Map

Your semantic map (vector index) worked perfectly for 100K locations. Now you have 100 million. The map doesn't fit on your desk (RAM), lookups take minutes (latency), and thousands of people ask for directions simultaneously (throughput).

Scaling options:
1. **Compress the map** (quantization) — less detail, fits in pocket
2. **Split into boroughs** (sharding) — each borough has its own local map
3. **Hire assistants** (replication) — multiple copies handle concurrent requests
4. **Store maps on shelves, not desks** (disk-based indexes) — slower access but unlimited size

Production systems combine all four.

---

## Scaling Challenges

| Challenge | Symptom | Threshold (1536d float32) |
|-----------|---------|----------------------------|
| **Memory** | Index doesn't fit in RAM; OOM kills | ~10M vectors ≈ 60 GB |
| **Latency** | p95 search > 100ms | Depends on index type, QPS |
| **Throughput** | Query queue backs up | ~500–2000 QPS per node |
| **Ingestion** | Bulk upsert takes hours | > 1M vectors per batch |
| **Updates** | HNSW graph degrades with churn | > 10% daily update rate |

---

## Memory Math

Before choosing an optimization, calculate your memory budget:

```python
def estimate_memory(num_vectors: int, dimensions: int, bytes_per_float: int = 4) -> dict:
    raw_bytes = num_vectors * dimensions * bytes_per_float
    return {
        "raw_gb": raw_bytes / 1e9,
        "float16_gb": raw_bytes / 2 / 1e9,
        "int8_gb": raw_bytes / 4 / 1e9,
        "pq_48byte_gb": num_vectors * 48 / 1e9,
        "hnsw_overhead_gb": raw_bytes * 1.3 / 1e9,  # ~30% graph overhead
    }

for n in [1_000_000, 10_000_000, 100_000_000]:
    mem = estimate_memory(n, 1536)
    print(f"{n/1e6:.0f}M vectors: raw={mem['raw_gb']:.1f}GB, "
          f"HNSW={mem['hnsw_overhead_gb']:.1f}GB, PQ={mem['pq_48byte_gb']:.1f}GB")
```

Typical output:
```
1M vectors:  raw=6.1GB, HNSW=8.0GB, PQ=0.05GB
10M vectors: raw=61.4GB, HNSW=79.9GB, PQ=0.48GB
100M vectors: raw=614.4GB, HNSW=798.7GB, PQ=4.8GB
```

At 10M vectors, raw HNSW requires ~80 GB RAM. Quantization or disk-based indexes become mandatory.

---

## Sharding Strategies

### Horizontal Sharding

Split vectors across nodes; query all shards in parallel; merge top-K results.

```
Query → Router
          ├─ Shard 1 (0-3.3M vectors)  → Top 10
          ├─ Shard 2 (3.3M-6.6M)       → Top 10
          └─ Shard 3 (6.6M-10M)        → Top 10
                    ↓
              Merge → Global Top 10
```

```python
def sharded_search(shards: list, query_vec, k: int = 10) -> list:
    """Query all shards and merge results by score."""
    all_results = []
    for shard in shards:
        results = shard.query(vector=query_vec, top_k=k)
        all_results.extend(results["matches"])

    all_results.sort(key=lambda x: x["score"], reverse=True)
    return all_results[:k]
```

| Database | Sharding Support | Configuration |
|----------|-----------------|---------------|
| Pinecone | Automatic (pods) | Pod size + replicas |
| Qdrant | Manual sharding | `shard_number` at collection creation |
| Weaviate | Automatic | `shardCount` in schema |
| Milvus | Automatic | Distributed deployment |

### Partition Sharding (Tenant/Domain Routing)

Route queries to specific partitions based on metadata — avoids searching irrelevant data entirely:

```python
SHARD_MAP = {
    "tenant-acme": shard_client_1,
    "tenant-globex": shard_client_2,
    "tenant-initech": shard_client_3,
}

def routed_search(tenant_id: str, query_vec, k: int = 5):
    shard = SHARD_MAP.get(tenant_id)
    if not shard:
        raise ValueError(f"Unknown tenant: {tenant_id}")
    return shard.query(vector=query_vec, top_k=k)
```

This is the most cost-effective sharding for multi-tenant RAG: each tenant's queries only search their own vectors.

### Sharding Strategy Selection

| Pattern | When to Use | Latency Impact | Cost |
|---------|-------------|----------------|------|
| Horizontal (query all shards) | Single large corpus | +Merge overhead | Linear with shards |
| Partition routing | Multi-tenant | Minimal (single shard) | Efficient |
| Time-based partitions | Log/event data | Low (recent data hot) | Archive old shards to cold storage |
| Hash sharding | Even distribution | +Merge overhead | Predictable |

---

## Memory Optimization: Quantization

Quantization reduces vector precision to save memory with minimal recall loss.

| Technique | Bytes/Vector (1536d) | Compression | Recall@10 Loss | Implementation |
|-----------|---------------------|-------------|----------------|----------------|
| Float32 (raw) | 6,144 | 1× | 0% | Default |
| Float16 | 3,072 | 2× | ~0.1% | Qdrant, Milvus |
| Int8 scalar | 1,536 | 4× | ~0.5-1% | Qdrant, Pinecone |
| Product Quantization | 48–192 | 32–128× | ~2-5% | Faiss, Milvus, Qdrant |
| Binary | 192 | 32× | ~4-8% | Qdrant, experimental |

### Qdrant Scalar Quantization

```python
from qdrant_client.models import ScalarQuantization, ScalarQuantizationConfig, ScalarType

client.update_collection(
    collection_name="documents",
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(type=ScalarType.INT8, always_ram=True)
    ),
)
```

### Matryoshka + Quantization Combined

From Lesson 2: truncate to 256 dimensions (6× savings) then apply int8 quantization (another 4×) = **24× total memory reduction** with ~3-6% recall loss. Often the best cost/quality trade-off for large deployments.

---

## Disk-Based Indexes

When RAM < dataset size, store indexes on SSD:

| Algorithm | Provider | Latency vs RAM | Scale |
|-----------|----------|---------------|-------|
| **DiskANN** | Milvus, Azure AI Search | 2-5× slower | 100M–1B vectors |
| **Vamana** | Milvus 2.x | 2-4× slower | 100M+ vectors |
| **Qdrant mmap** | Qdrant | 1.5-3× slower | Configurable |

DiskANN keeps a compressed graph on SSD with a small in-memory cache for hot nodes. p95 latency typically 10–50ms vs 1–5ms for RAM-resident HNSW — acceptable for many RAG workloads.

---

## Replication for High Availability

```
                Writes
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

| Component | Role | Failover |
|-----------|------|----------|
| **Primary** | Handles writes, replicates | Promote replica on failure |
| **Replica** | Serves read queries | Automatic in managed services |
| **Load balancer** | Distributes read QPS | Health-check based routing |

Managed services (Pinecone, Qdrant Cloud, Weaviate Cloud) handle replication automatically. Self-hosted: configure at least 2 replicas for production.

---

## Performance Optimization Checklist

| # | Optimization | Expected Impact | Effort |
|---|-------------|-------------------|--------|
| 1 | Tune HNSW `ef` to minimum meeting recall target | -20-50% latency | Low |
| 2 | Pre-filter by tenant/category before ANN | -30-70% search space | Low |
| 3 | Scalar quantization (int8) | 4× memory, ~1% recall loss | Low |
| 4 | Matryoshka 256d truncation | 6× memory, ~3% recall loss | Low |
| 5 | Partition sharding by tenant | Eliminates cross-tenant search | Medium |
| 6 | Cache frequent query embeddings | -20ms per cached query | Medium |
| 7 | Batch concurrent queries | +2-5× throughput | Medium |
| 8 | DiskANN for RAM-constrained datasets | Enables 10× larger corpus | High |

### Latency Budget Template

| Stage | Target | Notes |
|-------|--------|-------|
| Query embedding | 15–30 ms | Cache for repeated queries |
| ANN search | 1–20 ms | Depends on index type and ef |
| Metadata fetch | 1–5 ms | Included in most DB responses |
| Reranking (optional) | 20–100 ms | Cross-encoder, Lesson 8 |
| **Total retrieval** | **30–150 ms** | Before LLM call |

---

## Failure Modes at Scale

**Hot shard.** One tenant with 90% of vectors dominates a shared shard; their queries are fast, others wait. Fix: dedicated shards for large tenants.

**Merge bottleneck.** Horizontal sharding with 10+ shards creates merge overhead at query time. Fix: partition routing or reduce shard count.

**Quantization recall cliff.** Aggressive PQ (m=16 on 1536d) drops recall 8-10%. Fix: benchmark on eval set before deploying; use scalar int8 first.

**Replication lag.** Writes to primary not yet visible on replicas. Fix: read-your-writes consistency mode or query primary for recent upserts.

**OOM during bulk ingest.** Loading 10M vectors into HNSW build exceeds RAM. Fix: batch index build, use IVF+PQ, or disk-based index.

---

## Production Notes

- Benchmark with your actual data at target scale before choosing architecture — synthetic benchmarks lie.
- Start with a single well-tuned HNSW index; shard only when memory or QPS demands it.
- Monitor p50/p95/p99 latency, QPS, and index size weekly. Set alerts at 2× baseline.
- Plan capacity 3–6 months ahead: embedding ingestion rate × chunk size × growth rate.

---

## Key Takeaways

- Memory is the first wall at ~10M vectors (60+ GB for 1536d float32). Quantization and sharding are the exits.
- Partition sharding by tenant is the most cost-effective multi-tenant scaling pattern.
- Scalar int8 quantization gives 4× memory savings with ~1% recall loss — deploy this before exotic optimizations.
- DiskANN enables billion-scale when RAM is insufficient; accept 2-5× latency increase.
- Always measure recall on your eval set after any scaling optimization.

---

## Resources

- [Pinecone: Performance Tuning](https://docs.pinecone.io/guides/performance/performance-tuning)
- [DiskANN Paper](https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html)
- [ANN Benchmarks](https://ann-benchmarks.com/)

---

## Next Lesson

**[Lesson 8: Hybrid Search with Vector Databases](08-lesson-08.md)** — Combine BM25 and vector search with RRF, benchmark alpha sweeps, and production hybrid pipelines.
