---
title: RAG in Production
description: >-
  Deploy, monitor, and maintain RAG systems at scale — caching, versioning,
  observability, and cost control
duration: 65 min
difficulty: advanced
has_code: true
module: module-09
youtube: 'https://www.youtube.com/watch?v=T-D1OfcDW1M'
---

## Prerequisites

- [Lesson 05 — Building a Basic RAG System](05-Building-a-Basic-RAG-System.md): working RAG pipeline
- [Lesson 08 — RAG Evaluation Metrics](08-RAG-Evaluation-Metrics.md): eval dataset and quality gates
- Comfortable with Python async, Redis basics, and REST API concepts
- Basic familiarity with logging and monitoring concepts

---

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand the gap between a notebook RAG demo and a production system | 65 min | Advanced |
| Design a production RAG architecture with service boundaries | | |
| Implement index versioning with blue-green deployment | | |
| Build a multi-layer caching strategy | | |
| Configure structured logging and observability dashboards | | |
| Control costs with model routing and context truncation | | |

---

## Intuition First

A notebook RAG demo has one job: prove the concept works. A production RAG system has a completely different set of jobs:

- It must work for 1,000 concurrent users, not one
- It must keep answering while documents are being updated
- It must stay within a monthly budget
- It must log every query so you can debug failures at 2 a.m.
- It must degrade gracefully when the LLM API returns errors
- It must never expose sensitive user data in logs or caches

None of these concerns appear in "Step 3: Run the notebook." This lesson closes that gap.

The mental model: think of your RAG system as three cooperating services — **ingestion**, **retrieval**, and **generation** — each with its own scaling, caching, and failure modes.

---

## Production vs Prototype

| Concern | Prototype (notebook) | Production |
|---------|---------------------|------------|
| Index updates | Manual cell re-run | Automated pipeline on doc changes |
| Multiple users | One at a time | Async, connection pooled |
| Monitoring | None | Retrieval + generation metrics dashboards |
| Caching | None | Query embedding, retrieval, and response caches |
| Error handling | Exception → crash | Graceful fallback, retry with backoff |
| Security | Open API keys in code | Auth, PII filtering, audit logs |
| Cost | Ignored | Budget alerts, model routing, token limits |
| Index changes | Replace everything | Blue-green versioning, rollback |
| Latency | Acceptable for dev | P99 < 3s for user-facing |

!!! tip "Incremental hardening"
    Ship a minimal working RAG pipeline first. Then add one production layer at a time: caching → monitoring → auth → versioning. Trying to build all layers before the first user is the fastest path to never launching.

---

## Production Architecture

```
                    ┌─────────────────┐
  User ──→ API ────→│  Query Router   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ↓              ↓              ↓
         Cache layer    Embed service   Auth / rate limit
              │              │
              └──────┬───────┘
                     ↓
              Retrieval service
                     │
         ┌───────────┼───────────┐
         ↓           ↓           ↓
    Vector DB   Metadata DB   Re-ranker (optional)
                     │
                     ↓
              Generation service
                     │
                     ↓
              Response + citations
                     │
                     ↓
              Observability (traces, logs, metrics)
```

**Service responsibilities:**

| Service | Responsibility | Technology options |
|---------|---------------|-------------------|
| Query Router | Parse request, check cache, route | FastAPI, Flask |
| Embed Service | Embed queries (separate from doc embedding) | Any embedding API or local model |
| Retrieval Service | Vector search, metadata filtering, re-ranking | Pinecone, Weaviate, pgvector |
| Generation Service | LLM call, prompt assembly, response formatting | OpenAI, Anthropic, local |
| Cache Layer | Response, retrieval, and embedding caches | Redis, Memcached |
| Observability | Traces, metrics, cost tracking | Langfuse, OpenTelemetry |

---

## Index Versioning and Updates

Documents change — pages get updated, policies get revised, products are discontinued. Your index must keep pace without downtime.

### The Blue-Green Pattern

Run two indexes simultaneously. Build and validate the new index in "staging" (green) while "production" (blue) serves live traffic. Swap atomically only after validation passes.

```python
import hashlib
from datetime import datetime, timezone

ACTIVE_INDEX = "docs_v3"       # serving live traffic
STAGING_INDEX = "docs_v4"      # being built

def build_new_index(documents: list[dict], embedder, vector_store) -> str:
    """Build a new versioned index from the current document set."""
    version = datetime.now(timezone.utc).strftime("docs_v%Y%m%d_%H%M")
    
    for doc in documents:
        chunks = chunk_text(doc["text"])
        embeddings = embedder.encode(chunks)
        vector_store.upsert(
            index_name=version,
            vectors=[
                {
                    "id": f"{doc['id']}_{i}",
                    "values": emb.tolist(),
                    "metadata": {
                        "doc_id": doc["id"],
                        "text": chunk,
                        "updated_at": doc["updated_at"],
                    },
                }
                for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
            ],
        )
    return version

def validate_and_swap(
    staging_index: str,
    eval_set: list[dict],
    rag_system,
    min_recall: float = 0.80,
) -> bool:
    """Run eval set against staging; swap to production if quality gate passes."""
    metrics = evaluate_retrieval(rag_system, eval_set, index=staging_index)

    if metrics["recall_at_5"] < min_recall:
        alert(
            f"Reindex failed quality gate: Recall@5={metrics['recall_at_5']:.2f} "
            f"(required {min_recall}). Keeping {ACTIVE_INDEX}."
        )
        return False

    # Atomic swap
    global ACTIVE_INDEX
    previous = ACTIVE_INDEX
    ACTIVE_INDEX = staging_index
    
    # Keep previous for rollback window (24h)
    schedule_index_deletion(previous, delay_hours=24)
    
    print(f"Swapped to {staging_index}. Previous: {previous} (rollback available).")
    return True
```

### Incremental Updates

For small document changes (a few pages updated), a full rebuild is wasteful. Incremental updates delete stale chunks and insert fresh ones:

```python
def upsert_document(
    doc_id: str,
    new_text: str,
    embedder,
    vector_store,
    index_name: str,
):
    """Update a single document: delete old chunks, add new ones."""
    # Delete all existing chunks for this document
    vector_store.delete(
        index_name=index_name,
        filter={"doc_id": {"$eq": doc_id}},
    )
    
    # Chunk and embed the new version
    chunks = chunk_text(new_text)
    embeddings = embedder.encode(chunks)
    
    vector_store.upsert(
        index_name=index_name,
        vectors=[
            {
                "id": f"{doc_id}_{i}",
                "values": emb.tolist(),
                "metadata": {"doc_id": doc_id, "text": chunk},
            }
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ],
    )

    print(f"Updated {len(chunks)} chunks for document {doc_id}")
```

**When to use each strategy:**

| Update type | Strategy |
|------------|----------|
| < 5% of documents changed | Incremental upsert |
| > 20% changed, or embedding model upgraded | Full blue-green rebuild |
| New documents added, no deletions | Incremental insert only |
| Schema or chunk size changed | Full rebuild required |

---

## Multi-Layer Caching

Caching in RAG has three distinct layers, each with different TTLs and cache key strategies:

| Cache layer | What to cache | TTL | Cache key |
|-------------|--------------|-----|-----------|
| **Embedding cache** | Query text → vector | 24h | hash(query_text) |
| **Retrieval cache** | Query → chunk IDs | 1h | hash(query + index_version) |
| **Response cache** | Query → full answer | 15 min | hash(query + index_version + model) |

```python
import hashlib
import json
import redis

redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)


def cache_key(namespace: str, *parts: str) -> str:
    """Generate a deterministic cache key."""
    raw = ":".join([namespace] + list(parts))
    return hashlib.sha256(raw.encode()).hexdigest()


def get_embedding_cached(query: str, embedder) -> list[float]:
    """Embed query with caching."""
    key = cache_key("embed", query.lower().strip())
    cached = redis_client.get(key)
    if cached:
        return json.loads(cached)
    
    embedding = embedder.encode([query])[0].tolist()
    redis_client.setex(key, 86400, json.dumps(embedding))  # 24h TTL
    return embedding


def get_retrieval_cached(
    query: str,
    index_version: str,
    retriever,
    k: int = 5,
) -> list[dict]:
    """Retrieve chunks with caching — invalidated when index version changes."""
    key = cache_key("retrieve", query.lower().strip(), index_version, str(k))
    cached = redis_client.get(key)
    if cached:
        return json.loads(cached)
    
    results = retriever.search(query, k=k)
    redis_client.setex(key, 3600, json.dumps(results))  # 1h TTL
    return results


def get_response_cached(
    query: str,
    index_version: str,
    model: str,
    rag_pipeline,
) -> str | None:
    """Response cache — shortest TTL, invalidated by index or model changes."""
    key = cache_key("response", query.lower().strip(), index_version, model)
    return redis_client.get(key)


def set_response_cache(
    query: str,
    index_version: str,
    model: str,
    response: str,
    ttl: int = 900,  # 15 minutes
):
    key = cache_key("response", query.lower().strip(), index_version, model)
    redis_client.setex(key, ttl, response)
```

!!! warning "Index version in cache keys"
    Always include `index_version` in retrieval and response cache keys. Without it, a user who queries right after a reindex gets stale answers from the old cache until TTL expires. This bug is subtle and hard to debug without version-aware keys.

**Cache hit rates to aim for:**
- Embedding cache: 40–60% (users repeat similar queries)
- Retrieval cache: 20–40% (more variation in phrasing)
- Response cache: 10–25% (identical queries from different users)

At scale, each cache hit saves approximately: 50ms (embed) + 150ms (retrieve) + 1,000ms (generate) + cost of API calls.

---

## Async RAG Pipeline

Serial execution (embed → retrieve → generate) is slow. Most steps are I/O-bound — parallelize where possible:

```python
import asyncio
import time
from typing import Any

async def embed_async(query: str, embedder) -> list[float]:
    return await asyncio.to_thread(embedder.encode, [query])

async def retrieve_async(query_vector: list[float], vector_store, k: int = 5) -> list[dict]:
    return await asyncio.to_thread(vector_store.query, query_vector, k)

async def generate_async(prompt: str, client) -> str:
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

async def rag_pipeline_async(
    query: str,
    embedder,
    vector_store,
    client,
    index_version: str,
) -> dict[str, Any]:
    start = time.perf_counter()

    # Check response cache first
    cached = get_response_cached(query, index_version, "gpt-4o-mini", None)
    if cached:
        return {"answer": cached, "cache_hit": True, "latency_ms": 0}

    # Embed (with embedding cache)
    t0 = time.perf_counter()
    query_vector = get_embedding_cached(query, embedder)
    embed_ms = (time.perf_counter() - t0) * 1000

    # Retrieve (with retrieval cache)
    t1 = time.perf_counter()
    chunks = get_retrieval_cached(query, index_version, vector_store)
    retrieve_ms = (time.perf_counter() - t1) * 1000

    # Generate
    context = "\n\n".join(c["text"] for c in chunks)
    prompt = f"Answer the question using only the context.\n\nContext:\n{context}\n\nQuestion: {query}"
    
    t2 = time.perf_counter()
    answer = await generate_async(prompt, client)
    generate_ms = (time.perf_counter() - t2) * 1000

    total_ms = (time.perf_counter() - start) * 1000

    # Cache response
    set_response_cache(query, index_version, "gpt-4o-mini", answer)

    return {
        "answer": answer,
        "cache_hit": False,
        "latency_ms": {
            "embed": round(embed_ms),
            "retrieve": round(retrieve_ms),
            "generate": round(generate_ms),
            "total": round(total_ms),
        },
        "chunks_used": len(chunks),
    }
```

---

## Observability

Structured logs enable dashboards, alerts, and post-mortem debugging. Log every RAG request as a single structured event:

```python
import logging
import json
import uuid

logger = logging.getLogger("rag.production")

def log_rag_request(
    query: str,
    answer: str,
    chunk_ids: list[str],
    retrieval_scores: list[float],
    latency_ms: dict,
    cache_hit: bool,
    index_version: str,
    model: str,
    token_usage: dict,
    user_id: str | None = None,
):
    log_entry = {
        "request_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        # Sanitize: never log raw query if PII is possible
        "query_hash": hashlib.sha256(query.lower().strip().encode()).hexdigest(),
        "query_length": len(query),
        "retrieved_chunk_ids": chunk_ids,
        "retrieval_scores": [round(s, 4) for s in retrieval_scores],
        "latency_ms": latency_ms,
        "cache_hit": cache_hit,
        "index_version": index_version,
        "model": model,
        "token_usage": token_usage,
        "user_id": user_id,  # anonymize if needed
    }
    logger.info(json.dumps(log_entry))
```

### Key Metrics Dashboard

| Metric | How to compute | Alert threshold |
|--------|---------------|-----------------|
| Retrieval latency p99 | histogram on `latency_ms.retrieve` | > 500ms |
| Empty retrieval rate | count(chunk_ids=[]) / total | > 5% |
| Cache hit rate | count(cache_hit=True) / total | < 10% → investigate |
| Total cost / day | sum(token_usage.total) × price / 1M | > budget |
| Faithfulness score (sampled) | LLM judge on 1% of queries | < 0.75 (7-day avg) |
| Error rate | count(status=error) / total | > 1% |

```python
# Prometheus metrics example
from prometheus_client import Histogram, Counter, Gauge

rag_latency = Histogram(
    "rag_latency_ms",
    "RAG request latency",
    ["stage"],  # embed, retrieve, generate, total
    buckets=[50, 100, 250, 500, 1000, 2000, 5000],
)

rag_cache_hits = Counter("rag_cache_hits_total", "Cache hits", ["layer"])
rag_retrieval_empty = Counter("rag_retrieval_empty_total", "Queries with empty retrieval")
rag_cost = Counter("rag_token_cost_usd_total", "Cumulative token cost", ["model"])
```

---

## Cost Control

### Model Routing

Route simple queries to cheaper models; complex queries to more capable (and expensive) models:

```python
def classify_query_complexity(query: str) -> str:
    """Simple heuristic classifier — replace with a trained classifier for production."""
    word_count = len(query.split())
    has_comparison = any(w in query.lower() for w in ["compare", "difference", "versus", "vs", "contrast"])
    has_multi_part = "and" in query.lower() and "?" in query
    
    if word_count > 30 or has_comparison or has_multi_part:
        return "complex"
    return "simple"

def route_model(query: str) -> str:
    complexity = classify_query_complexity(query)
    if complexity == "simple":
        return "gpt-4o-mini"     # ~$0.15 / 1M input tokens
    return "gpt-4o"              # ~$2.50 / 1M input tokens

# Expected cost reduction: if 70% of queries are "simple", routing saves ~60% on generation costs
```

### Context Truncation

Don't send more context than needed. Unused tokens cost money and can degrade quality:

```python
def truncate_context(
    chunks: list[str],
    query: str,
    max_tokens: int = 3_000,
    chars_per_token: float = 4.0,
) -> list[str]:
    """
    Include as many chunks as fit in the token budget.
    Prefer shorter, higher-ranked chunks over longer, lower-ranked ones.
    """
    max_chars = int(max_tokens * chars_per_token)
    result: list[str] = []
    used_chars = len(query)

    for chunk in chunks:
        chunk_chars = len(chunk)
        if used_chars + chunk_chars > max_chars:
            # Try to include a truncated version if we're at the first chunk
            if not result:
                result.append(chunk[:max_chars - used_chars - 100] + "…")
            break
        result.append(chunk)
        used_chars += chunk_chars

    return result

# Token cost example:
# 5 chunks × 500 tokens = 2,500 context tokens per query
# Truncating to 3 relevant chunks × 400 tokens = 1,200 tokens
# At gpt-4o pricing ($2.50/1M input): saves $0.00325 per query
# At 100k queries/day: saves $325/day
```

---

## Security Checklist

**PII filtering (ingestion):**

```python
import re

PII_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN REDACTED]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL REDACTED]"),
    (re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"), "[CARD REDACTED]"),
]

def filter_pii(text: str) -> str:
    for pattern, replacement in PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text
```

**Document access control (retrieval):**

```python
def retrieve_with_access_control(
    query_vector: list[float],
    vector_store,
    user_permissions: set[str],
    k: int = 5,
) -> list[dict]:
    """Only return documents the user has permission to read."""
    results = vector_store.query(
        vector=query_vector,
        k=k * 3,  # over-fetch to account for filtered docs
        filter={"allowed_groups": {"$in": list(user_permissions)}},
    )
    return results[:k]
```

**Additional security controls:**
- Never log raw queries that may contain sensitive user data — log query hashes for correlation
- Rate-limit API endpoints per authenticated user (e.g., 100 queries/hour free tier)
- Rotate API keys automatically; never hardcode in source files
- Audit trail for all index changes: who triggered, when, what document

---

## Launch Checklist

- [ ] Eval set with 100+ labeled queries passing all quality gates (Recall@5 ≥ 0.80, faithfulness ≥ 0.80)
- [ ] Index versioning with blue-green deployment and rollback capability
- [ ] Multi-layer caching (embedding + retrieval + response) with index-version-aware keys
- [ ] Structured logging with at minimum: latency, token usage, cache hit, index version
- [ ] Monitoring dashboards for latency (p50/p99), cost per query, empty retrieval rate
- [ ] Graceful fallback response when retrieval returns empty results
- [ ] PII filtering in document ingestion pipeline
- [ ] Access control enforcement in retrieval queries
- [ ] Model routing for cost optimization
- [ ] Load test to expected QPS before go-live (use k6, Locust, or Artillery)
- [ ] Runbook documenting: how to rollback index, how to investigate latency spikes, alert response playbook

---

## Common Misconceptions

**"Caching the response is enough."** Response caches have low hit rates (10–25%) because users rarely send identical queries. Embedding and retrieval caches are often more valuable — they save the slow steps without requiring exact query matches.

**"Blue-green is only for big deployments."** Even a 100-document internal knowledge base benefits from blue-green index updates. A bad reindex (wrong embedding model, broken chunking) can silently serve wrong answers for hours if you can't roll back.

**"More logging means slower queries."** Async structured logging (fire-and-forget to a log aggregator) adds < 2ms to query latency. The insight from structured logs is worth orders of magnitude more than this cost.

**"PII filtering is a one-time setup."** PII patterns change — new regulations, new data formats, new document types. Review your PII filter rules quarterly and run regression tests against known PII examples.

---

## Key Takeaways

- Production RAG systems require six concerns a notebook demo ignores: **multi-user concurrency**, **index versioning**, **caching**, **observability**, **cost control**, and **security**
- **Blue-green index versioning** lets you rebuild, validate, and swap indexes without downtime — with a rollback window
- **Multi-layer caching** (embedding, retrieval, response) with index-version-aware keys reduces latency and cost by 30–60% in typical deployments
- **Structured logging** with consistent fields enables dashboards, alerts, and post-mortem debugging
- **Model routing** (simple → cheap, complex → powerful) is the single highest-leverage cost reduction for most RAG systems
- Ship minimal → add hardening one layer at a time → don't build everything before launch

---

## Related Papers

| Paper | Year | Key contribution |
|-------|------|-----------------|
| [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) | 2020 | Original RAG paper — foundational architecture and training |
| [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059) | 2024 | Hierarchical indexing for long-document RAG at scale |
| [Seven Failure Points When Engineering a Retrieval Augmented Generation System](https://arxiv.org/abs/2401.05856) | 2024 | Taxonomy of production RAG failure modes |
| [HELMET: How to Evaluate Long-Context Language Models Effectively and Thoroughly](https://arxiv.org/abs/2410.02694) | 2024 | Evaluation methodology for long-context and RAG systems |

---

## Next Lesson

**[Lesson 11 — Graph RAG and Knowledge Graphs](11-graph-rag-and-knowledge-graphs.md):** Move beyond flat document retrieval to graph-structured knowledge — entity relationships, multi-hop traversal, and when knowledge graphs outperform vector search.
