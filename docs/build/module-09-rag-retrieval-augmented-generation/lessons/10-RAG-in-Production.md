---
title: RAG in Production
description: >-
  Deploy, monitor, and maintain RAG systems at scale — caching, versioning,
  observability, and cost control
duration: 50 min
difficulty: advanced
has_code: false
module: module-09
youtube: 'https://www.youtube.com/watch?v=T-D1OfcDW1M'
---
## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Design production RAG architecture | 50 min | Advanced |
| Implement caching and index versioning | | |
| Monitor retrieval and generation quality | | |
| Control cost and latency at scale | | |

---

## Production vs Prototype

A notebook RAG demo skips everything that matters in production:

| Concern | Prototype | Production |
|---------|-----------|------------|
| Index updates | Manual re-run | Automated pipeline on doc changes |
| Monitoring | None | Retrieval + generation metrics |
| Caching | None | Query + embedding cache |
| Error handling | Crash | Graceful fallback |
| Security | Open API keys | Auth, PII filtering, audit logs |
| Cost | Ignored | Budget alerts, model routing |

> **Tip:** Ship a minimal RAG pipeline first, then add production hardening one layer at a time. Don't build everything before launch.

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
    Vector DB   Metadata DB   Re-ranker
                     │
                     ↓
              Generation service
                     │
                     ↓
              Response + citations
                     │
                     ↓
              Observability (logs, traces, metrics)
```

---

## Index Versioning and Updates

Documents change. Your index must keep up without downtime.

```python
# Blue-green index pattern
ACTIVE_INDEX = "docs_v3"
STAGING_INDEX = "docs_v4"

def reindex_pipeline(documents: list[dict]):
    """Build new index in staging, validate, then swap."""
    for doc in documents:
        chunks = chunk_text(doc["text"])
        index_to(STAGING_INDEX, doc["id"], chunks)

    # Validate on eval set before swap
    metrics = evaluate_retrieval(STAGING_INDEX, eval_set)
    if metrics["recall_at_5"] >= 0.80:
        swap_active_index(STAGING_INDEX)
    else:
        alert("Reindex failed quality gate — keeping current index")
```

**Incremental updates** for small changes:

```python
def upsert_document(doc_id: str, text: str):
    delete_chunks(doc_id)          # Remove stale chunks
    chunks = chunk_text(text)
    index_chunks(doc_id, chunks)   # Add fresh chunks
```

---

## Caching Strategy

| Cache layer | What to cache | TTL |
|-------------|--------------|-----|
| Embedding cache | Query → vector | 24h (queries repeat) |
| Retrieval cache | Query → chunk IDs | 1h (index may update) |
| Response cache | Query → full answer | 15m (for identical queries) |

```python
import hashlib
import json

def cache_key(query: str, index_version: str) -> str:
    raw = f"{index_version}:{query.lower().strip()}"
    return hashlib.sha256(raw.encode()).hexdigest()

# Redis example
def get_cached_answer(query: str) -> str | None:
    key = cache_key(query, ACTIVE_INDEX)
    return redis.get(key)

def set_cached_answer(query: str, answer: str, ttl: int = 900):
    key = cache_key(query, ACTIVE_INDEX)
    redis.setex(key, ttl, answer)
```

> **Warning:** Cache invalidation must include index version in the key. Otherwise users get stale answers after reindexing.

---

## Observability

Log these fields for every RAG request:

```python
log_entry = {
    "query": user_query,
    "retrieved_chunk_ids": chunk_ids,
    "retrieval_scores": scores,
    "model": "gpt-4o-mini",
    "latency_ms": {
        "embed": 45,
        "retrieve": 120,
        "generate": 890,
        "total": 1055,
    },
    "token_usage": {"input": 2400, "output": 180},
    "cache_hit": False,
    "index_version": ACTIVE_INDEX,
}
```

**Alerts to configure:**

| Signal | Threshold | Action |
|--------|-----------|--------|
| Retrieval latency p99 | > 500ms | Scale vector DB |
| Empty retrieval rate | > 10% | Check index health |
| Faithfulness score drop | > 5% week-over-week | Review recent index changes |
| Cost per query | > budget | Enable response caching |

---

## Cost Control

```python
def route_model(query: str, complexity: str) -> str:
    """Use cheaper models for simple queries."""
    if complexity == "simple" or len(query) < 50:
        return "gpt-4o-mini"      # ~$0.15/1M input tokens
    return "gpt-4o"               # ~$2.50/1M input tokens

def truncate_context(chunks: list[str], max_tokens: int = 3000) -> str:
    """Don't send more context than needed."""
    result = []
    total = 0
    for chunk in chunks:
        tokens = len(chunk.split()) * 1.3  # rough estimate
        if total + tokens > max_tokens:
            break
        result.append(chunk)
        total += tokens
    return "

".join(result)
```

---

## Security Checklist

- Filter PII before indexing (emails, SSNs, phone numbers)
- Enforce document-level access control in retrieval queries
- Never log raw user queries containing sensitive data
- Rate-limit API endpoints per user/IP
- Audit trail for index changes and admin actions

---

## Launch Checklist

- [ ] Eval set with 50+ labeled queries passing quality gates
- [ ] Index versioning with rollback capability
- [ ] Monitoring dashboards for latency, cost, and quality
- [ ] Caching with index-version-aware keys
- [ ] Graceful fallback when retrieval returns empty results
- [ ] PII filtering in ingestion pipeline
- [ ] Load tested to expected QPS

---

## Recommended Videos

- [Production RAG Systems](https://www.youtube.com/watch?v=T-D1OfcDW1M)
- [RAG at Scale](https://www.youtube.com/watch?v=wdK68K1FuHk)

---

## Additional Resources

- [LangSmith Production Guide](https://docs.smith.langchain.com/)
- [Pinecone Production Best Practices](https://www.pinecone.io/learn/production/)
- [OpenAI Production Safety](https://platform.openai.com/docs/guides/safety-best-practices)
