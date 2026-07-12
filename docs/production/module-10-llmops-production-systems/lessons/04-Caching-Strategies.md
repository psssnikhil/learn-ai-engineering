---
title: Caching Strategies for LLM Applications
description: >-
  Learn how to implement semantic caching, exact-match caching, and tiered
  caching to reduce latency and costs in LLM apps
duration: 30 min
difficulty: intermediate
has_code: false
module: module-10
objectives:
  - Explain the difference between exact-match and semantic caching
  - Implement an in-memory LLM response cache
  - Design a cache key strategy for chat completions
  - Understand cache invalidation challenges with LLMs
  - Measure cache hit rates and cost savings
---
# Caching Strategies for LLM Applications

## What You'll Learn

By the end of this lesson, you'll understand:
- Why caching is critical for LLM applications
- Exact-match vs. semantic caching approaches
- How to design effective cache keys for LLM calls
- Tiered caching strategies (memory, Redis, disk)
- Cache invalidation and TTL strategies

**Time to Complete**: 30 minutes
**Difficulty**: Intermediate

---

## Why Cache LLM Responses?

LLM API calls are:
- **Slow**: 500ms to 30+ seconds depending on model and output length
- **Expensive**: GPT-4o costs ~$5/million input tokens, $15/million output tokens
- **Rate-limited**: Most providers cap requests per minute
- **Deterministic enough**: Similar inputs often produce acceptable cached responses

A well-designed cache can:
- **Reduce latency by 100x** (cache hit: <10ms vs. API call: 1-5s)
- **Cut costs by 50-80%** for repetitive queries
- **Handle traffic spikes** without hitting rate limits

---

## Exact-Match Caching

The simplest approach: if you've seen this exact request before, return the cached response.

### Cache Key Design

```python
import hashlib
import json

def make_cache_key(model: str, messages: list, temperature: float, **kwargs) -> str:
    """Create a deterministic cache key from request parameters."""
    key_data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": kwargs.get("max_tokens"),
        "top_p": kwargs.get("top_p"),
    }
    key_string = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_string.encode()).hexdigest()
```

### In-Memory Cache

```python
from datetime import datetime, timedelta
from typing import Optional

class LLMCache:
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 1000):
        self._cache = {}
        self._ttl = timedelta(seconds=ttl_seconds)
        self._max_size = max_size
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[str]:
        if key in self._cache:
            entry = self._cache[key]
            if datetime.now() - entry["created_at"] < self._ttl:
                self.hits += 1
                return entry["response"]
            else:
                del self._cache[key]
        self.misses += 1
        return None

    def set(self, key: str, response: str):
        if len(self._cache) >= self._max_size:
            oldest_key = min(self._cache, key=lambda k: self._cache[k]["created_at"])
            del self._cache[oldest_key]
        self._cache[key] = {"response": response, "created_at": datetime.now()}

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
```

### Cached LLM Client

```python
class CachedLLMClient:
    def __init__(self, client, cache: LLMCache):
        self.client = client
        self.cache = cache

    def chat_completion(self, model, messages, temperature=0, **kwargs):
        if temperature == 0:
            key = make_cache_key(model, messages, temperature, **kwargs)
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        response = self.client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, **kwargs
        )
        result = response.choices[0].message.content

        if temperature == 0:
            self.cache.set(key, result)
        return result
```

---

## Semantic Caching

Exact-match caching misses near-identical queries. Semantic caching uses embeddings to find similar previous queries.

```
User A: "What is machine learning?"
User B: "Can you explain machine learning?"
User C: "Define ML for me"
  All three should hit the same cache entry
```

### How It Works

1. Embed the query using a fast embedding model
2. Search the cache for similar embeddings (cosine similarity)
3. If similarity > threshold, return the cached response
4. Otherwise, call the LLM and cache the result with its embedding

```python
import numpy as np

class SemanticCache:
    def __init__(self, embedding_client, similarity_threshold=0.95):
        self.embedding_client = embedding_client
        self.threshold = similarity_threshold
        self.entries = []

    def _get_embedding(self, text: str) -> list[float]:
        response = self.embedding_client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return response.data[0].embedding

    def _cosine_similarity(self, a, b) -> float:
        a, b = np.array(a), np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get(self, query: str) -> Optional[str]:
        query_embedding = self._get_embedding(query)
        best_score, best_response = 0, None

        for embedding, cached_query, response in self.entries:
            score = self._cosine_similarity(query_embedding, embedding)
            if score > best_score:
                best_score = score
                best_response = response

        return best_response if best_score >= self.threshold else None

    def set(self, query: str, response: str):
        embedding = self._get_embedding(query)
        self.entries.append((embedding, query, response))
```

### Trade-offs

| Approach | Hit Rate | Latency Overhead | Complexity |
|----------|----------|-----------------|------------|
| Exact-match | Low-Medium | None (hash lookup) | Simple |
| Semantic | High | ~50ms (embedding) | Moderate |
| Hybrid | Highest | Conditional | Higher |

---

## Tiered Caching

Combine multiple cache layers for the best performance:

```
Request -> L1 (In-Memory) -> L2 (Redis) -> L3 (LLM API)
              ~0.1ms           ~2ms          ~1-5s
```

---

## Cache Invalidation

### Strategies

1. **Time-based TTL**: Short TTL (1h) for changing data, long TTL (24h+) for stable queries
2. **Version-based**: Invalidate when the prompt version changes
3. **Content-based**: Invalidate when source data changes (for RAG)
4. **Manual flush**: Admin endpoint to clear on demand

---

## Resources

- **GPTCache** — Open-source semantic caching library for LLMs
- **Redis** — Industry standard for distributed caching
- **LangChain CacheBackedEmbeddings** — Built-in caching for embeddings

---

## Key Takeaways

1. **Cache deterministic calls** (temperature=0) for the highest hit rates
2. **Semantic caching** dramatically increases hit rates for similar queries
3. **Tiered caching** combines speed (memory) with scale (Redis)
4. **Cache keys must include** all parameters that affect output
5. **Monitor hit rates** to quantify your cost and latency savings
