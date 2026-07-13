---
title: Caching Strategies for LLM Applications
description: >-
  Learn how to implement semantic caching, exact-match caching, and tiered
  caching to reduce latency and costs in LLM apps
duration: 40 min
difficulty: intermediate
has_code: true
module: module-10
---
# Caching Strategies for LLM Applications

## Prerequisites

- Completed Lessons 1–3 (LLMOps Introduction, Observability, Prompt Versioning)
- Familiarity with Python dicts and basic hashing
- Basic understanding of what embeddings are (vectors representing text semantics)

---

## What You'll Learn

| Objective | Outcome |
|-----------|---------|
| Explain why LLM API calls are uniquely worth caching | Can calculate the cost and latency savings potential |
| Design cache keys that correctly identify equivalent requests | Can avoid both false hits and missed hits |
| Implement exact-match caching with TTL and eviction | Can deploy a working in-memory cache for any LLM app |
| Build a semantic cache using embeddings | Can serve cached responses to paraphrased queries |
| Choose a tiered caching architecture | Can pick the right layer for any workload |

---

## Intuition First: Why Cache LLM Responses?

Every LLM API call has three undesirable properties that caching can address:

**Slow**: A GPT-4o call takes 1–10 seconds depending on output length. A cache hit returns in under 1 millisecond — a 1,000–10,000x speedup for the user.

**Expensive**: At 2.50 per million input tokens and 10.00 per million output tokens (GPT-4o), a typical customer support query costs 0.01–0.05 per request. At 100,000 requests/day, that's $1,000–$5,000 daily. A 50% cache hit rate cuts this in half.

**Rate-limited**: Most LLM providers cap you at some requests-per-minute limit. A traffic spike that would blow through your rate limit gets absorbed by the cache.

The fundamental trade-off: **freshness vs. cost**. A cached response may not reflect the very latest information or the exact model behavior today. For many applications (FAQ answers, document summaries, classification results), this trade-off heavily favors caching.

```
Query: "What are your business hours?"
→ Without cache: $0.003, 1.4 seconds
→ With cache hit: $0.000, 0.8 milliseconds

For 10,000 queries/month where 70% are cacheable:
→ Without cache: $30/month, avg 1.4s latency
→ With cache:    $9/month (savings: $21), avg 0.7s latency (mix of hits and misses)
```

---

## Exact-Match Caching

The simplest approach: if you've seen this *exact* request before, return the stored response.

### Cache Key Design

A cache key must uniquely identify all parameters that affect the LLM output. Miss any parameter and you get false cache hits (serving wrong responses); include too many and you get a near-zero hit rate.

```python
import hashlib
import json

def make_cache_key(
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int | None = None,
    top_p: float | None = None,
    system_prompt_version: str | None = None,
) -> str:
    """
    Deterministic cache key from all parameters that affect the LLM output.

    Important: Include the prompt version! If you update the system prompt
    but not the user message, the cache should be invalidated.
    """
    key_data = {
        "model": model,
        "messages": messages,          # Full message list, not just the last user message
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "prompt_version": system_prompt_version,  # Include version for invalidation
    }
    # Remove None values to avoid key instability
    key_data = {k: v for k, v in key_data.items() if v is not None}
    key_string = json.dumps(key_data, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(key_string.encode()).hexdigest()
```

!!! note "Only Cache Deterministic Calls"
    Only cache requests with `temperature=0` (or very low temperature). High-temperature requests intentionally produce varied outputs — caching defeats that purpose and produces a misleadingly "sticky" creative experience.

### Full In-Memory Cache Implementation

```python
from datetime import datetime, timedelta
from typing import Optional
from dataclasses import dataclass
import threading

@dataclass
class CacheEntry:
    response: str
    created_at: datetime
    hit_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

class LLMCache:
    """
    Thread-safe in-memory LLM response cache with TTL and LRU eviction.
    """

    def __init__(self, ttl_seconds: int = 3600, max_size: int = 5_000):
        self._cache: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self._ttl = timedelta(seconds=ttl_seconds)
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
        self._total_saved_usd = 0.0

    def get(self, key: str) -> Optional[str]:
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None
            entry = self._cache[key]
            if datetime.now() - entry.created_at > self._ttl:
                del self._cache[key]
                self._misses += 1
                return None
            entry.hit_count += 1
            self._hits += 1
            self._total_saved_usd += entry.cost_usd
            return entry.response

    def set(self, key: str, response: str,
            input_tokens: int = 0, output_tokens: int = 0, cost_usd: float = 0.0):
        with self._lock:
            if len(self._cache) >= self._max_size:
                self._evict_oldest()
            self._cache[key] = CacheEntry(
                response=response,
                created_at=datetime.now(),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
            )

    def _evict_oldest(self):
        """Remove the oldest 10% of entries when at capacity."""
        sorted_keys = sorted(
            self._cache.keys(),
            key=lambda k: self._cache[k].created_at
        )
        for key in sorted_keys[:len(sorted_keys) // 10]:
            del self._cache[key]

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self.hit_rate:.1%}",
            "size": len(self._cache),
            "estimated_savings_usd": round(self._total_saved_usd, 4),
        }


class CachedLLMClient:
    """Wraps any OpenAI-compatible client with transparent caching."""

    def __init__(self, client, cache: LLMCache):
        self.client = client
        self.cache = cache

    def chat_completion(self, model: str, messages: list[dict],
                        temperature: float = 0.0, **kwargs) -> str:
        # Only cache deterministic requests
        should_cache = temperature == 0.0 or temperature < 0.1

        if should_cache:
            key = make_cache_key(model, messages, temperature, **kwargs)
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        response = self.client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, **kwargs
        )
        result = response.choices[0].message.content
        usage = response.usage

        cost = (usage.prompt_tokens / 1e6 * 0.15 +
                usage.completion_tokens / 1e6 * 0.60)  # gpt-4o-mini rates

        if should_cache:
            self.cache.set(key, result,
                           input_tokens=usage.prompt_tokens,
                           output_tokens=usage.completion_tokens,
                           cost_usd=cost)
        return result
```

---

## Semantic Caching

Exact-match caching only helps when users ask *exactly* the same question. In practice, users paraphrase:

```
User A: "What are your business hours?"
User B: "When are you open?"
User C: "What time does your office close?"
User D: "Are you open on Saturdays?"
```

These four queries have different text but the same (or very similar) intent. A semantic cache uses embeddings to find previously answered queries that are *semantically similar*, not just textually identical.

### How Semantic Caching Works

```
New query: "When do you close on Fridays?"
    ↓
Embed query → vector [0.23, -0.11, 0.87, ...]
    ↓
Search existing cache entries by cosine similarity
    ↓
Find: "What are your closing hours?" → similarity 0.97
    ↓
0.97 > threshold (0.95) → Return cached response
```

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class SemanticCacheEntry:
    embedding: list[float]
    query: str
    response: str
    created_at: datetime
    cost_usd: float = 0.0

class SemanticCache:
    """
    Semantic cache using embedding similarity.
    Embedding cost (~$0.00002) is far lower than LLM inference cost (~$0.01+),
    making this cost-effective even when the embedding call adds latency.
    """

    def __init__(self, embedding_client, similarity_threshold: float = 0.95,
                 ttl_seconds: int = 3600):
        self.embedding_client = embedding_client
        self.threshold = similarity_threshold
        self._ttl = timedelta(seconds=ttl_seconds)
        self.entries: list[SemanticCacheEntry] = []
        self._hits = 0
        self._misses = 0

    def _embed(self, text: str) -> list[float]:
        response = self.embedding_client.embeddings.create(
            model="text-embedding-3-small", input=text
        )
        return response.data[0].embedding

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        a_arr, b_arr = np.array(a), np.array(b)
        norm_product = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        if norm_product == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / norm_product)

    def _prune_expired(self):
        now = datetime.now()
        self.entries = [e for e in self.entries
                        if now - e.created_at < self._ttl]

    def get(self, query: str) -> Optional[str]:
        self._prune_expired()
        if not self.entries:
            self._misses += 1
            return None

        query_embedding = self._embed(query)
        best_score = 0.0
        best_entry = None

        for entry in self.entries:
            score = self._cosine_similarity(query_embedding, entry.embedding)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= self.threshold:
            self._hits += 1
            return best_entry.response

        self._misses += 1
        return None

    def set(self, query: str, response: str, cost_usd: float = 0.0):
        embedding = self._embed(query)
        self.entries.append(SemanticCacheEntry(
            embedding=embedding,
            query=query,
            response=response,
            created_at=datetime.now(),
            cost_usd=cost_usd,
        ))

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
```

### Threshold Calibration

The `similarity_threshold` is the most important tuning parameter:

| Threshold | Effect | Risk |
|-----------|--------|------|
| 0.99 | Very conservative — only near-identical queries hit cache | Low hit rate, misses most paraphrases |
| 0.95 | Balanced — catches clear paraphrases | Occasional false hits on similar-sounding but different questions |
| 0.90 | Aggressive — catches loose paraphrases | Higher false hit rate; test thoroughly |
| < 0.85 | Too aggressive for most applications | High false hit rate; users get wrong answers |

**Calibration process**: Take 200 real query pairs from your logs. Label each as "should hit cache" or "should not hit cache." Run at different thresholds and pick the one that maximizes hits while keeping false positives under 1%.

---

## Tiered Caching

Combine layers for maximum coverage with minimum complexity:

```
User Request
     ↓
[L1: In-Memory Cache]  — 0.5ms hit latency, 1,000 entry capacity
     ↓ miss
[L2: Redis Cache]      — 2ms hit latency, unlimited capacity, shared across instances
     ↓ miss
[L3: Semantic Cache]   — 50ms (embedding call), catches paraphrases
     ↓ miss
[LLM API Call]         — 1,000–10,000ms, full cost
     ↓
Store result in all layers
```

```python
class TieredCache:
    """
    Three-tier cache: L1 in-memory → L2 Redis → L3 semantic.
    Each tier handles a different slice of the hit distribution.
    """

    def __init__(self, l1: LLMCache, l2_redis, l3: SemanticCache):
        self.l1 = l1
        self.l2 = l2_redis     # Standard Redis client
        self.l3 = l3

    def get(self, query: str, cache_key: str) -> tuple[Optional[str], str]:
        """
        Returns (response, tier_hit) where tier_hit is 'l1', 'l2', 'l3', or 'miss'.
        """
        # L1: exact in-memory
        result = self.l1.get(cache_key)
        if result:
            return result, "l1"

        # L2: exact Redis (shared across app instances)
        result = self.l2.get(cache_key)
        if result:
            self.l1.set(cache_key, result)    # Promote to L1
            return result.decode(), "l2"

        # L3: semantic (catches paraphrases)
        result = self.l3.get(query)
        if result:
            self.l1.set(cache_key, result)    # Promote to L1
            return result, "l3"

        return None, "miss"

    def set(self, query: str, cache_key: str, response: str,
            cost_usd: float = 0.0, l2_ttl: int = 3600):
        self.l1.set(cache_key, response, cost_usd=cost_usd)
        self.l2.setex(cache_key, l2_ttl, response)
        self.l3.set(query, response, cost_usd=cost_usd)
```

---

## Cache Invalidation Strategies

Cache invalidation is notoriously hard. For LLM applications, four strategies cover most cases:

**Time-based TTL**: The simplest approach. Choose TTL based on how quickly the underlying data changes:

| Content Type | Recommended TTL |
|-------------|-----------------|
| FAQ answers (stable policies) | 24–72 hours |
| Product info (changes rarely) | 4–12 hours |
| Customer account data | 5–15 minutes |
| Real-time pricing | Do not cache |
| News/current events | Do not cache |

**Version-based invalidation**: When you update a prompt version, the cache key should include the version number. Old keys naturally expire; new ones start fresh.

**Content-based invalidation**: For RAG applications, invalidate cache entries when source documents change. Tag each cache entry with the document IDs it used; when a document is updated, expire all tagged entries.

**Manual flush**: Implement an admin endpoint to clear the cache on demand. Useful after data migrations or emergency fixes.

```python
def invalidate_by_prompt_version(cache: LLMCache, prompt_name: str, old_version: int):
    """
    In production, cache keys include prompt_version in their hash.
    Simply changing the prompt version auto-invalidates old cache entries.
    No explicit flush needed — old entries expire via TTL.
    Log the expected cache cold-start period after a prompt update.
    """
    avg_ttl_hours = cache._ttl.seconds / 3600
    print(f"Prompt {prompt_name} updated to v{old_version + 1}.")
    print(f"Old cache entries will expire within {avg_ttl_hours:.1f} hours.")
    print(f"Expect elevated miss rates and API costs for ~{avg_ttl_hours:.1f} hours.")
```

---

## Worked Example: Measuring Cache Effectiveness

After deploying caching, monitor these numbers to validate it's working:

```python
def compute_cache_report(cache: LLMCache, daily_request_volume: int,
                          avg_cost_per_call: float) -> dict:
    hit_rate = cache.hit_rate
    daily_llm_calls = daily_request_volume * (1 - hit_rate)
    daily_cache_hits = daily_request_volume * hit_rate
    daily_cost_uncached = daily_request_volume * avg_cost_per_call
    daily_cost_with_cache = daily_llm_calls * avg_cost_per_call
    monthly_savings = (daily_cost_uncached - daily_cost_with_cache) * 30

    return {
        "hit_rate": f"{hit_rate:.1%}",
        "daily_llm_calls_avoided": int(daily_cache_hits),
        "daily_cost_uncached": f"${daily_cost_uncached:.2f}",
        "daily_cost_with_cache": f"${daily_cost_with_cache:.2f}",
        "daily_savings": f"${daily_cost_uncached - daily_cost_with_cache:.2f}",
        "monthly_savings": f"${monthly_savings:.2f}",
    }

# Example output for a support bot with 5,000 requests/day:
# hit_rate: 43%
# daily_llm_calls_avoided: 2,150
# daily_cost_uncached: $50.00
# daily_cost_with_cache: $28.50
# monthly_savings: $645.00
```

---

## Edge Cases and Misconceptions

**"Cache all LLM responses by default."**
Only cache responses from low-temperature (deterministic) calls. Creative writing, brainstorming, and any intentionally varied output should not be cached — a cached creative response is a stale one.

**"Semantic caching always increases hit rate."**
Semantic caching adds embedding latency (~50ms) on every request, even misses. If your queries are diverse and novel, the semantic cache will miss most of the time while adding 50ms to every request. Profile your actual query distribution before enabling semantic caching.

**"RAG responses are easy to cache."**
RAG responses are difficult to cache because the retrieved context varies as your knowledge base changes. Cache only when the underlying documents are stable. Consider caching the *retrieval results* (the chunks) separately from the *generation results*, with different TTLs.

**"Cache keys are set and forget."**
Cache key design must be revisited every time you add a parameter that affects output: a new system prompt parameter, a user preference setting, a temperature change. Missing a parameter from the key causes false hits—users getting wrong responses.

---

## Production Scenario: Caching a Customer Support Bot

Your support bot handles 5,000 requests per day at $0.01/request average cost ($50/day). You suspect many queries are repetitive—billing questions, password resets, return policy questions—that could be cached. Here's how you validate and deploy caching incrementally.

### Step 1: Analyze Your Query Distribution Before Building Anything

```python
from collections import Counter
import hashlib

def analyze_query_diversity(queries: list[str]) -> dict:
    """
    Understand your exact-match and near-duplicate distribution
    before committing to a caching strategy.
    """
    normalized = [q.strip().lower() for q in queries]
    exact_counts = Counter(normalized)

    exact_duplicates = sum(v - 1 for v in exact_counts.values() if v > 1)
    unique = len(exact_counts)

    # Rough semantic cluster estimate: queries shorter than 8 words
    # that share the first 4 words are likely near-duplicates
    def prefix(q):
        return " ".join(q.split()[:4])

    prefix_counts = Counter(prefix(q) for q in normalized)
    near_dupes_estimate = sum(v - 1 for v in prefix_counts.values() if v > 1)

    return {
        "total_queries": len(queries),
        "unique_exact": unique,
        "exact_duplicate_pct": round(exact_duplicates / len(queries) * 100, 1),
        "near_duplicate_estimate_pct": round(near_dupes_estimate / len(queries) * 100, 1),
        "top_10_most_common": exact_counts.most_common(10),
    }

# Run on last 7 days of production queries
analysis = analyze_query_diversity(last_7_days_queries)
print(analysis)
# Output:
# exact_duplicate_pct: 12%    → L1 (exact-match) cache saves 12%
# near_duplicate_estimate_pct: 31%  → Semantic cache may save up to 31%
```

### Step 2: Deploy Exact-Match Cache First (Low Risk, Immediate Returns)

Exact-match cache is deterministic and risk-free. Deploy it first:

```
Expected impact:
- Hit rate: 12% (from analysis above)
- Daily LLM calls avoided: 600
- Daily cost saved: $6.00
- Monthly savings: ~$180
- Added latency on miss: 2ms (Redis lookup)
- Added latency on hit: 2ms (no LLM call)
```

### Step 3: Add Semantic Cache at Threshold 0.96

Start with a conservative similarity threshold to minimize false positive risk:

```python
# At threshold 0.96: only serve cached responses for very close paraphrases
# "What is your return policy?" → "What's your return policy?" → HIT
# "What is your return policy?" → "Can I return my purchase?" → MISS

# At threshold 0.90: serve cached responses for semantically similar queries
# "What is your return policy?" → "Can I return my purchase?" → likely HIT
```

Start at 0.96. Sample 50 semantic cache hits per day and rate them with a human reviewer. If false positive rate < 1%, lower threshold to 0.93. Continue calibrating down until false positive rate approaches 1%, then lock the threshold.

### Combined Impact After Full Deployment

| Layer | Hit Rate | Daily Calls Saved | Monthly Savings |
|-------|----------|-------------------|----------------|
| L1 Exact-match | 12% | 600 | $180 |
| L2 Redis (24h TTL) | 8% (additional) | 400 | $120 |
| L3 Semantic (0.93) | 19% (additional) | 950 | $285 |
| **Total** | **39%** | **1,950** | **$585** |

Caching reduces your monthly LLM cost from $1,500 to $915 — a 39% reduction — with no change to the user experience.

---

## Key Takeaways

- LLM API calls are slow (1–10s), expensive ($0.001–0.10/request), and rate-limited—all three problems are mitigated by caching
- Only cache requests with `temperature=0`; high-temperature calls intentionally vary and caching defeats their purpose
- Cache keys must include every parameter that affects the output: model, messages, temperature, max_tokens, and prompt version
- Semantic caching uses embedding similarity to serve cached responses to paraphrased queries; calibrate the threshold with real query pairs
- Tiered caching (in-memory → Redis → semantic) combines speed, scale, and coverage
- Cache hit rate and estimated savings are the metrics that prove caching is working; track them from day one

---

## Further Reading

- [Efficient Inference of Large Language Models](https://arxiv.org/abs/2312.12456) — Survey of LLM inference optimization including caching at the KV-cache level
- [Semantic Caching for Large Language Model Inference](https://arxiv.org/abs/2401.08686) — Research paper on semantic cache design and threshold selection
- [GPTCache GitHub](https://github.com/zilliztech/GPTCache) — Open-source semantic caching library for LLMs
- [Redis as an LLM Cache](https://redis.io/blog/llm-caching/) — Production patterns for using Redis with LLM applications

---

## Next Lesson

**Lesson 5: A/B Testing for AI Applications** — Learn to design statistically rigorous experiments that compare prompt variants, models, and configurations without contaminating results or calling winners prematurely.
