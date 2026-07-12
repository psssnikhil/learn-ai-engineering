---
title: Cost Optimization for LLM Applications
description: >-
  Learn practical strategies to reduce LLM API costs by 50-90% without
  sacrificing quality, including caching, model routing, and prompt optimization
duration: 35 min
difficulty: intermediate
has_code: false
---
# Cost Optimization for LLM Applications

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand the LLM cost equation | 35 min | Intermediate |
| Implement caching strategies (exact + semantic) | | |
| Use model routing to match cost to task complexity | | |
| Optimize prompts to reduce token usage | | |

---

## The LLM Cost Equation

```
Total Cost = (Input Tokens x Input Price) + (Output Tokens x Output Price)
             x Number of Requests
             x Average Steps per Request (for agents)
```

### Where Costs Hide

| Cost Driver | Impact | Easy to Overlook? |
|------------|--------|-------------------|
| Long system prompts | Sent with EVERY request | Yes - seems small but multiplies |
| Agent loops (5-15 LLM calls per request) | 5-15x per user interaction | Yes - each step grows context |
| RAG context chunks | 1,000-5,000 tokens of retrieved text | Sometimes |
| Retry on errors | Double cost for failed requests | Yes |
| Development/testing | Developers testing prompts | Yes |

---

## Strategy 1: Caching (Biggest Impact)

### Exact Match Cache

Cache identical requests. Simple but effective for common queries.

```python
import hashlib
import json
import time

class LLMCache:
    def __init__(self, ttl_seconds=3600):
        self.cache = {}
        self.ttl = ttl_seconds
        self.stats = {"hits": 0, "misses": 0}
    
    def _make_key(self, model, messages, params):
        content = json.dumps({
            "model": model,
            "messages": messages,
            "temperature": params.get("temperature", 1.0),
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, model, messages, params):
        key = self._make_key(model, messages, params)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                self.stats["hits"] += 1
                return entry["response"]
            else:
                del self.cache[key]
        self.stats["misses"] += 1
        return None
    
    def put(self, model, messages, params, response):
        key = self._make_key(model, messages, params)
        self.cache[key] = {
            "response": response,
            "timestamp": time.time()
        }
    
    def hit_rate(self):
        total = self.stats["hits"] + self.stats["misses"]
        return self.stats["hits"] / total if total > 0 else 0

# Usage
cache = LLMCache(ttl_seconds=3600)

def cached_llm_call(model, messages, **params):
    cached = cache.get(model, messages, params)
    if cached:
        return cached  # Free!
    
    response = client.chat.completions.create(
        model=model, messages=messages, **params
    )
    cache.put(model, messages, params, response)
    return response
```

### Semantic Cache

Cache similar (not just identical) queries using embeddings.

```python
import numpy as np

class SemanticCache:
    def __init__(self, similarity_threshold=0.95):
        self.entries = []  # List of (embedding, response) pairs
        self.threshold = similarity_threshold
    
    def get(self, query_embedding):
        if not self.entries:
            return None
        
        for entry_emb, response in self.entries:
            similarity = np.dot(query_embedding, entry_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(entry_emb)
            )
            if similarity >= self.threshold:
                return response
        return None
    
    def put(self, query_embedding, response):
        self.entries.append((query_embedding, response))

# "What's the weather in NYC?" and "Weather in New York City?"
# have different text but similar embeddings -> cache hit!
```

**Typical savings**: 30-60% cost reduction on production workloads.

---

## Strategy 2: Model Routing

Use expensive models only when needed. Route simple tasks to cheaper models.

```python
def route_to_model(query, context=None):
    """Route queries to the most cost-effective model."""
    
    # Tier 1: Simple/short queries -> cheapest model
    if len(query.split()) < 20 and not requires_reasoning(query):
        return "gpt-4.1-mini"  # ~10x cheaper
    
    # Tier 2: Standard queries -> mid-tier model
    if not requires_deep_reasoning(query):
        return "gpt-4.1-mini"
    
    # Tier 3: Complex reasoning, coding, analysis -> flagship
    return "gpt-4.1"

def requires_reasoning(query):
    """Simple heuristic for routing."""
    reasoning_signals = [
        "analyze", "compare", "explain why", "debug",
        "what are the tradeoffs", "step by step",
        "write code", "review this"
    ]
    return any(signal in query.lower() for signal in reasoning_signals)

# In practice, you can also use a small classifier model
# or the LLM itself to classify query complexity
```

**Typical savings**: 40-70% by routing 60-80% of queries to smaller models.

---

## Strategy 3: Prompt Optimization

### Reduce System Prompt Length

```python
# BEFORE: 500 tokens system prompt (sent with EVERY request)
system_v1 = """You are a helpful customer support agent for Acme Corp. 
You should always be polite, professional, and empathetic. When a customer
asks about our products, provide accurate information from our knowledge
base. If you don't know the answer, say so honestly. Always suggest
relevant help articles when available. Format your responses using
markdown with clear headers. Keep responses under 200 words unless
the customer asks for more detail. Never share internal pricing or
confidential information..."""  # Goes on for 500+ tokens

# AFTER: 150 tokens (same behavior when fine-tuned or well-prompted)
system_v2 = """Acme Corp support agent. Be helpful, concise, professional.
Use knowledge base for product info. If unsure, say so and suggest help
articles. Use markdown. Max 200 words. Never share internal/confidential info."""

# Savings: 350 tokens x 1M requests/month = 350M tokens saved
# At $2.50/1M tokens = $875/month saved just from this one change
```

### Minimize Context Window Usage

```python
# For RAG: send only the most relevant chunks
def optimized_rag(query, knowledge_base, max_chunks=3, max_tokens=1000):
    chunks = knowledge_base.search(query, top_k=10)
    
    # Filter by relevance threshold
    relevant = [c for c in chunks if c.score > 0.7]
    
    # Take top N and truncate if needed
    context = ""
    for chunk in relevant[:max_chunks]:
        if len(context.split()) + len(chunk.text.split()) > max_tokens:
            break
        context += chunk.text + "

"
    
    return context  # Much less than blindly sending 10 full chunks
```

---

## Strategy 4: Batch Processing

For non-real-time workloads, batch requests for better pricing.

```python
# Many providers offer batch APIs at 50% discount
# OpenAI Batch API example:
batch_requests = [
    {"custom_id": f"req-{i}", "method": "POST", "url": "/v1/chat/completions",
     "body": {"model": "gpt-4.1-mini", "messages": msgs}}
    for i, msgs in enumerate(all_message_lists)
]

# Submit batch (results available within 24 hours)
# 50% cost reduction for non-urgent processing
```

---

## Cost Monitoring Dashboard

```python
# Track these metrics daily
cost_metrics = {
    "daily_total_usd": "sum of all API costs",
    "cost_per_request_avg": "total / request_count",
    "cost_per_user_avg": "total / active_users",
    "cache_hit_rate": "cached / total requests",
    "model_distribution": {
        "gpt-4.1": "15% of requests, 60% of cost",
        "gpt-4.1-mini": "85% of requests, 40% of cost"
    },
    "tokens_per_request_avg": "input + output tokens / requests",
    "waste_tokens": "tokens in failed/retried requests",
}

# Set budget alerts
DAILY_BUDGET = 100  # USD
if daily_total > DAILY_BUDGET * 0.8:
    alert("Approaching daily budget limit")
```

---

## Quick Wins Summary

| Strategy | Effort | Savings | When to Use |
|----------|--------|---------|-------------|
| Exact caching | Low | 20-40% | Repetitive queries |
| Semantic caching | Medium | 30-60% | Similar but varied queries |
| Model routing | Medium | 40-70% | Mixed complexity workloads |
| Shorter prompts | Low | 10-30% | Long system prompts |
| Fewer RAG chunks | Low | 10-20% | RAG applications |
| Batch processing | Medium | 50% | Non-real-time workloads |

---

## Key Takeaways

- Caching is the single biggest cost saver (exact + semantic)
- Route simple queries to cheaper models, reserve flagship for complex tasks
- Optimize system prompts and RAG context to reduce per-request tokens
- Use batch APIs for non-urgent processing at 50% discount
- Monitor cost per request and set daily budget alerts
- Aim for 50-90% cost reduction by combining strategies

---

## Next Lesson

**Lesson 7: Model Deployment Patterns** - Learn strategies for deploying LLM applications including blue-green deployments, canary releases, and graceful model migration.
