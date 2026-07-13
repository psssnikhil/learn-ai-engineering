---
title: Cost Optimization for LLM Applications
description: >-
  Learn practical strategies to reduce LLM API costs by 50-90% without
  sacrificing quality, including caching, model routing, and prompt optimization
duration: 45 min
difficulty: intermediate
has_code: true
module: module-10
---
# Cost Optimization for LLM Applications

## Prerequisites

- Completed Lessons 1–5 (LLMOps fundamentals through A/B Testing)
- Understanding of tokenization (input tokens vs output tokens)
- Familiarity with the caching concepts from Lesson 4

---

## What You'll Learn

| Objective | Outcome |
|-----------|---------|
| Decompose the LLM cost equation to find the largest savings | Can identify the highest-impact cost drivers in any LLM app |
| Implement multi-level caching for maximum hit rate | Can combine exact-match and semantic caching appropriately |
| Design model routing logic to match cost to task complexity | Can reduce cost by 40–70% without degrading quality |
| Compress prompts without losing behavioral fidelity | Can reduce system prompt tokens by 30–60% |
| Use batch APIs for non-real-time workloads | Can access 50% cost discounts on appropriate jobs |

---

## Intuition First: Where Does Your Money Actually Go?

Before optimizing, you need to understand the cost equation. LLM costs are not flat per request—they are proportional to the tokens you send and receive.

```
Total Cost = (Input Tokens × Input Price/1M) + (Output Tokens × Output Price/1M)
           × Number of Requests
           × Average LLM Calls Per Request (for agentic systems)
```

For GPT-4o pricing (approximately $2.50/M input, $10.00/M output):

| Component | Tokens per Request | Cost per Request | Monthly (100k requests) |
|-----------|-------------------|------------------|------------------------|
| System prompt | 300 | $0.00075 | $75 |
| User message | 50 | $0.00013 | $13 |
| RAG context | 1,500 | $0.00375 | $375 |
| Output tokens | 300 | $0.00300 | $300 |
| **Total** | **2,150** | **$0.00763** | **$763/month** |

Now scale this: an agent that makes 5 LLM calls per user request costs 5× as much. A system prompt that runs 800 tokens instead of 300 adds $38/month at 100k requests — just from making it more verbose.

The cost hierarchy from highest to lowest impact:
1. **Number of agent steps** — each step is a full LLM call
2. **RAG context size** — often 60–70% of total input tokens
3. **Output token length** — expensive and slow
4. **System prompt length** — multiplied by every request
5. **Model tier selection** — 10–100× price difference between tiers

---

## Strategy 1: Caching (Highest ROI, Lowest Effort)

We covered caching in Lesson 4. This is the single highest-ROI optimization. Implement it before any other cost reduction:

```python
import hashlib
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

class CostTrackingCache:
    """
    Exact-match cache that tracks how much money it saves.
    Use this to build the business case for continued investment in caching.
    """

    def __init__(self, ttl_seconds: int = 3600):
        self._cache: dict[str, dict] = {}
        self._ttl = timedelta(seconds=ttl_seconds)
        self._hits = 0
        self._misses = 0
        self._total_saved_usd = 0.0

    def _make_key(self, model: str, messages: list, params: dict) -> str:
        content = json.dumps(
            {"model": model, "messages": messages, **params}, sort_keys=True
        )
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, model: str, messages: list, params: dict) -> Optional[str]:
        key = self._make_key(model, messages, params)
        if key in self._cache:
            entry = self._cache[key]
            if datetime.now() - entry["timestamp"] < self._ttl:
                self._hits += 1
                self._total_saved_usd += entry["original_cost_usd"]
                return entry["response"]
            del self._cache[key]
        self._misses += 1
        return None

    def put(self, model: str, messages: list, params: dict,
            response: str, cost_usd: float):
        key = self._make_key(model, messages, params)
        self._cache[key] = {
            "response": response,
            "timestamp": datetime.now(),
            "original_cost_usd": cost_usd,
        }

    @property
    def report(self) -> dict:
        total = self._hits + self._misses
        return {
            "hit_rate": f"{self._hits / total:.1%}" if total else "N/A",
            "total_saved_usd": f"${self._total_saved_usd:.2f}",
            "cache_entries": len(self._cache),
        }
```

**Expected savings**: 30–60% cost reduction on most production workloads. Support bots and FAQ systems typically see 50–70%.

---

## Strategy 2: Model Routing

The most powerful long-term cost strategy: match the cost of the model to the complexity of the task.

```
GPT-4o:      ~$0.007/request (complex reasoning, analysis, code review)
GPT-4o-mini: ~$0.0007/request (FAQ, classification, summarization)
Ratio: 10× price difference
```

If 70% of your requests are simple queries that gpt-4o-mini handles equally well, routing them to the cheaper model saves ~65% of total cost.

### A Production-Ready Model Router

```python
from enum import Enum
from dataclasses import dataclass
from typing import Callable

class ModelTier(str, Enum):
    NANO = "nano"        # Fast, cheapest: classification, simple Q&A
    STANDARD = "standard"  # Balanced: most customer queries, summarization
    PREMIUM = "premium"    # Complex: reasoning, analysis, code review, edge cases

MODEL_MAP: dict[ModelTier, str] = {
    ModelTier.NANO: "gpt-4o-mini",
    ModelTier.STANDARD: "gpt-4o-mini",
    ModelTier.PREMIUM: "gpt-4o",
}

COST_PER_1K_INPUT = {
    "gpt-4o-mini": 0.00015,
    "gpt-4o": 0.0025,
}

@dataclass
class RoutingRule:
    name: str
    tier: ModelTier
    check: Callable[[str, dict], bool]
    description: str

ROUTING_RULES: list[RoutingRule] = [
    RoutingRule(
        name="simple_faq",
        tier=ModelTier.NANO,
        check=lambda query, ctx: (
            len(query.split()) < 15
            and not any(kw in query.lower() for kw in
                       ["why", "compare", "analyze", "explain", "debug", "code"])
        ),
        description="Short queries without analytical keywords",
    ),
    RoutingRule(
        name="premium_reasoning",
        tier=ModelTier.PREMIUM,
        check=lambda query, ctx: (
            any(kw in query.lower() for kw in
                ["analyze", "compare", "debug", "step by step", "why did",
                 "write code", "review this", "tradeoffs", "edge cases"])
            or len(query.split()) > 150
            or ctx.get("requires_long_output", False)
        ),
        description="Complex reasoning, code, or long outputs",
    ),
]

def route_to_model(query: str, context: dict | None = None) -> tuple[str, ModelTier]:
    """
    Returns (model_name, tier) by applying routing rules in priority order.
    Defaults to STANDARD if no rule matches.
    """
    ctx = context or {}

    for rule in ROUTING_RULES:
        if rule.check(query, ctx):
            model = MODEL_MAP[rule.tier]
            return model, rule.tier

    return MODEL_MAP[ModelTier.STANDARD], ModelTier.STANDARD


class CostAwareLLMClient:
    """LLM client that automatically routes to the most cost-effective model."""

    def __init__(self, openai_client, cache: CostTrackingCache):
        self.client = openai_client
        self.cache = cache
        self._routing_stats: dict[str, int] = {tier.value: 0 for tier in ModelTier}

    def complete(self, query: str, system_prompt: str,
                 context: dict | None = None) -> dict:
        model, tier = route_to_model(query, context)
        self._routing_stats[tier.value] += 1

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        # Check cache first
        cached = self.cache.get(model, messages, {"temperature": 0.0})
        if cached:
            return {"response": cached, "model": model, "tier": tier, "cached": True}

        response = self.client.chat.completions.create(
            model=model, messages=messages, temperature=0.0
        )
        text = response.choices[0].message.content
        cost = (response.usage.prompt_tokens / 1e6 * COST_PER_1K_INPUT.get(model, 0.001) * 1000 +
                response.usage.completion_tokens / 1e6 * 0.60)

        self.cache.put(model, messages, {"temperature": 0.0}, text, cost)
        return {"response": text, "model": model, "tier": tier, "cached": False, "cost_usd": cost}

    def routing_report(self) -> dict:
        total = sum(self._routing_stats.values())
        if total == 0:
            return {}
        return {
            tier: {"count": count, "pct": f"{count / total:.0%}"}
            for tier, count in self._routing_stats.items()
        }
```

**Expected savings**: 40–70% by routing 60–80% of requests to cheaper models.

---

## Strategy 3: Prompt Compression

System prompts are paid for on every single request. A 500-token system prompt sent to a 10-token FAQ costs more in system prompt overhead than in actual generation.

### Principles of Prompt Compression

1. **Remove redundancy** — "You are a helpful, friendly, professional assistant. Be polite and kind to users." → "Be helpful, professional, and polite."
2. **Use imperative form** — "When the user asks a question, you should respond by..." → "Respond with..."
3. **Use structured shorthand** — Lists over paragraphs; bullet constraints over prose rules
4. **Move rarely needed instructions to few-shot examples** — Instead of explaining edge cases, show them

```python
# BEFORE: 520 tokens (~2,080 characters)
system_v1 = """You are a customer support agent for AcmeCorp, a leading provider
of cloud software solutions. Your role is to help customers with any questions
they might have about our products and services. You should always be polite,
professional, empathetic, and helpful. When you don't know the answer to a
question, it's important to acknowledge that honestly rather than making up
information, because accuracy is very important to us and our customers.

When customers ask about refunds or returns, you should refer them to our
refund policy which states that we offer a 30-day money-back guarantee on
all annual subscriptions. Monthly subscriptions can be cancelled at any time
but are not eligible for pro-rated refunds.

When customers ask about billing, always verify their understanding before
giving detailed account information.

Please format your responses using clear markdown formatting with headers
when appropriate, and keep responses under 200 words unless the customer
explicitly requests more detail. Always maintain a positive, solution-focused
tone even when customers are frustrated."""

# AFTER: 160 tokens (~640 characters) — 69% reduction
system_v2 = """AcmeCorp support agent. Rules:
- Polite, professional, empathetic. Max 200 words (unless asked for more).
- If uncertain: say so — never fabricate.
- Refunds: 30-day guarantee on annual plans; monthly plans non-refundable.
- Billing: confirm customer understanding before sharing account details.
- Format: use markdown; headers for multi-part answers.
- Tone: solution-focused, even when customer is frustrated."""

# Savings at 100k requests/month (GPT-4o at $2.50/1M input tokens):
# (520 - 160) tokens × 100,000 requests / 1,000,000 × $2.50 = $90/month
# From one system prompt change.
```

!!! warning "Test After Compression"
    Prompt compression can change model behavior in subtle ways. Always run your eval suite on the compressed prompt before deploying. The pass rate must stay above your quality gate threshold.

### Automated Token Budget Tracking

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

def analyze_prompt_costs(system_prompt: str, avg_user_message_tokens: int,
                          avg_rag_context_tokens: int, avg_output_tokens: int,
                          monthly_requests: int, model: str = "gpt-4o-mini") -> dict:
    PRICING = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
    }
    rates = PRICING.get(model, PRICING["gpt-4o-mini"])

    system_tokens = count_tokens(system_prompt, model)
    total_input = system_tokens + avg_user_message_tokens + avg_rag_context_tokens
    total_cost_per_request = (
        total_input / 1e6 * rates["input"] +
        avg_output_tokens / 1e6 * rates["output"]
    )

    return {
        "system_prompt_tokens": system_tokens,
        "system_prompt_pct_of_input": f"{system_tokens / total_input:.0%}",
        "cost_per_request_usd": round(total_cost_per_request, 5),
        "monthly_cost_usd": round(total_cost_per_request * monthly_requests, 2),
        "system_prompt_monthly_cost_usd": round(
            system_tokens / 1e6 * rates["input"] * monthly_requests, 2
        ),
    }
```

---

## Strategy 4: Minimize RAG Context

RAG context often accounts for 60–70% of total input tokens. Over-retrieval is expensive.

```python
def optimized_rag_context(
    query: str,
    knowledge_base,
    max_chunks: int = 3,
    min_relevance_score: float = 0.75,
    max_context_tokens: int = 1_500,
    model: str = "gpt-4o-mini",
) -> tuple[str, dict]:
    """
    Retrieve only the most relevant chunks within a token budget.
    Returns (context_string, retrieval_stats).
    """
    # Retrieve more than we need, then filter
    candidates = knowledge_base.search(query, top_k=10)

    # Filter by relevance threshold
    relevant = [c for c in candidates if c.score >= min_relevance_score]

    if not relevant:
        return "", {"chunks_used": 0, "tokens": 0, "reason": "no relevant chunks"}

    # Fill context within token budget
    context_parts = []
    total_tokens = 0

    for chunk in relevant[:max_chunks]:
        chunk_tokens = count_tokens(chunk.text, model)
        if total_tokens + chunk_tokens > max_context_tokens:
            break
        context_parts.append(f"[Source: {chunk.source}]\n{chunk.text}")
        total_tokens += chunk_tokens

    context = "\n\n".join(context_parts)
    return context, {
        "chunks_available": len(relevant),
        "chunks_used": len(context_parts),
        "tokens": total_tokens,
        "avg_relevance": sum(c.score for c in relevant[:len(context_parts)]) / len(context_parts),
    }
```

**Expected savings**: 20–40% reduction in input tokens for RAG applications that previously sent all retrieved chunks regardless of relevance.

---

## Strategy 5: Batch Processing

For non-real-time workloads, provider batch APIs offer 50% cost discounts with 24-hour SLAs. This is ideal for nightly processing, report generation, document indexing, and training data generation.

```python
import json
from pathlib import Path

def create_openai_batch(
    prompts: list[dict[str, str]],  # Each dict has "id" and "user_message"
    system_prompt: str,
    model: str = "gpt-4o-mini",
    output_file: str = "batch_requests.jsonl",
) -> str:
    """
    Create an OpenAI Batch API request file.
    Submit with: client.batches.create(input_file_id=..., endpoint="/v1/chat/completions")
    Results available within 24 hours at 50% cost reduction.
    """
    requests = []
    for item in prompts:
        requests.append({
            "custom_id": item["id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": item["user_message"]},
                ],
                "temperature": 0,
                "max_tokens": 500,
            },
        })

    with open(output_file, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    print(f"Created {len(requests)} batch requests in {output_file}")
    print(f"Estimated cost: ${len(requests) * 0.003 * 0.5:.2f} (50% batch discount)")
    return output_file
```

**When to use batch**: Any processing that doesn't need real-time response—customer feedback analysis, document classification, generating summaries for a product catalog, nightly report generation.

---

## Putting It Together: A Cost Optimization Roadmap

Apply these strategies in order of effort vs impact:

| Priority | Strategy | Effort | Typical Savings | When |
|----------|----------|--------|-----------------|------|
| 1 | Exact-match caching | Low (1 day) | 20–40% | Immediately |
| 2 | Prompt compression | Low (hours) | 10–30% | After testing |
| 3 | RAG context trimming | Low (hours) | 20–40% | RAG apps only |
| 4 | Model routing | Medium (1 week) | 40–70% | After baseline established |
| 5 | Semantic caching | Medium (2 days) | 30–60% additional | After exact-match |
| 6 | Batch processing | Medium (2 days) | 50% on eligible workloads | Non-real-time only |

Combined, these strategies routinely achieve 70–90% cost reduction on production workloads.

### Cost Monitoring Dashboard

```python
def daily_cost_report(requests_log: list[dict]) -> dict:
    """
    Generate a daily cost report from request logs.
    Log schema: {model, input_tokens, output_tokens, cached, tier, cost_usd, timestamp}
    """
    MODELS = set(r["model"] for r in requests_log)

    total_cost = sum(r["cost_usd"] for r in requests_log if not r.get("cached"))
    cache_hits = sum(1 for r in requests_log if r.get("cached"))
    saved_by_cache = sum(r.get("cost_usd", 0) for r in requests_log if r.get("cached"))

    by_model = {}
    for model in MODELS:
        reqs = [r for r in requests_log if r["model"] == model]
        by_model[model] = {
            "requests": len(reqs),
            "pct": f"{len(reqs) / len(requests_log):.0%}",
            "cost_usd": round(sum(r["cost_usd"] for r in reqs if not r.get("cached")), 2),
        }

    return {
        "total_requests": len(requests_log),
        "total_cost_usd": round(total_cost, 2),
        "avg_cost_per_request": round(total_cost / len(requests_log), 5) if requests_log else 0,
        "cache_hit_rate": f"{cache_hits / len(requests_log):.1%}" if requests_log else "N/A",
        "estimated_savings_from_cache_usd": round(saved_by_cache, 2),
        "by_model": by_model,
    }
```

Set alerts at 80% of your daily budget. When an alert fires, the cost report immediately shows which model or endpoint is the culprit.

---

## Edge Cases and Misconceptions

**"The cheapest model per token is always the cheapest option."**
Not necessarily. A cheaper model that requires more tokens of context to achieve the same quality (e.g., more few-shot examples, longer instructions) may cost more overall. Measure actual cost-per-good-response, not cost-per-token.

**"Caching makes RAG answers stale."**
For FAQ-style RAG (stable knowledge base), short cache TTLs (4–8 hours) keep content fresh while capturing the bulk of repeated queries. For rapidly changing data, cache only the retrieval step (which chunks to use) not the generation step, since chunks can be invalidated when source documents change.

**"Batch processing always saves money."**
Batch APIs save money only on eligible workloads. If your users expect real-time responses, batch APIs aren't applicable. Use batch for background processing, not interactive applications.

**"I should optimize cost before launching."**
Launch first. Measure actual cost under real traffic. Then optimize the highest-cost hotspots. Premature cost optimization often targets the wrong bottleneck.

---

## Key Takeaways

- The LLM cost equation is: `(input tokens × input rate + output tokens × output rate) × requests × agent steps`; agent steps are the highest-leverage multiplier to control
- Caching is the highest ROI optimization: implement exact-match first, then semantic caching for paraphrase coverage
- Model routing saves 40–70% by sending simple queries to 10× cheaper models and reserving premium models for complex tasks
- System prompt compression reduces the per-request baseline cost; even a 200-token reduction at 1M requests/month saves $30–50 depending on the model
- RAG context trimming (filtering by relevance score + token budget) eliminates the biggest single source of input token waste
- Use batch APIs for non-real-time workloads to access 50% discounts; never use them for interactive applications

---

## Further Reading

- [Efficient Large Language Models: A Survey](https://arxiv.org/abs/2312.03863) — Comprehensive survey of LLM efficiency techniques including inference optimization
- [Reducing LLM Costs in Production](https://arxiv.org/abs/2401.02954) — Practical techniques validated across production deployments
- [LLM Router: Content-Based Routing for Language Model Inference](https://arxiv.org/abs/2402.01869) — Academic treatment of intelligent model routing
- [OpenAI Batch API documentation](https://platform.openai.com/docs/guides/batch) — Official guide to batch processing at 50% discount

---

## Next Lesson

**Lesson 7: Model Deployment Patterns** — Learn API gateway patterns, fallback chains, blue-green deployments, and canary releases that keep your LLM application reliable even when providers have outages.
