---
title: Observability & Monitoring for LLM Applications
description: >-
  Learn to trace LLM calls, monitor quality and latency, set up alerts, and
  debug complex AI chains in production
duration: 45 min
difficulty: intermediate
has_code: true
module: module-10
---
# Observability & Monitoring for LLM Applications

## Prerequisites

- Completed Lesson 1 (Introduction to LLMOps)
- Familiarity with Python logging and basic HTTP concepts
- Optional: exposure to distributed tracing concepts (OpenTelemetry, Jaeger)

---

## What You'll Learn

| Objective | Outcome |
|-----------|---------|
| Explain why LLM observability is harder than traditional monitoring | Can articulate the multi-step, non-deterministic challenge |
| Implement span-level tracing for LLM chains | Can instrument any LLM application with traces |
| Design a metrics schema covering latency, cost, and quality | Can spec out a production monitoring dashboard |
| Write alerting rules for LLM-specific failure modes | Can configure alerts that fire before users notice problems |
| Choose the right observability tool for your stack | Can evaluate LangSmith, Langfuse, and Arize for your use case |

---

## Intuition First: The Blind Spot Problem

Traditional APIs are opaque boxes with one input and one output. If the box returns a 200, you're done. LLM applications are transparent pipelines: a user query triggers a retrieval step, which feeds an LLM, which calls a tool, which triggers another LLM call, which produces a response. Any step can silently degrade quality without raising an error code.

```
Traditional API (simple):
  User → Request → Response
  Monitor: latency, status code. Done.

LLM Agent (complex):
  User → Query Rewrite (LLM 1)
              ↓
         Vector Search (3 chunks retrieved)
              ↓
         Context Assembly (1,800 tokens)
              ↓
         LLM 2 (reasoning + tool selection)
              ↓
         Tool Call: search_web(query)
              ↓
         LLM 3 (synthesis, 400 output tokens)
              ↓
         Output Validation
              ↓
         User Response

  Monitor: total latency + per-step latency, per-step cost,
           retrieval quality, tool success rate, output quality,
           token usage per step, hallucination flags.
```

A 200 OK from the final step tells you nothing about whether the retrieval returned the right documents, whether the reasoning step chose the correct tool, or whether the synthesis hallucinated a fact. Observability means capturing the full picture.

---

## The Three Pillars of LLM Observability

### Pillar 1: Tracing — The Full Execution Path

A **trace** is a record of the entire request lifecycle from user input to final response. A trace contains multiple **spans**, each representing one step (one LLM call, one tool invocation, one retrieval operation).

```python
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, Any

@dataclass
class Span:
    span_id: str
    trace_id: str
    span_type: str          # "llm", "tool", "retrieval", "embedding"
    name: str
    start_time: float
    end_time: float = 0.0
    input_data: Any = None
    output_data: Any = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000


@dataclass
class Trace:
    trace_id: str
    user_input: str
    start_time: float
    spans: list[Span] = field(default_factory=list)
    end_time: float = 0.0
    final_output: Optional[str] = None

    @property
    def total_duration_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    @property
    def total_tokens(self) -> int:
        return sum(s.input_tokens + s.output_tokens for s in self.spans)

    @property
    def total_cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.spans)

    def llm_spans(self) -> list[Span]:
        return [s for s in self.spans if s.span_type == "llm"]

    def failed_spans(self) -> list[Span]:
        return [s for s in self.spans if s.error is not None]


class Tracer:
    """Lightweight tracer for LLM applications."""

    def __init__(self):
        self.active_traces: dict[str, Trace] = {}

    def start_trace(self, user_input: str) -> str:
        trace_id = str(uuid.uuid4())
        self.active_traces[trace_id] = Trace(
            trace_id=trace_id,
            user_input=user_input,
            start_time=time.time(),
        )
        return trace_id

    def start_span(self, trace_id: str, span_type: str, name: str, input_data: Any = None) -> str:
        span_id = str(uuid.uuid4())
        span = Span(
            span_id=span_id,
            trace_id=trace_id,
            span_type=span_type,
            name=name,
            start_time=time.time(),
            input_data=input_data,
        )
        self.active_traces[trace_id].spans.append(span)
        return span_id

    def end_span(self, trace_id: str, span_id: str, output_data: Any = None,
                 input_tokens: int = 0, output_tokens: int = 0,
                 cost_usd: float = 0.0, error: Optional[str] = None):
        trace = self.active_traces[trace_id]
        span = next(s for s in trace.spans if s.span_id == span_id)
        span.end_time = time.time()
        span.output_data = output_data
        span.input_tokens = input_tokens
        span.output_tokens = output_tokens
        span.cost_usd = cost_usd
        span.error = error

    def end_trace(self, trace_id: str, final_output: str) -> Trace:
        trace = self.active_traces[trace_id]
        trace.end_time = time.time()
        trace.final_output = final_output
        return trace
```

### Using the Tracer in a Real Agent

```python
tracer = Tracer()

def run_rag_agent(user_query: str) -> str:
    trace_id = tracer.start_trace(user_query)

    # Step 1: Embed the query
    span_id = tracer.start_span(trace_id, "embedding", "query_embedding", user_query)
    embedding = embed(user_query)   # ~5ms, ~150 tokens
    tracer.end_span(trace_id, span_id, output_data="[vector]",
                    input_tokens=len(user_query.split()), cost_usd=0.00002)

    # Step 2: Retrieve context
    span_id = tracer.start_span(trace_id, "retrieval", "vector_search", embedding)
    chunks = vector_db.search(embedding, top_k=5)
    tracer.end_span(trace_id, span_id, output_data=f"{len(chunks)} chunks retrieved")

    # Step 3: Generate response
    context = "\n".join(c.text for c in chunks)
    messages = [
        {"role": "system", "content": "Answer based only on the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_query}"},
    ]
    span_id = tracer.start_span(trace_id, "llm", "generation", messages)
    response = llm_client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    output_text = response.choices[0].message.content
    usage = response.usage
    tracer.end_span(
        trace_id, span_id,
        output_data=output_text,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        cost_usd=usage.prompt_tokens / 1e6 * 0.15 + usage.completion_tokens / 1e6 * 0.60,
    )

    trace = tracer.end_trace(trace_id, output_text)
    print(f"[TRACE] {trace.total_duration_ms:.0f}ms | "
          f"{trace.total_tokens} tokens | ${trace.total_cost_usd:.5f}")
    return output_text
```

This gives you per-step timing, per-step cost, and a complete record of every input and output—invaluable when debugging why a response was wrong.

### Pillar 2: Metrics — Track What Matters

Metrics are aggregated measurements over time. Unlike traces (per-request), metrics tell you trends: "latency has been creeping up for three days" or "quality scores dropped 8% after last Tuesday's prompt update."

```python
# The complete metrics schema for an LLM application
LLM_METRICS_SCHEMA = {
    # ── Performance ──────────────────────────────────────────
    "latency_total_ms":          "End-to-end wall-clock time",
    "latency_ttft_ms":           "Time to first token (streaming)",
    "latency_retrieval_ms":      "Vector search + reranking",
    "latency_llm_ms":            "Pure LLM inference time",
    "latency_tool_ms":           "Tool execution time",

    # ── Cost ─────────────────────────────────────────────────
    "input_tokens":              "Tokens in the prompt (billed at input rate)",
    "output_tokens":             "Tokens generated (billed at output rate)",
    "cost_usd":                  "Total cost for this request in USD",
    "system_prompt_tokens":      "Baseline overhead per request",
    "context_tokens":            "RAG chunks or conversation history",

    # ── Quality ───────────────────────────────────────────────
    "user_feedback":             "+1 / -1 / null (explicit thumbs up/down)",
    "relevance_score":           "0-1 automated relevance (LLM-judged)",
    "faithfulness_score":        "0-1 groundedness to retrieved context",
    "hallucination_detected":    "Boolean flag from fact-checking step",
    "refusal_rate":              "% of requests the model declined to answer",

    # ── Reliability ───────────────────────────────────────────
    "retries":                   "Number of retries needed for this request",
    "fallback_used":             "True if secondary model was activated",
    "error_type":                "null | rate_limit | timeout | content_filter | ...",
    "cache_hit":                 "True if response served from cache",

    # ── RAG-specific ──────────────────────────────────────────
    "chunks_retrieved":          "Number of context chunks fetched",
    "avg_chunk_similarity":      "Mean cosine similarity of retrieved chunks",
    "citations_verified":        "% of cited sources actually in retrieved context",
}
```

### Pillar 3: Logging — Capture for Debugging

Logs are the raw event stream. They differ from metrics in that logs contain the actual content of requests and responses—useful for debugging individual failures, but expensive to store at scale.

```python
import json
import logging
from enum import Enum

class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"

class LLMLogger:
    """
    Structured logger for LLM applications.
    Logs metadata by default; logs content only in debug mode.
    Never log full content in production — privacy + cost + storage.
    """

    def __init__(self, debug_mode: bool = False):
        self.logger = logging.getLogger("llmops")
        self.debug_mode = debug_mode

    def log_request(self, trace_id: str, model: str, num_messages: int,
                    temperature: float, estimated_tokens: int):
        entry = {
            "event": "llm_request",
            "trace_id": trace_id,
            "model": model,
            "num_messages": num_messages,
            "temperature": temperature,
            "estimated_tokens": estimated_tokens,
        }
        self.logger.info(json.dumps(entry))

    def log_response(self, trace_id: str, model: str, input_tokens: int,
                     output_tokens: int, latency_ms: float, cost_usd: float,
                     error: str | None = None):
        entry = {
            "event": "llm_response",
            "trace_id": trace_id,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": round(latency_ms, 1),
            "cost_usd": round(cost_usd, 6),
            "error": error,
        }
        level = "error" if error else "info"
        getattr(self.logger, level)(json.dumps(entry))

    def log_quality(self, trace_id: str, relevance: float,
                    faithfulness: float, user_feedback: int | None):
        entry = {
            "event": "quality_score",
            "trace_id": trace_id,
            "relevance": round(relevance, 3),
            "faithfulness": round(faithfulness, 3),
            "user_feedback": user_feedback,
        }
        self.logger.info(json.dumps(entry))
```

!!! note "What to Log vs What to Metric"
    Log individual events with their trace ID. Metric the aggregates (averages, percentiles, counts). When an alert fires on a metric, you use the trace ID from the log to find the exact request that caused the problem.

---

## Production Dashboard Design

Every LLM application needs one primary dashboard with four panels. This is not a "nice to have"—it is the instrument panel for your application.

```
┌──────────────────────────────────────────────────────────────────────┐
│  LLM Application Dashboard — Last 24h                               │
│                                                                      │
│  Requests: 28,431    Error Rate: 0.4%    P99 Latency: 4.8s          │
│  Daily Cost: $73.20   Cache Hit Rate: 34%   Avg Quality: 0.86       │
│                                                                      │
│  ┌───────────────────────┐   ┌───────────────────────┐              │
│  │  Latency (P50 / P99)  │   │  Cost per Request ($)  │             │
│  │  1.3s / 4.8s          │   │  0.0026 avg            │             │
│  │  ▁▂▃▂▁▁▂▃▄▃▂▁▁▂▃      │   │  ▁▁▁▂▁▁▁▁▂▁▁▁▁▁▁      │             │
│  └───────────────────────┘   └───────────────────────┘              │
│                                                                      │
│  ┌───────────────────────┐   ┌───────────────────────┐              │
│  │  Quality Score (avg)  │   │  User Feedback          │            │
│  │  0.86 (target: 0.85)  │   │  91% positive           │            │
│  │  ▅▆▅▆▆▅▄▃▂▃▄▅▆▆▅      │   │  ▇▇▇▇▆▆▆▆▇▆▆▇▇▇▇       │            │
│  └───────────────────────┘   └───────────────────────┘              │
│                                                                      │
│  ⚠ ALERT: P99 latency exceeded 5s for 8 minutes at 14:32            │
│  ✓ RESOLVED: Cost budget at 73% for the day                         │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Alerting Rules That Matter

Alerts must be **actionable** and **severity-appropriate**. Noisy alerts train teams to ignore them.

```python
ALERT_DEFINITIONS = {
    # ── Reliability alerts (page-worthy) ───────────────────────────────
    "error_rate_critical": {
        "condition": "error_rate_pct > 5 for 2 consecutive minutes",
        "severity": "critical",
        "channel": "pagerduty",
        "runbook": "1. Check provider status page. 2. Activate fallback model. 3. Notify on-call.",
    },
    "all_models_down": {
        "condition": "error_rate_pct > 50 for 1 minute",
        "severity": "critical",
        "channel": "pagerduty",
        "runbook": "Activate maintenance mode. Return static fallback responses.",
    },

    # ── Quality alerts (Slack + investigation) ──────────────────────────
    "quality_degradation_warning": {
        "condition": "avg_quality_score < 0.75 for 20 consecutive minutes",
        "severity": "warning",
        "channel": "slack:#ai-alerts",
        "runbook": "1. Sample 10 recent traces. 2. Check if prompt or model changed. 3. Inspect retrieval quality.",
    },
    "hallucination_rate_high": {
        "condition": "hallucination_rate_pct > 8 for 15 minutes",
        "severity": "critical",
        "channel": "pagerduty",
        "runbook": "1. Increase faithfulness guardrails in output validator. 2. Consider reverting last prompt change.",
    },

    # ── Performance alerts ───────────────────────────────────────────────
    "latency_p99_high": {
        "condition": "latency_p99_ms > 6000 for 5 minutes",
        "severity": "warning",
        "channel": "slack:#ai-alerts",
        "runbook": "1. Check model provider status. 2. Check vector DB latency. 3. Consider enabling aggressive caching.",
    },

    # ── Cost alerts ──────────────────────────────────────────────────────
    "daily_budget_80pct": {
        "condition": "daily_cost_usd > daily_budget_usd * 0.80",
        "severity": "warning",
        "channel": "slack:#ai-costs",
        "runbook": "Review token usage by endpoint. Consider tightening max_tokens or increasing cache TTL.",
    },
    "per_request_cost_spike": {
        "condition": "p95_cost_per_request_usd > 0.10 for 10 minutes",
        "severity": "warning",
        "channel": "slack:#ai-alerts",
        "runbook": "Likely cause: agent loop consuming excessive steps or context window bloat.",
    },
}
```

---

## Worked Example: Debugging with Traces

Suppose your quality score alert fires at 2 AM. Here's how traces make debugging tractable:

1. **Alert fires**: `avg_quality_score < 0.75 for 25 minutes`
2. **Open dashboard**: Quality score dropped from 0.88 to 0.71 starting at 01:47 AM
3. **Filter traces by time window**: Pull all traces from 01:45–02:15 AM
4. **Spot pattern**: 68% of failed traces have `faithfulness_score < 0.5` — model is ignoring the retrieved context
5. **Inspect a specific trace**:
   - Retrieval span: returned 5 chunks, avg similarity 0.91 ✓
   - Context assembly: 2,800 tokens of context ✓
   - LLM span: output is 420 tokens, but faithfulness is 0.31 ✗
6. **Root cause**: The chunks are fine, but the system prompt was updated at 01:44 AM. The new prompt removed the instruction "only use the provided context"
7. **Fix**: Revert the prompt change, redeploy

Without per-span tracing, you would know quality dropped but have no way to isolate whether the failure was in retrieval, context assembly, or LLM generation.

---

## Observability Tools Comparison

| Tool | Type | Strengths | Best For |
|------|------|-----------|----------|
| **LangSmith** | Managed (LangChain) | Deep LangChain/LangGraph integration, prompt playground | LangChain apps, collaborative teams |
| **Langfuse** | Open source / managed | Full tracing, scoring, prompt management, self-hostable | Privacy-conscious teams, custom stacks |
| **Arize Phoenix** | Open source | Embeddings analysis, RAG evaluation, UMAP visualization | Deep RAG debugging, embedding drift |
| **Helicone** | Managed proxy | Drop-in OpenAI proxy, automatic cost tracking | Minimal instrumentation, OpenAI-only stacks |
| **Braintrust** | Managed platform | Eval + logging in one, experiment tracking | Teams running offline + online evals |

**Recommendation**: Start with Langfuse (self-hostable, free tier available) or LangSmith (best-in-class for LangChain). Add Arize if you have embedding or RAG-specific debugging needs.

---

## Production Scenario: Diagnosing a Latency Spike

At 2:17 AM on a Tuesday, your p99 latency alert fires: response times have spiked from 3.2 seconds to 14.8 seconds. Here's how observability data drives a 12-minute diagnosis and resolution.

### Step 1: Dashboard Triage (2 minutes)

Your monitoring dashboard shows:
- p99 latency: 14.8s (threshold: 8s) → **ALERTING**
- p50 latency: 6.4s (was 1.8s) → severely elevated
- Error rate: 0.3% (normal)
- LLM-specific latency: 12.1s (was 2.6s)
- Retrieval latency: 0.4s (normal)
- Post-processing latency: 0.3s (normal)

This immediately narrows the problem to the LLM call—not retrieval, not post-processing.

### Step 2: Metric Drill-Down (3 minutes)

```python
# Query: LLM latency broken down by model and region
SELECT
    model,
    region,
    avg(llm_latency_ms) as avg_latency,
    percentile(llm_latency_ms, 99) as p99_latency,
    count(*) as requests
FROM llm_spans
WHERE timestamp > NOW() - INTERVAL '30 minutes'
GROUP BY model, region
ORDER BY p99_latency DESC;

# Output:
# gpt-4o | us-east-1 | 12,340ms | 18,200ms | 1,247
# gpt-4o-mini | us-east-1 | 1,840ms | 3,200ms | 892
# gpt-4o | eu-west-1 | 2,100ms | 4,100ms | 234
```

The problem is isolated to gpt-4o in us-east-1. GPT-4o-mini and all EU traffic are normal.

### Step 3: Provider Status Check (2 minutes)

OpenAI status page shows: "Investigating increased latency for GPT-4o API in North America."

### Step 4: Remediation (5 minutes)

```python
# Activate fallback: route to gpt-4o-mini while provider incident is ongoing
FALLBACK_ACTIVE = True

# Update routing in feature flag system (no code deployment needed)
feature_flags.set("model_fallback_active", True, region="us-east-1")

# Alert fires 5 minutes later: p99 back to 3.8s
```

**Total time from alert to resolution**: 12 minutes. Without structured observability, this investigation would have taken 40–90 minutes of log grepping.

### The Key Lesson

The trace data answered three questions in sequence:
1. *Which pipeline stage is slow?* → LLM call, not retrieval
2. *Which model/region is affected?* → gpt-4o, us-east-1 only
3. *What's the remediation?* → Route to fallback model

Without per-span metrics broken down by model and region, you would have seen "latency is high" and spent an hour trying to understand why.

---

## Edge Cases and Misconceptions

**"We log the full prompt—that's enough."**
Full prompt logging is a good start but insufficient. You need structured span-level data: per-step latency, per-step token usage, per-step cost. A full log of a 30-second agent trace tells you what happened; a structured trace tells you *where* 25 of those 30 seconds were spent.

**"Quality monitoring is too expensive to run continuously."**
You do not need to judge every request. Sample 5–10% of production traffic for quality scoring (randomized or stratified by user segment). For a system serving 10,000 requests per day, that is 500–1,000 quality evaluations—a few dollars at most.

**"A 200 status code means the response was good."**
An LLM that confidently states a wrong fact returns HTTP 200. Status codes tell you about transport-level success, not semantic quality. Every LLM application needs quality-layer monitoring on top of standard HTTP monitoring.

---

## Key Takeaways

- LLM observability requires three layers working together: traces (per-request execution path), metrics (aggregated trends), and logs (raw event stream)
- Each trace should capture every span—LLM call, retrieval, tool use—with its own timing, token count, and cost
- Monitor four pillars simultaneously: quality, performance, cost, and reliability; they interact and can mask each other
- Alerts must be actionable with clear runbooks—noisy alerts without context get ignored
- Trace IDs link metrics to logs: when an alert fires on an aggregated metric, trace IDs let you find the exact failing request
- Sample 5–10% of production traffic for quality scoring; increase sampling on negative feedback and high-risk intent categories

---

## Further Reading

- [Logging and observability for AI systems](https://arxiv.org/abs/2401.10957) — Research on observability gaps in production AI
- [Challenges in Deploying Large Language Models](https://arxiv.org/abs/2311.02428) — Survey of production failure modes across real deployments
- [LangSmith Documentation](https://docs.smith.langchain.com/) — Detailed tracing and evaluation patterns
- [OpenTelemetry for LLMs](https://opentelemetry.io/blog/2024/llm-observability/) — How standard distributed tracing applies to LLM systems

---

## Next Lesson

**Lesson 3: Prompt Versioning & Management** — Learn to version prompts as first-class code artifacts, build a file-based registry, implement parameterized templates, and roll back safely when a prompt change breaks production.
