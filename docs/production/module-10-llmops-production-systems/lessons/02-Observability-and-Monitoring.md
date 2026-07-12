---
title: Observability & Monitoring for LLM Applications
description: >-
  Learn to trace LLM calls, monitor quality and latency, set up alerts, and
  debug complex AI chains in production
duration: 40 min
difficulty: intermediate
has_code: false
module: module-10
---
# Observability & Monitoring for LLM Applications

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand why LLM observability differs from traditional monitoring | 40 min | Intermediate |
| Learn to trace multi-step LLM chains and agent loops | | |
| Set up monitoring for latency, cost, and quality | | |
| Build alerting for production AI systems | | |

---

## Why LLM Observability is Different

Traditional software monitoring tracks request/response, errors, and latency. LLM applications add unique challenges:

- **Non-deterministic outputs**: The same input can produce different outputs
- **Quality is subjective**: A 200 OK response can still be a bad answer
- **Multi-step chains**: Agents make multiple LLM calls per user request
- **Cost per request varies**: Token usage differs per interaction
- **Hallucination detection**: Errors that look like valid responses

```
Traditional API:                    LLM Application:
Request -> Response                 Request -> LLM Call 1 (reasoning)
Metrics: latency, status code              -> Tool Call (search)
                                           -> LLM Call 2 (with context)
                                           -> Tool Call (code execution)
                                           -> LLM Call 3 (synthesis)
                                           -> Response
                                    Metrics: latency, cost, quality,
                                    hallucination rate, token usage,
                                    tool success rate, per-step timing
```

---

## The Three Pillars of LLM Observability

### 1. Tracing: Follow Every Step

Tracing captures the full execution path of an LLM request, including every LLM call, tool invocation, and retrieval step.

```python
# Conceptual tracing implementation
import time
import uuid

class LLMTracer:
    def __init__(self):
        self.traces = {}
    
    def start_trace(self, user_input):
        trace_id = str(uuid.uuid4())
        self.traces[trace_id] = {
            "id": trace_id,
            "input": user_input,
            "start_time": time.time(),
            "spans": [],
            "total_tokens": 0,
            "total_cost": 0.0,
        }
        return trace_id
    
    def add_span(self, trace_id, span_type, name, input_data, output_data, 
                 tokens_used=0, cost=0.0, duration_ms=0):
        span = {
            "type": span_type,  # "llm", "tool", "retrieval"
            "name": name,
            "input": input_data,
            "output": output_data,
            "tokens": tokens_used,
            "cost": cost,
            "duration_ms": duration_ms,
            "timestamp": time.time(),
        }
        self.traces[trace_id]["spans"].append(span)
        self.traces[trace_id]["total_tokens"] += tokens_used
        self.traces[trace_id]["total_cost"] += cost
    
    def end_trace(self, trace_id, final_output):
        trace = self.traces[trace_id]
        trace["output"] = final_output
        trace["end_time"] = time.time()
        trace["total_duration_ms"] = (trace["end_time"] - trace["start_time"]) * 1000
        return trace

# Usage in your agent
tracer = LLMTracer()

def run_agent_with_tracing(user_query):
    trace_id = tracer.start_trace(user_query)
    
    # Step 1: Initial LLM reasoning
    start = time.time()
    reasoning = llm.generate(user_query)
    tracer.add_span(trace_id, "llm", "initial_reasoning",
                    input_data=user_query, output_data=reasoning,
                    tokens_used=150, cost=0.003,
                    duration_ms=(time.time() - start) * 1000)
    
    # Step 2: Tool call
    start = time.time()
    search_results = search_tool(reasoning.tool_args)
    tracer.add_span(trace_id, "tool", "web_search",
                    input_data=reasoning.tool_args, output_data=search_results,
                    duration_ms=(time.time() - start) * 1000)
    
    # Step 3: Final synthesis
    start = time.time()
    final = llm.generate(f"Context: {search_results}
Query: {user_query}")
    tracer.add_span(trace_id, "llm", "synthesis",
                    input_data=search_results, output_data=final,
                    tokens_used=300, cost=0.006,
                    duration_ms=(time.time() - start) * 1000)
    
    trace = tracer.end_trace(trace_id, final)
    print(f"Trace complete: {trace['total_duration_ms']:.0f}ms, "
          f"${trace['total_cost']:.4f}, {trace['total_tokens']} tokens")
    return final
```

### 2. Metrics: Track What Matters

```python
# Key metrics to collect for every LLM request
metrics_schema = {
    # Performance
    "latency_total_ms": "End-to-end response time",
    "latency_first_token_ms": "Time to first token (TTFT)",
    "latency_per_step_ms": "Duration of each chain step",
    
    # Cost
    "input_tokens": "Tokens sent to the model",
    "output_tokens": "Tokens generated by the model",
    "cost_usd": "Total cost of all LLM calls in this request",
    
    # Quality
    "user_feedback": "Thumbs up/down from user",
    "relevance_score": "Automated relevance scoring (0-1)",
    "hallucination_detected": "Boolean flag from fact-checking",
    
    # Reliability
    "retries": "Number of retries needed",
    "fallback_used": "Whether fallback model was activated",
    "error": "Error type if request failed",
    
    # RAG-specific
    "chunks_retrieved": "Number of context chunks fetched",
    "retrieval_relevance": "Average similarity score of retrieved chunks",
}
```

### 3. Logging: Capture for Debugging

```python
import json
import logging

class LLMLogger:
    def __init__(self):
        self.logger = logging.getLogger("llm_ops")
    
    def log_request(self, trace_id, model, messages, params):
        self.logger.info(json.dumps({
            "event": "llm_request",
            "trace_id": trace_id,
            "model": model,
            "message_count": len(messages),
            "temperature": params.get("temperature"),
            "max_tokens": params.get("max_tokens"),
            # DO NOT log full message content in production
            # (privacy, cost, storage). Log content only in debug mode.
        }))
    
    def log_response(self, trace_id, model, tokens_used, latency_ms, cost):
        self.logger.info(json.dumps({
            "event": "llm_response",
            "trace_id": trace_id,
            "model": model,
            "input_tokens": tokens_used["input"],
            "output_tokens": tokens_used["output"],
            "latency_ms": latency_ms,
            "cost_usd": cost,
        }))
```

---

## Production Monitoring Dashboard

The essential dashboard for any LLM application:

```
┌──────────────────────────────────────────────────────────┐
|  LLM Application Dashboard                               |
|                                                          |
|  Requests/min: 342    Error Rate: 0.3%   P99 Latency: 4.2s |
|  Daily Cost: $47.23   Avg Tokens/Req: 1,240              |
|                                                          |
|  ┌─────────────────┐  ┌──────────────────┐               |
|  | Latency (P50/P99)|  | Cost per Request  |             |
|  | 1.2s / 4.2s     |  | $0.014 avg       |              |
|  | [graph over time]|  | [graph over time] |             |
|  └─────────────────┘  └──────────────────┘               |
|                                                          |
|  ┌─────────────────┐  ┌──────────────────┐               |
|  | Quality Score    |  | User Feedback     |             |
|  | 0.87 avg        |  | 92% positive      |             |
|  | [graph over time]|  | [graph over time] |             |
|  └─────────────────┘  └──────────────────┘               |
|                                                          |
|  Alerts:                                                 |
|  [!] P99 latency > 5s for 3 minutes (12:34 PM)         |
|  [!] Daily cost exceeded $50 budget (2:15 PM)           |
└──────────────────────────────────────────────────────────┘
```

---

## Alerting Rules

```python
# Essential alerts for production LLM apps
alerts = {
    "latency_p99_high": {
        "condition": "p99_latency_ms > 5000 for 3 minutes",
        "severity": "warning",
        "action": "Check model provider status, consider fallback"
    },
    "error_rate_spike": {
        "condition": "error_rate > 5% for 2 minutes",
        "severity": "critical",
        "action": "Page on-call, activate fallback model"
    },
    "daily_cost_exceeded": {
        "condition": "daily_cost_usd > budget_usd",
        "severity": "warning",
        "action": "Notify team, consider rate limiting"
    },
    "quality_degradation": {
        "condition": "avg_quality_score < 0.7 for 30 minutes",
        "severity": "warning",
        "action": "Review recent outputs, check RAG pipeline"
    },
    "hallucination_rate_high": {
        "condition": "hallucination_rate > 10% for 15 minutes",
        "severity": "critical",
        "action": "Review system prompt, check context retrieval"
    }
}
```

---

## LLM Observability Tools

| Tool | Type | Key Feature |
|------|------|-------------|
| **LangSmith** | Managed platform | Deep LangChain integration, prompt playground |
| **Langfuse** | Open source / managed | Tracing, scoring, prompt management |
| **Arize Phoenix** | Open source | LLM traces, evaluations, embeddings analysis |
| **Helicone** | Managed proxy | Drop-in logging proxy, cost tracking |
| **Braintrust** | Managed platform | Evals, logging, prompt playground |

---

## Key Takeaways

- LLM observability requires tracing (multi-step), metrics (cost/quality/latency), and logging
- Always track cost per request alongside performance metrics
- Quality monitoring (user feedback, automated scoring) is as important as latency
- Set up alerts for latency, errors, cost budgets, and quality degradation
- Use specialized LLM observability tools for tracing complex chains

---

## Next Lesson

**Lesson 3: Prompt Versioning & Management** - Learn to version, test, and deploy prompt changes safely.
