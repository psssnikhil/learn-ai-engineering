---
title: Observability in the Harness
description: >-
  Trace agent runs with spans for each LLM call and tool invocation, structure
  logs for debugging, and integrate Langfuse or OpenTelemetry into the harness
duration: 55 min
difficulty: advanced
has_code: true
module: module-18
---
# Observability in the Harness

## Prerequisites

- [Lesson 2 — Agent Loop and State](02-agent-loop-and-state.md): the perceive-reason-act loop and checkpoints
- [Lesson 3 — Tools and Function Calling](03-tools-and-function-calling.md): tool execution and ToolResult
- Basic familiarity with structured logging (JSON logs) and distributed tracing concepts
- Python `logging` module and context managers

---

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand why agent observability requires per-step traces, not request logs | 55 min | Advanced |
| Instrument the harness with traces (runs) and spans (perceive, LLM, tool, checkpoint) | | |
| Write structured logs with consistent event names for search and alerting | | |
| Integrate Langfuse for LLM-native traces with cost and eval attribution | | |
| Export spans to OpenTelemetry for existing APM stacks | | |
| Debug failed agent runs using trace replay combined with checkpoints | | |

---

## Intuition First

A web server logs one request/response pair per HTTP call. You can answer: did this request succeed? How long did it take?

An agent run is not one request. It is 5, 10, or 20 LLM API calls plus tool executions, interleaved over minutes. Traditional HTTP logging captures the outer shell but misses everything inside.

Without per-step observability, when a user reports "the agent gave a wrong answer," you can only see: query came in, answer went out. You can't answer: *Which tool call returned wrong data? Did a policy block a needed action? Did the agent loop before giving up? Was the context window trimmed mid-run?*

Per-step observability answers all of these. A **trace** captures the full agent run. **Spans** capture each timed unit of work inside it: perceive (context assembly), reason (LLM call), act (tool execution), checkpoint (state persistence), and policy check.

This lesson shows you how to build that instrumentation from scratch, then how to connect it to production-grade platforms.

---

## Why Harness Observability Differs from API Logs

A single user request to an agent may trigger **many** LLM calls and tool invocations. Traditional HTTP logging captures one request/response pair. You need **per-step** visibility inside the loop.

```
User: "Fix the failing test in auth.py"
  │
  ├─ span: harness.perceive      (5ms, assembled 4,200 token context)
  ├─ span: llm.completion        (1.2s, 2,400 input + 180 output tokens, $0.006)
  ├─ span: tool.read_file        (12ms, read 3,400 chars)
  ├─ span: harness.perceive      (4ms, assembled 6,100 token context)
  ├─ span: llm.completion        (0.9s, 3,100 input + 220 output tokens, $0.008)
  ├─ span: tool.run_terminal     (4.1s, pytest returned exit code 1)
  ├─ span: harness.perceive      (6ms)
  ├─ span: llm.completion        (1.5s, 4,800 input + 300 output tokens, $0.013)
  └─ span: llm.final_answer      (0.8s, 4,900 input + 120 output tokens, $0.013)
  
Total: 8.5s, 12 spans, $0.040
```

[M10 Lesson 2](../../../production/module-10-llmops-production-systems/lessons/02-Observability-and-Monitoring.md) covers LLMOps observability broadly. This lesson focuses on **instrumenting the harness itself**.

---

## Traces and Spans

A **trace** is one agent run from goal to finish. **Spans** are timed units of work inside it.

| Span type | What it captures |
|-----------|------------------|
| `harness.run` | Full run — run_id, goal, final status, total cost |
| `harness.perceive` | Context assembly time, message count, trimming events |
| `llm.completion` | Model, input tokens, output tokens, cost, latency |
| `tool.execute` | Tool name, redacted args, result size, success flag |
| `harness.checkpoint` | Save/load latency, path |
| `harness.policy` | Budget check result, approval wait time |

```python
import uuid
import time
from dataclasses import dataclass, field
from contextlib import contextmanager

@dataclass
class Span:
    span_id: str
    trace_id: str
    parent_id: str | None
    name: str
    span_type: str
    start_ms: float
    end_ms: float | None = None
    attributes: dict = field(default_factory=dict)
    status: str = "ok"

    @property
    def duration_ms(self) -> float | None:
        if self.end_ms is None:
            return None
        return self.end_ms - self.start_ms

class HarnessTracer:
    def __init__(self):
        self.traces: dict[str, list[Span]] = {}

    def start_trace(self, run_id: str, goal: str) -> str:
        trace_id = str(uuid.uuid4())
        self.traces[trace_id] = []
        root = Span(
            span_id=str(uuid.uuid4()),
            trace_id=trace_id,
            parent_id=None,
            name="harness.run",
            span_type="harness",
            start_ms=time.time() * 1000,
            attributes={"run_id": run_id, "goal": goal[:200]},
        )
        self.traces[trace_id].append(root)
        return trace_id

    @contextmanager
    def span(self, trace_id: str, name: str, span_type: str, **attrs):
        parent = self.traces[trace_id][-1] if self.traces[trace_id] else None
        s = Span(
            span_id=str(uuid.uuid4()),
            trace_id=trace_id,
            parent_id=parent.span_id if parent else None,
            name=name,
            span_type=span_type,
            start_ms=time.time() * 1000,
            attributes=attrs,
        )
        self.traces[trace_id].append(s)
        try:
            yield s
            s.status = "ok"
        except Exception as e:
            s.status = "error"
            s.attributes["error"] = str(e)
            raise
        finally:
            s.end_ms = time.time() * 1000
```

### Instrumented Harness Step

```python
def run_step(self, state: AgentState, trace_id: str) -> AgentState:
    with self.tracer.span(trace_id, "harness.perceive", "harness",
                          step=state.step) as span:
        context = self.perceive(state)
        span.attributes["message_count"] = len(context)
        span.attributes["estimated_tokens"] = sum(
            len(m.get("content", "") or "") // 4 for m in context
        )

    with self.tracer.span(trace_id, "llm.completion", "llm",
                          model="gpt-4.1", step=state.step) as span:
        response = self.model.complete(context, tools=self.tools.schemas)
        span.attributes["input_tokens"] = response.usage.input_tokens
        span.attributes["output_tokens"] = response.usage.output_tokens
        span.attributes["cost_usd"] = response.usage.cost_usd
        span.attributes["has_tool_calls"] = bool(response.tool_calls)

    for call in response.tool_calls or []:
        with self.tracer.span(trace_id, f"tool.{call.name}", "tool",
                              tool_name=call.name, step=state.step) as span:
            # Redact sensitive argument values
            span.attributes["argument_keys"] = list(
                json.loads(call.arguments).keys()
                if isinstance(call.arguments, str)
                else call.arguments.keys()
            )
            result = self.tools.execute(call.name, call.arguments)
            span.attributes["result_chars"] = len(result)
            span.attributes["success"] = '"ok": true' in result

    return state
```

---

## Structured Logging

Logs complement traces — searchable in log aggregators, alertable, and lightweight to emit.

```python
import logging
import json
from datetime import datetime, timezone

logger = logging.getLogger("agent.harness")

def log_step(event: str, run_id: str, step: int, **fields):
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "run_id": run_id,
        "step": step,
        **fields,
    }
    logger.info(json.dumps(payload))

# Standard event names — consistency enables aggregation across runs
log_step("run_start", state.run_id, 0,
         goal_length=len(state.goal), model="gpt-4.1", policy_max_steps=20)

log_step("llm_complete", state.run_id, state.step,
         input_tokens=response.usage.input_tokens,
         output_tokens=response.usage.output_tokens,
         cost_usd=response.usage.cost_usd,
         duration_ms=round(duration_ms))

log_step("tool_execute", state.run_id, state.step,
         tool=call.name,
         success=success,
         duration_ms=round(result.duration_ms))

log_step("budget_check", state.run_id, state.step,
         ok=ok, reason=reason,
         steps_used=state.step, cost_so_far=state.total_cost_usd)

log_step("run_end", state.run_id, state.step,
         status=state.status, total_steps=state.step,
         total_cost_usd=state.total_cost_usd,
         total_tokens=state.total_tokens)
```

| Event | Fields to capture |
|-------|-------------------|
| `run_start` | goal length, model, policy settings |
| `llm_complete` | input tokens, output tokens, cost, latency |
| `tool_execute` | tool name, success flag, duration |
| `approval_requested` | tool name, redacted argument keys |
| `budget_exceeded` | which limit, partial progress summary |
| `run_end` | final status, total steps, total cost, total tokens |

!!! tip "Log fields, not paragraphs"
    Structured JSON logs feed dashboards and alerts. Reserve free-text for `error_message` fields only. Unstructured prose is hard to query and impossible to aggregate.

---

## Langfuse Integration

[Langfuse](https://langfuse.com/) is an open-source LLM observability platform with a trace/observation model that maps naturally onto harness spans.

```python
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

langfuse = Langfuse()

@observe()
def llm_completion(messages: list[dict], tools: list[dict]) -> dict:
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        tools=tools,
    )
    langfuse_context.update_current_observation(
        name="llm.completion",
        input=messages,
        output=response.choices[0].message.model_dump(),
        model="gpt-4.1",
        usage={
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
            "unit": "TOKENS",
        },
    )
    return response

@observe()
def tool_execution(tool_name: str, arguments: dict, registry) -> str:
    result = registry.execute(tool_name, arguments)
    langfuse_context.update_current_observation(
        name=f"tool.{tool_name}",
        input={"tool": tool_name, "args": list(arguments.keys())},
        output={"result_length": len(result), "ok": '"ok": true' in result},
    )
    return result

@observe()
def run_agent(goal: str) -> AgentState:
    langfuse_context.update_current_trace(
        name="agent-run",
        input={"goal": goal[:200]},
        tags=["production"],
    )

    state = AgentState(goal=goal, messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": goal},
    ])

    while state.status == "running" and state.step < MAX_STEPS:
        response = llm_completion(state.messages, registry.schemas)
        if not response.tool_calls:
            state.status = "finished"
        else:
            for call in response.tool_calls:
                result = tool_execution(call.name, json.loads(call.arguments), registry)
                state.messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        state.step += 1

    langfuse_context.update_current_trace(
        output={"status": state.status, "steps": state.step, "cost_usd": state.total_cost_usd},
    )
    return state
```

**What Langfuse gives you out of the box:**

- Trace waterfall UI — visualize each step's latency and cost
- Cost dashboards — per-run, per-user, per-model spending
- Score attachment — attach faithfulness/relevance scores to traces for eval
- Prompt versioning — compare traces across prompt iterations
- Dataset creation — export traces as eval datasets

---

## OpenTelemetry Integration

For teams already on an APM stack (Datadog, Grafana Tempo, Jaeger), export harness spans via OTel:

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(
    endpoint="http://otel-collector:4317",
)))
trace.set_tracer_provider(provider)
otel_tracer = trace.get_tracer("agent.harness")

def run_with_otel(state: AgentState):
    with otel_tracer.start_as_current_span("harness.run") as run_span:
        run_span.set_attribute("run_id", state.run_id)
        run_span.set_attribute("goal", state.goal[:200])

        while state.status == "running":
            with otel_tracer.start_as_current_span("harness.perceive") as perceive_span:
                context = perceive(state)
                perceive_span.set_attribute("message_count", len(context))

            with otel_tracer.start_as_current_span("llm.completion") as llm_span:
                response = model.complete(context)
                llm_span.set_attribute("model", "gpt-4.1")
                llm_span.set_attribute("tokens.input", response.usage.input_tokens)
                llm_span.set_attribute("tokens.output", response.usage.output_tokens)
                llm_span.set_attribute("cost.usd", response.usage.cost_usd)

            for call in response.tool_calls or []:
                with otel_tracer.start_as_current_span(f"tool.{call.name}") as tool_span:
                    tool_span.set_attribute("tool.name", call.name)
                    result = tools.execute(call.name, call.arguments)
                    tool_span.set_attribute("result.length", len(result))
                    tool_span.set_attribute("result.ok", '"ok": true' in result)

        run_span.set_attribute("final.status", state.status)
        run_span.set_attribute("total.cost_usd", state.total_cost_usd)
        run_span.set_attribute("total.tokens", state.total_tokens)
```

| Tool | Best for |
|------|----------|
| **Langfuse** | LLM-native UI, cost dashboards, prompt management, eval scores |
| **OpenTelemetry** | Unified infra observability, existing APM, service mesh tracing |
| **Both** | OTel for infrastructure spans; Langfuse for LLM-specific metadata and evals |

---

## Debugging with Trace Replay

When a user reports "the agent did the wrong thing," traces + checkpoints answer:

**Step 1 — Find the trace:**
```python
# Query Langfuse or search structured logs for the run_id
traces = langfuse.fetch_traces(tags=["production"], limit=100)
problem_trace = next(t for t in traces.data if t.input["goal"] == user_reported_goal)
```

**Step 2 — Walk the spans:**
```python
def debug_run(run_id: str, tracer: HarnessTracer, store: CheckpointStore):
    state = store.load(run_id, step=0)
    trace = tracer.traces.get(state.run_id, [])

    print(f"=== Run {run_id} — {state.goal[:80]} ===")
    for span in trace:
        indent = "  " * (1 if span.parent_id else 0)
        dur = f"{span.duration_ms:.0f}ms" if span.duration_ms else "running"
        attrs = {k: v for k, v in span.attributes.items() if k not in {"goal"}}
        print(f"{indent}[{span.status}] {span.name} ({dur}) {attrs}")

    print(f"\nFinal: status={state.status}, steps={state.step}, cost=${state.total_cost_usd:.4f}")
```

**Step 3 — Diagnose the failure:**

The traces answer four questions:

1. **What did the model see?** → `perceive` span: `message_count`, `estimated_tokens`
2. **Which tools fired and succeeded?** → `tool.*` spans: `tool_name`, `success`, `result_chars`
3. **Did policy block a needed action?** → `harness.policy` spans, `budget_check` logs
4. **Did a budget stop the run?** → `budget_exceeded` log event, `run_end` `status` field

```python
# Common diagnostic pattern: tool called but returned error
for span in trace:
    if span.span_type == "tool" and span.attributes.get("success") is False:
        print(f"FAILED TOOL: {span.name} at step {span.attributes['step']}")
        print(f"  Duration: {span.duration_ms:.0f}ms")
        print(f"  Result size: {span.attributes.get('result_chars', 'unknown')} chars")
        # Load checkpoint to see the actual tool result
        step_state = store.load(run_id, span.attributes["step"])
        failed_result = next(
            (r for r in step_state.tool_results if not r.success),
            None,
        )
        if failed_result:
            print(f"  Error: {failed_result.output[:200]}")
```

---

## SLOs and Alerting

Define Service Level Objectives for your agent and alert when they're at risk:

| SLO | Target | Alert threshold |
|-----|--------|-----------------|
| P99 run latency | < 30s | > 25s (5-min window) |
| Tool error rate | < 5% | > 10% (5-min window) |
| Run cost P99 | < $0.10 | > $0.08 (rolling daily) |
| Steps P95 | < policy max × 0.8 | > policy max × 0.9 |
| Approval rejection rate | Baseline | Spike vs 7-day avg > 2σ |
| LLM latency P95 | < 10s | > 8s |

```python
# Prometheus-style metrics (use prometheus_client in production)
from dataclasses import dataclass
from collections import deque
import time

class MetricsCollector:
    def __init__(self):
        self.run_latencies: deque = deque(maxlen=1000)
        self.tool_results: deque = deque(maxlen=1000)
        self.run_costs: deque = deque(maxlen=1000)

    def record_run(self, latency_ms: float, cost_usd: float, status: str):
        ts = time.time()
        self.run_latencies.append((ts, latency_ms))
        self.run_costs.append((ts, cost_usd))

    def record_tool(self, tool_name: str, success: bool, duration_ms: float):
        self.tool_results.append((time.time(), tool_name, success, duration_ms))

    def p99_latency_ms(self, window_seconds: int = 300) -> float:
        cutoff = time.time() - window_seconds
        recent = sorted(lat for ts, lat in self.run_latencies if ts > cutoff)
        if not recent:
            return 0.0
        idx = int(len(recent) * 0.99)
        return recent[min(idx, len(recent) - 1)]

    def tool_error_rate(self, window_seconds: int = 300) -> float:
        cutoff = time.time() - window_seconds
        recent = [(ok, dur) for ts, name, ok, dur in self.tool_results if ts > cutoff]
        if not recent:
            return 0.0
        errors = sum(1 for ok, _ in recent if not ok)
        return errors / len(recent)
```

---

## Metrics Worth Alerting On

| Metric | Alert threshold (starting point) |
|--------|----------------------------------|
| `tool.error_rate` | > 20% over 5 min |
| `run.cost_p99` | > 2× baseline |
| `run.steps_p99` | > policy max × 0.9 |
| `approval.rejection_rate` | spike vs 7-day avg |
| `llm.latency_p95` | > 10s |

---

## Common Misconceptions

**"One log per request is enough."** For agents, one log captures the shell — not the interior. Without per-step spans you cannot distinguish "model made a bad decision at step 3" from "tool returned corrupted data at step 5."

**"Tracing adds too much latency."** Adding spans with a context manager adds ~0.1ms per span. For 20 spans per run, that's 2ms — negligible against the 10–60s agent run time.

**"I can reconstruct traces from logs after the fact."** Structured logs can approximate traces if they carry consistent `run_id` and `step` fields. But you lose parent-child relationships, timing precision, and the ability to visualize a waterfall. Investing in traces from day one is significantly cheaper than retrofitting.

**"Langfuse is only for evals."** Langfuse handles traces, cost attribution, prompt versioning, and dataset creation — all of which are useful independent of eval pipelines.

**"I only need observability in production."** Development traces are even more valuable — they let you iterate on the agent without building a debugging mental model from scratch each session.

---

## Production Tips

- **Emit traces asynchronously.** Use a background queue to send spans to Langfuse/OTel so trace emission doesn't add to user-facing latency.
- **Set trace sampling for high-volume agents.** At 100 queries/second, tracing 100% may be cost-prohibitive. Sample 10–20% and always trace errors and slow runs (duration > P95 threshold).
- **Attribute costs to users/tasks.** Pass `user_id` and `task_type` as trace attributes. This enables per-feature, per-user cost dashboards — essential for cost allocation in multi-tenant products.
- **Review traces weekly.** A 30-minute weekly trace review catches patterns invisible in aggregated metrics: specific tools that fail repeatedly, models that loop on certain question types, approval prompts that always get rejected.
- **Redact PII before storing.** Never store raw user queries in traces without scrubbing. Store query hashes for correlation and query lengths for context size debugging.

---

## Key Takeaways

- Agent observability requires **per-step traces** — one HTTP log is not enough to understand a multi-step agent run
- Model **traces** (runs) and **spans** (perceive, LLM, tool, checkpoint, policy) map directly to harness phases
- Use **structured JSON logs** with consistent event names (`run_start`, `llm_complete`, `tool_execute`) for search and alerting
- **Langfuse** is LLM-native: cost dashboards, prompt versioning, eval score attachment; **OpenTelemetry** fits existing APM
- Pair traces with **checkpoints** (Lesson 2) to enable full replay when debugging production failures
- Define **SLOs** for latency, tool error rate, and cost — alert before they breach, not after

---

## Related Papers

| Paper | Year | Key contribution |
|-------|------|-----------------|
| [Dspy: Compiling Declarative Language Model Calls into State-of-the-Art Pipelines](https://arxiv.org/abs/2310.03714) | 2023 | Programmatic compilation of LLM pipelines — includes execution tracing |
| [LangSmith: Evaluate and Monitor LLM Applications](https://docs.smith.langchain.com/) | 2023 | Production tracing and eval platform from LangChain |
| [Benchmarking LLM Inference Backends](https://arxiv.org/abs/2407.12873) | 2024 | Latency profiling methodology for LLM systems — relevant to SLO setting |
| [Observability for Large Language Models](https://arxiv.org/abs/2408.06918) | 2024 | Survey of observability challenges and approaches specific to LLM deployments |

---

## Further Reading

- [M10 · Observability & Monitoring](../../../production/module-10-llmops-production-systems/lessons/02-Observability-and-Monitoring.md) — LLMOps monitoring foundations
- [Langfuse docs](https://langfuse.com/docs) — tracing SDK and self-hosting
- [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/) — OTLP exporters
- [Awesome Harness Engineering](https://github.com/ai-boost/awesome-harness-engineering) — observability tooling list
- [Agents Towards Production](https://github.com/NirDiamant/agents-towards-production) — production monitoring patterns

---

## Next Steps

You have completed **M18 · Agent Harness, Tools & Runtime**. Continue to:

- [M12 · Multi-Agent Systems](../../module-12-multi-agent-systems/index.md) — orchestrate multiple harnessed agents
- [M19 · LLM Evaluation & Quality](../../../production/module-19-llm-evaluation-quality/index.md) — score agent traces automatically
- [M10 · LLMOps](../../../production/module-10-llmops-production-systems/index.md) — deploy harnessed agents to production
