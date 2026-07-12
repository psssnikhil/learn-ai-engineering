---
title: Observability in the Harness
description: >-
  Trace agent runs with spans for each LLM call and tool invocation, structure
  logs for debugging, and integrate Langfuse or OpenTelemetry into the harness
duration: 40 min
difficulty: advanced
has_code: true
module: module-18
---
# Observability in the Harness

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Instrument the harness with traces and spans | 40 min | Advanced |
| Log each perceive-reason-act step with structured fields | | |
| Integrate Langfuse or OpenTelemetry for production visibility | | |
| Debug failed agent runs using trace replay | | |

---

## Why Harness Observability Differs from API Logs

A single user request to an agent may trigger **many** LLM calls and tool invocations. Traditional HTTP logging captures one request/response pair. You need **per-step** visibility inside the loop.

```
User: "Fix the failing test in auth.py"
  │
  ├─ span: llm.reason          (1.2s, 2,400 tokens, $0.04)
  ├─ span: tool.read_file      (12ms)
  ├─ span: llm.reason          (0.9s, 1,800 tokens, $0.03)
  ├─ span: tool.run_terminal   (4.1s)
  ├─ span: llm.reason          (1.5s, 3,100 tokens, $0.05)
  └─ span: llm.final_answer    (0.8s, 900 tokens, $0.01)
```

[M10 Lesson 2](../../../production/module-10-llmops-production-systems/lessons/02-Observability-and-Monitoring.md) covers LLMOps observability broadly. This lesson focuses on **instrumenting the harness itself**.

---

## Traces and Spans

A **trace** is one agent run. **Spans** are timed units of work inside it.

| Span type | What it captures |
|-----------|------------------|
| `harness.run` | Full run from goal to finish |
| `harness.perceive` | Context assembly, trimming |
| `llm.completion` | Model call — tokens, cost, latency |
| `tool.execute` | Tool name, args (redacted), result size |
| `harness.checkpoint` | Persist latency |
| `harness.policy` | Budget check, approval wait |

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

### Instrumented harness step

```python
def run_step(self, state: AgentState, trace_id: str) -> AgentState:
    with self.tracer.span(trace_id, "harness.perceive", "harness",
                          step=state.step) as span:
        context = self.perceive(state)
        span.attributes["message_count"] = len(context)

    with self.tracer.span(trace_id, "llm.completion", "llm",
                          model="gpt-4.1", step=state.step) as span:
        response = self.model.complete(context, tools=self.tools.schemas)
        span.attributes["input_tokens"] = response.usage.input_tokens
        span.attributes["output_tokens"] = response.usage.output_tokens
        span.attributes["cost_usd"] = response.usage.cost_usd

    for call in response.tool_calls or []:
        with self.tracer.span(trace_id, f"tool.{call.name}", "tool",
                              step=state.step) as span:
            span.attributes["arguments"] = self._redact(call.arguments)
            result = self.tools.execute(call.name, call.arguments)
            span.attributes["result_chars"] = len(result)
            span.attributes["success"] = '"ok": true' in result

    return state
```

---

## Structured Logging

Logs complement traces — searchable, aggregatable, alert-friendly.

```python
import logging
import json

logger = logging.getLogger("agent.harness")

def log_step(event: str, run_id: str, step: int, **fields):
    payload = {
        "event": event,
        "run_id": run_id,
        "step": step,
        **fields,
    }
    logger.info(json.dumps(payload))

# Usage inside the loop
log_step("llm_complete", state.run_id, state.step,
         tokens=response.usage.total_tokens,
         cost_usd=response.usage.cost_usd,
         duration_ms=turn["duration_ms"])

log_step("tool_execute", state.run_id, state.step,
         tool=call.name,
         success=success,
         duration_ms=result.duration_ms)
```

| Event | Fields to capture |
|-------|-------------------|
| `run_start` | goal, policy, model |
| `llm_complete` | tokens, cost, latency |
| `tool_execute` | tool name, success, duration |
| `approval_requested` | tool, args summary |
| `budget_exceeded` | limit type, partial progress |
| `run_end` | status, total steps, total cost |

!!! tip "Log fields, not paragraphs"
    Structured JSON logs feed dashboards and alerts. Reserve free-text for `error_message` only.

---

## Langfuse Integration

[Langfuse](https://langfuse.com/) is an open-source LLM observability platform — well suited for agent traces. It models **traces → observations** (spans/generations).

```python
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context

langfuse = Langfuse()

@observe()
def llm_completion(messages, tools):
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        tools=tools,
    )
    langfuse_context.update_current_observation(
        input=messages,
        output=response.choices[0].message,
        model="gpt-4.1",
        usage={
            "input": response.usage.prompt_tokens,
            "output": response.usage.completion_tokens,
        },
    )
    return response

@observe()
def run_agent(goal: str):
    langfuse_context.update_current_trace(name="agent-run", input=goal)
    state = AgentState(goal=goal, messages=[...])
    harness = AgentHarness(...)
    state = harness.run(state)
    langfuse_context.update_current_trace(output=state.messages[-1])
    return state
```

Langfuse gives you:

- **Trace UI** — waterfall view of LLM + tool steps
- **Cost dashboards** — per-run and per-user spend
- **Scores** — attach eval results to traces (pairs with [M19](../../../production/module-19-llm-evaluation-quality/index.md))
- **Prompt versioning** — compare runs across prompt changes

---

## OpenTelemetry Integration

For teams already on OTel, export harness spans to your existing stack (Jaeger, Grafana Tempo, Datadog):

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

provider = TracerProvider()
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("agent.harness")

def run_with_otel(state, trace_id):
    with tracer.start_as_current_span("harness.run") as run_span:
        run_span.set_attribute("run_id", state.run_id)
        run_span.set_attribute("goal", state.goal[:200])

        with tracer.start_as_current_span("llm.completion") as llm_span:
            response = model.complete(state.messages)
            llm_span.set_attribute("tokens.total", response.usage.total_tokens)
            llm_span.set_attribute("cost.usd", response.usage.cost_usd)

        for call in response.tool_calls or []:
            with tracer.start_as_current_span(f"tool.{call.name}") as tool_span:
                tool_span.set_attribute("tool.name", call.name)
                result = tools.execute(call.name, call.arguments)
                tool_span.set_attribute("result.length", len(result))
```

| Tool | Best for |
|------|----------|
| **Langfuse** | LLM-native UI, prompt management, eval scores |
| **OpenTelemetry** | Unified infra observability, existing APM |
| **Both** | OTel for infra spans, Langfuse for LLM-specific metadata |

---

## Debugging with Trace Replay

When a user reports "the agent did the wrong thing," traces answer:

1. **What did the model see?** → `perceive` span attributes / logged message count
2. **Which tools fired?** → tool spans in order
3. **Did policy block anything?** → `policy` spans, `approval_requested` logs
4. **Did a budget stop the run?** → `budget_exceeded` event

Combine with [Lesson 2 checkpoints](02-agent-loop-and-state.md) for full replay:

```python
def debug_run(run_id: str, step: int, tracer: HarnessTracer, store: CheckpointStore):
    state = store.load(run_id, step)
    trace = tracer.traces.get(state.run_id, [])
    print(f"=== Run {run_id} @ step {step} ===")
    for span in trace:
        indent = "  " if span.parent_id else ""
        dur = f"{span.duration_ms:.0f}ms" if span.duration_ms else "..."
        print(f"{indent}{span.name} [{span.span_type}] {dur} {span.attributes}")
    print(f"Messages: {len(state.messages)}, Tools: {len(state.tool_results)}")
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

## Key Takeaways

- Agent observability requires **per-step traces** — one HTTP log is not enough
- Model **traces** (runs) and **spans** (perceive, LLM, tool, checkpoint) map directly to harness phases
- Use **structured JSON logs** with consistent event names for search and alerts
- **Langfuse** is LLM-native; **OpenTelemetry** fits existing APM — use either or both
- Pair traces with **checkpoints** for replay when debugging production failures

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
