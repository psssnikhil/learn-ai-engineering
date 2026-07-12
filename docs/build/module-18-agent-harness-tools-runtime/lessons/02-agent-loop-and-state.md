---
title: Agent Loop and State
description: >-
  Implement the perceive-reason-act cycle in a harness, manage conversation and
  working state, and design checkpoints for resume and audit
duration: 45 min
difficulty: advanced
has_code: true
module: module-18
---
# Agent Loop and State

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Implement perceive-reason-act inside a harness | 45 min | Advanced |
| Design state schemas for conversation and working memory | | |
| Build checkpoint and restore for long-running agents | | |
| Handle context growth without losing critical information | | |

---

## Perceive → Reason → Act in the Harness

[M11 Lesson 3](../../module-11-ai-agents-fundamentals/lessons/03-ReAct-Pattern.md) introduced ReAct: the model *reasons* in text and *acts* via tools. The harness maps that pattern onto three concrete phases each loop iteration:

```
  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │   PERCEIVE   │────▶│    REASON    │────▶│     ACT      │
  │ Build context│     │  LLM call    │     │ Run tools    │
  │ from state   │     │  + parse     │     │ in sandbox   │
  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘
         │                    │                    │
         │                    │                    │
         └────────────────────┴────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  UPDATE STATE    │
                    │  + checkpoint    │
                    └──────────────────┘
```

| Phase | Harness job | Model job |
|-------|-------------|-----------|
| **Perceive** | Select messages, inject memories, trim context | — |
| **Reason** | Call API, handle retries, record span | Decide next action or final answer |
| **Act** | Validate, execute tools, capture results | Emit structured tool calls |

The harness is active in every phase. The model only participates in **Reason**.

---

## A Structured Agent State

Flat message lists work for demos. Production harnesses use typed state:

```python
from dataclasses import dataclass, field
from typing import Any, Literal
import time
import uuid

@dataclass
class ToolResult:
    tool_call_id: str
    name: str
    arguments: dict
    output: str
    duration_ms: float
    success: bool

@dataclass
class AgentState:
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    goal: str = ""
    messages: list[dict] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    step: int = 0
    status: Literal["running", "paused", "finished", "failed"] = "running"
    scratchpad: dict[str, Any] = field(default_factory=dict)
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    def append_assistant(self, content: str | None, tool_calls: list | None = None):
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.messages.append(msg)

    def append_tool_result(self, result: ToolResult):
        self.tool_results.append(result)
        self.messages.append({
            "role": "tool",
            "tool_call_id": result.tool_call_id,
            "content": result.output,
        })
```

**Why separate `tool_results` from `messages`?** Messages are the model's view (possibly trimmed). `tool_results` is the harness audit log — full payloads, timings, and success flags for observability and replay.

---

## The Harness Loop

```python
class AgentHarness:
    def __init__(self, model, tools, policy, checkpoint_dir: str | None = None):
        self.model = model
        self.tools = tools
        self.policy = policy
        self.checkpoint_dir = checkpoint_dir

    def perceive(self, state: AgentState) -> list[dict]:
        """Build the context window the model will see."""
        messages = list(state.messages)

        # Inject working memory into system context
        if state.scratchpad.get("plan"):
            plan_note = f"\nCurrent plan: {state.scratchpad['plan']}"
            messages[0] = {
                **messages[0],
                "content": messages[0]["content"] + plan_note,
            }

        # Trim if approaching context limits
        return self._trim_messages(messages, max_tokens=100_000)

    def reason(self, state: AgentState) -> dict:
        """Call the model with retry logic."""
        context = self.perceive(state)
        start = time.perf_counter()
        response = self.model.complete(context, tools=self.tools.schemas)
        duration_ms = (time.perf_counter() - start) * 1000
        return {"response": response, "duration_ms": duration_ms}

    def act(self, state: AgentState, tool_calls: list) -> list[ToolResult]:
        """Execute tool calls inside the sandbox."""
        results = []
        for call in tool_calls:
            start = time.perf_counter()
            try:
                output = self.tools.execute(call.name, call.arguments)
                success = not output.startswith("Error:")
            except Exception as e:
                output = f"Error: {e}"
                success = False
            duration_ms = (time.perf_counter() - start) * 1000
            results.append(ToolResult(
                tool_call_id=call.id,
                name=call.name,
                arguments=call.arguments,
                output=output,
                duration_ms=duration_ms,
                success=success,
            ))
        return results

    def run(self, goal: str, system_prompt: str) -> AgentState:
        state = AgentState(
            goal=goal,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": goal},
            ],
        )

        while state.status == "running":
            if state.step >= self.policy.max_steps:
                state.status = "finished"
                break

            # REASON
            turn = self.reason(state)
            response = turn["response"]
            state.total_tokens += response.usage.total_tokens
            state.total_cost_usd += response.usage.cost_usd

            if response.tool_calls:
                state.append_assistant(response.text, response.tool_calls)
                # ACT
                results = self.act(state, response.tool_calls)
                for r in results:
                    state.append_tool_result(r)
            else:
                state.append_assistant(response.text)
                state.status = "finished"

            state.step += 1
            self._maybe_checkpoint(state)

        return state
```

!!! tip "Perceive is underrated"
    Most debugging happens in **perceive**, not in the model. If the agent "forgets" something, check whether your harness trimmed it, failed to inject a tool result, or used the wrong message format for your provider.

---

## Checkpoints: Resume, Audit, Replay

Checkpoints serialize harness state so you can resume after crashes, pause for human approval, or replay a run for debugging.

```python
import json
from pathlib import Path
from datetime import datetime, timezone

class CheckpointStore:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, state: AgentState) -> Path:
        path = self.base_dir / f"{state.run_id}_step{state.step}.json"
        payload = {
            "run_id": state.run_id,
            "goal": state.goal,
            "step": state.step,
            "status": state.status,
            "messages": state.messages,
            "tool_results": [
                {
                    "tool_call_id": r.tool_call_id,
                    "name": r.name,
                    "arguments": r.arguments,
                    "output": r.output,
                    "duration_ms": r.duration_ms,
                    "success": r.success,
                }
                for r in state.tool_results
            ],
            "scratchpad": state.scratchpad,
            "total_tokens": state.total_tokens,
            "total_cost_usd": state.total_cost_usd,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(payload, indent=2))
        return path

    def load(self, run_id: str, step: int) -> AgentState:
        path = self.base_dir / f"{run_id}_step{step}.json"
        payload = json.loads(path.read_text())
        state = AgentState(
            run_id=payload["run_id"],
            goal=payload["goal"],
            messages=payload["messages"],
            step=payload["step"],
            status=payload["status"],
            scratchpad=payload.get("scratchpad", {}),
            total_tokens=payload.get("total_tokens", 0),
            total_cost_usd=payload.get("total_cost_usd", 0.0),
        )
        for r in payload.get("tool_results", []):
            state.tool_results.append(ToolResult(**r))
        return state
```

### When to checkpoint

| Strategy | Trade-off |
|----------|-----------|
| **Every step** | Maximum recoverability; more I/O |
| **After tool calls** | Good for human-in-the-loop pause points |
| **On error** | Capture failure context before retry |
| **Periodic (every N steps)** | Balance for long research tasks |

```python
def _maybe_checkpoint(self, state: AgentState):
    if not self.checkpoint_dir:
        return
    if state.step % 1 == 0:  # every step — tune for your workload
        store = CheckpointStore(self.checkpoint_dir)
        store.save(state)
```

LangGraph and similar frameworks provide built-in checkpoint backends (SQLite, Postgres). Whether you use a framework or not, the **semantics** are the same: immutable snapshots keyed by `run_id` + `step`.

---

## Context Growth and Compaction

Each loop iteration appends to `messages`. Token usage grows roughly **linearly with step count** for the prompt side — a common source of cost blowups (see [M11 Lesson 3](../../module-11-ai-agents-fundamentals/lessons/03-ReAct-Pattern.md)).

Harness strategies:

```python
def _trim_messages(self, messages: list[dict], max_tokens: int) -> list[dict]:
    """Keep system + user goal + recent tail."""
    if self._count_tokens(messages) <= max_tokens:
        return messages

    # Always preserve first two messages (system + original goal)
    head = messages[:2]
    tail = messages[-20:]  # last 20 messages
    return head + [{"role": "system", "content": "[Earlier steps summarized]"}] + tail

def compact_tool_outputs(self, state: AgentState, max_chars: int = 4000):
    """Replace verbose tool outputs in older messages."""
    for msg in state.messages:
        if msg.get("role") == "tool" and len(msg["content"]) > max_chars:
            msg["content"] = msg["content"][:max_chars] + "\n...[truncated by harness]"
```

!!! warning "Compaction is lossy"
    Summarizing or truncating can hide errors the agent needs to recover from. Keep full payloads in `tool_results` even when you trim what the model sees.

---

## Pause and Resume (Human-in-the-Loop)

Checkpoints enable **pause** without losing work:

```python
# Agent requests a sensitive action → harness pauses
if call.name in SENSITIVE_TOOLS:
    state.status = "paused"
    store.save(state)
    return state  # UI prompts human; on approval, reload and continue

# Resume after approval
state = store.load(run_id=run_id, step=last_step)
state.status = "running"
harness.run_from(state)  # continue the loop
```

This pattern is how Cursor and Claude Desktop gate filesystem writes and terminal commands.

---

## Key Takeaways

- The harness implements **perceive** (context), **reason** (model call), and **act** (sandboxed tools) — the model only reasons
- Use **typed state** with separate audit logs (`tool_results`) and model-facing `messages`
- **Checkpoints** at step boundaries enable resume, human approval, and post-mortem replay
- **Context compaction** is a harness responsibility — preserve full history for observability, trim for the model
- Frameworks like LangGraph provide checkpoint primitives; understand the semantics regardless of tooling

---

## Further Reading

- [Agents Towards Production — Agent Architecture](https://github.com/NirDiamant/agents-towards-production) — stateful agent patterns
- [Awesome Harness Engineering](https://github.com/ai-boost/awesome-harness-engineering) — checkpoint and state management references

---

## Next Lesson

**Lesson 3: Tools and Function Calling** — Tool schemas, execution sandboxes, and structured error handling in the harness.
