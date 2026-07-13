---
title: Agent Loop and State
description: >-
  Implement the perceive-reason-act cycle in a harness, manage conversation and
  working state, and design checkpoints for resume and audit
duration: 60 min
difficulty: advanced
has_code: true
module: module-18
---
# Agent Loop and State

## Prerequisites

- [Lesson 1 — What Is an Agent Harness?](01-what-is-an-agent-harness.md): loop, state, and termination primitives
- [M11 Lesson 3 — ReAct Pattern](../../module-11-ai-agents-fundamentals/lessons/03-ReAct-Pattern.md): the reason-act trace structure
- Comfortable with Python dataclasses, `uuid`, and JSON serialization
- Basic understanding of context window / token counting

---

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Implement perceive-reason-act inside a harness, with each phase clearly separated | 60 min | Advanced |
| Design typed state schemas that separate the model's view from the audit log | | |
| Build checkpoint and restore for long-running and resumable agents | | |
| Handle context growth without losing critical information | | |
| Implement human-in-the-loop pause and resume at the harness level | | |

---

## Intuition First

Every software system that manages long-running work must solve two problems: **what is the current state?** and **how do we recover if something fails?**

For databases, the answer is transactions and WAL logs. For agents, the answer is a typed state schema and checkpoints.

Without structured state, an agent run is a black box. You know the user's input and the final output. If the run fails at step 8 of 15, you lose all work. If a human needs to approve a sensitive action at step 5, there's nowhere to pause.

With structured state, every step is observable: you know which tool was called, what arguments were passed, how long it took, and whether it succeeded. You can pause at any step, serialize the state to disk, and resume hours later.

The harness is the component that *owns* this structure. The model sees a window of messages. The harness sees everything.

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

**Why separate `tool_results` from `messages`?**

`messages` is the model's view — it may be trimmed, compacted, or restructured for different providers. `tool_results` is the harness's audit log — full payloads, timings, and success flags that live independently of what the model sees.

**The `scratchpad` field:** A typed dictionary for harness-managed working memory — the agent's current plan, a list of discovered entity IDs, a running count of retries. Storing this in state (rather than in messages) keeps it out of the model's context unless the perceive phase explicitly injects it.

**Worked example — state after 3 steps:**

```python
# Initial state
state = AgentState(goal="Find pricing for plans A and B")
print(state.step)           # 0
print(state.total_cost_usd) # 0.0

# After step 1 (search tool call)
state.step = 1
state.total_tokens = 1_847
state.total_cost_usd = 0.00277

# After step 2 (second search)
state.step = 2
state.total_tokens = 3_412
state.total_cost_usd = 0.00512

# After step 3 (final answer, no tool calls)
state.step = 3
state.status = "finished"
state.total_tokens = 4_891
state.total_cost_usd = 0.00734
```

At any step, you can serialize this to disk, inspect it, or pass it to a monitoring system.

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

    def list_checkpoints(self, run_id: str) -> list[int]:
        """Return step numbers for all saved checkpoints for a run."""
        steps = []
        for path in self.base_dir.glob(f"{run_id}_step*.json"):
            step_str = path.stem.split("_step")[-1]
            try:
                steps.append(int(step_str))
            except ValueError:
                pass
        return sorted(steps)
```

### When to Checkpoint

| Strategy | Trade-off | Best for |
|----------|-----------|----------|
| **Every step** | Maximum recoverability; more I/O | Short runs, critical tasks |
| **After tool calls** | Good pause points for human-in-the-loop | Coding agents, file writers |
| **On error** | Capture failure context before retry | Debugging unstable tools |
| **Periodic (every N steps)** | Balance for long research tasks | Multi-hour research agents |

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

Each loop iteration appends to `messages`. Token usage grows roughly **linearly with step count** for the prompt side — a common source of cost blowups.

**Numerical example:**

```
Step 0: system(200) + user(50) = 250 tokens in context
Step 1: + assistant(tool_call 80) + tool_result(400) = 730 tokens
Step 2: + assistant(tool_call 80) + tool_result(600) = 1,410 tokens
Step 5: ~4,000 tokens
Step 10: ~8,000 tokens
Step 20: ~16,000 tokens
Step 50: ~40,000 tokens (approaching many context limits)
```

At `gpt-4o` pricing ($2.50/1M input tokens), 50 steps costs ~$0.10 in *input tokens alone* per run. At 10,000 runs/day, that's $1,000/day just from context accumulation.

Harness strategies for managing growth:

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

def summarize_old_steps(self, state: AgentState, client, keep_recent: int = 5) -> AgentState:
    """Summarize steps older than keep_recent into a single system note."""
    if len(state.messages) <= keep_recent + 2:  # +2 for system + original user
        return state

    to_summarize = state.messages[2:-keep_recent]
    summary_resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Summarize the following agent steps concisely. Include: tools called, key findings, and any decisions made."},
            {"role": "user", "content": str(to_summarize)},
        ],
    )
    summary = summary_resp.choices[0].message.content

    state.messages = (
        state.messages[:2]  # system + original goal
        + [{"role": "system", "content": f"[Summary of earlier steps]: {summary}"}]
        + state.messages[-keep_recent:]
    )
    return state
```

!!! warning "Compaction is lossy"
    Summarizing or truncating can hide errors the agent needs to recover from. Keep full payloads in `tool_results` even when you trim what the model sees.

---

## Pause and Resume (Human-in-the-Loop)

Checkpoints enable **pause** without losing work:

```python
SENSITIVE_TOOLS = {"write_file", "run_terminal", "send_email", "delete_resource"}

def run_with_hitl(
    self,
    state: AgentState,
    approval_callback: callable,  # (tool_name, args) -> bool
) -> AgentState:
    """Loop with human-in-the-loop pause for sensitive tool calls."""
    store = CheckpointStore(self.checkpoint_dir)

    while state.status == "running" and state.step < self.policy.max_steps:
        turn = self.reason(state)
        response = turn["response"]

        if not response.tool_calls:
            state.append_assistant(response.text)
            state.status = "finished"
            break

        state.append_assistant(response.text, response.tool_calls)

        for tool_call in response.tool_calls:
            if tool_call.name in SENSITIVE_TOOLS:
                # Pause and ask for approval
                state.status = "paused"
                store.save(state)

                approved = approval_callback(tool_call.name, tool_call.arguments)
                if not approved:
                    state.append_tool_result(ToolResult(
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                        arguments=tool_call.arguments,
                        output='{"error": "User rejected this action. Try a different approach."}',
                        duration_ms=0,
                        success=False,
                    ))
                    state.status = "running"
                    continue

            # Execute approved or non-sensitive tool
            results = self.act(state, [tool_call])
            for r in results:
                state.append_tool_result(r)

            state.status = "running"

        state.step += 1

    return state

# Resume from checkpoint after human approves
def resume_from_checkpoint(run_id: str, step: int, harness) -> AgentState:
    store = CheckpointStore(harness.checkpoint_dir)
    state = store.load(run_id, step)
    state.status = "running"
    return harness.run_with_hitl(state, approval_callback=interactive_approval)
```

This pattern is how Cursor and Claude Desktop gate filesystem writes and terminal commands.

---

## State Machine View

It helps to think of `AgentState.status` as a mini state machine — each transition is explicit and testable:

```
            ┌──────────┐
 goal ─────▶│ running  │◀──────────────────────┐
            └────┬─────┘                        │
                 │ [no tool calls]  [tool approved, resume]
                 │                              │
          ┌──────▼──────┐      ┌───────────────┴─┐
          │  finished   │      │    paused (HITL) │
          └─────────────┘      └─────────────────┘
                 │
          ┌──────▼──────┐
          │   failed    │  ← budget exceeded, unrecoverable error
          └─────────────┘
```

Every `status` transition should be logged and checkpointed. Test all four paths (running → finished, running → paused → running, running → failed, step budget → finished) with unit tests before the first production deployment.

---

## Common Misconceptions

**"I can reconstruct state from the message history alone."** You can reconstruct *what the model saw*, but not the full audit data — tool execution durations, success/failure status, raw arguments before validation, approval records. Typed `ToolResult` objects capture these separately.

**"Checkpoints are only needed for long runs."** Any agent that makes irreversible side effects (file writes, API calls, database mutations) should checkpoint before those actions — even on a 2-step run. A checkpoint before the dangerous step enables forensics when something goes wrong.

**"Compaction loses context the model needed."** Full payloads live in `tool_results`. The model's *view* can be summarized; the harness's audit log stays intact. This is the correct separation.

**"Step count equals iteration count."** Not always. If you execute parallel tool calls in a single step (multiple `tool_calls` in one response), the harness can execute all of them before incrementing `step`. The semantics of "step" are yours to define consistently.

---

## Production Tips

- **Use typed state from day one.** Migrating from a flat dict to a `@dataclass` after you have 10,000 checkpoint files in S3 is painful. Design the schema before writing the loop.
- **Store `run_id` everywhere.** Correlation across logs, traces, checkpoints, and user-facing output becomes trivial when every artifact carries the same `run_id`.
- **Version your checkpoint schema.** Add a `"schema_version": 1` field. When you change the state structure, bump the version and write a migration loader.
- **Test restore.** Write a test that saves a checkpoint after step 3, loads it, and continues the loop. Restore bugs are common and silent — they only appear in the middle of production incidents.
- **Compact aggressively, audit completely.** Trim messages for the model, but never throw away `tool_results`. Storage is cheap; debugging without audit data is not.

---

## Key Takeaways

- The harness implements **perceive** (context), **reason** (model call), and **act** (sandboxed tools) — the model only reasons
- Use **typed state** with separate audit logs (`tool_results`) and model-facing `messages` — they serve different consumers
- **Checkpoints** at step boundaries enable resume, human approval, and post-mortem replay
- **Context compaction** is a harness responsibility — preserve full history for observability, trim for the model
- Token counts grow linearly with steps; design compaction strategy before you hit context limits in production
- Frameworks like LangGraph provide checkpoint primitives; understand the semantics regardless of tooling

---

## Related Papers

| Paper | Year | Key contribution |
|-------|------|-----------------|
| [Cognitive Architectures for Language Agents (CoALA)](https://arxiv.org/abs/2309.02427) | 2023 | Formal framework for agent memory types: working, episodic, semantic, procedural |
| [A Survey on Large Language Model-based Autonomous Agents](https://arxiv.org/abs/2308.11432) | 2023 | Memory and state management patterns across agent architectures |
| [LangGraph: Building Stateful, Multi-Actor Applications with LLMs](https://langchain-ai.github.io/langgraph/) | 2024 | Checkpoint-backed state machine for production agent loops |
| [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291) | 2023 | Long-running agent with skill library (persistent state across sessions) |

---

## Further Reading

- [Agents Towards Production — Agent Architecture](https://github.com/NirDiamant/agents-towards-production) — stateful agent patterns
- [Awesome Harness Engineering](https://github.com/ai-boost/awesome-harness-engineering) — checkpoint and state management references

---

## Next Lesson

**[Lesson 3: Tools and Function Calling](03-tools-and-function-calling.md)** — Tool schemas, execution sandboxes, and structured error handling in the harness.
