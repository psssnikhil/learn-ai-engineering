---
title: Parallel vs Sequential Execution
description: >-
  Learn when to run agents concurrently or in sequence, optimize latency and
  cost, and implement fan-out/fan-in with LangGraph and asyncio
duration: 40 min
difficulty: intermediate
has_code: true
module: module-12
---
# Parallel vs Sequential Execution

## What You'll Learn

| What You'll Learn | Time | Difficulty |
|-------------------|------|------------|
| Compare parallel and sequential multi-agent execution | 40 min | Intermediate |
| Build dependency graphs to determine execution order | | |
| Implement fan-out/fan-in patterns for concurrent agents | | |
| Balance latency, cost, and correctness tradeoffs | | |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python 3.10+** | With `asyncio` and typing |
| **Module 12, Lessons 3 and 6** | Orchestrator patterns and handoffs |
| **Module 5: AI Agents** | Agent execution and tool use |
| **Basic graph concepts** | Dependencies, topological ordering |

```bash
pip install openai pydantic python-dotenv
# Optional:
pip install langgraph
```

---

## Intuition First

Imagine assembling a research report. You cannot write the conclusion before gathering data — that step depends on research completing first. But three independent reviewers can read the same draft simultaneously: a security reviewer, a style editor, and a fact-checker do not need each other's output. They all read the same input and produce separate opinions.

**Sequential execution** is an assembly line: each station waits for the previous one. Latency is the sum of all steps, but data flows cleanly. **Parallel execution** is a team working side-by-side on independent subtasks. Latency is roughly the slowest agent in the batch, but you pay for multiple LLM calls at once and must merge results afterward.

The critical question is not "which is faster?" but "what depends on what?" Map dependencies first. If task B reads output from task A, they must be sequential. If tasks B and C both read only the original input, they can run in parallel. Most production workflows are **mixed**: sequential stages with parallel batches inside them.

Getting this wrong causes race conditions (parallel agents writing the same state), wasted tokens (sequential when parallel was safe), or premature synthesis (merging before all branches finish).

---

## Why Execution Order Matters

In multi-agent systems, **how** agents run affects latency, cost, and correctness. Run agents in the wrong order and you get race conditions or wasted tokens. Run everything sequentially when parallelism is safe and users wait unnecessarily.

**Sequential execution** runs agents one after another. Each step depends on the previous output. Research before writing. Design before implementation.

**Parallel execution** runs independent agents at the same time. Security review, style review, and performance review on the same codebase can happen concurrently.

```
Sequential:                         Parallel (fan-out/fan-in):

  A ──→ B ──→ C                       ┌──→ B ──┐
                                      A         ├──→ D (merge)
                                      └──→ C ──┘
```

The [agents-towards-production](https://github.com/NirDiamant/agents-towards-production) repository demonstrates both modes: sequential pipelines for data dependencies and parallel map-reduce for independent analysis.

---

## What You'll Build

By the end of this lesson, you will have:

- [ ] A dependency analyzer that produces parallel execution batches
- [ ] A mixed execution engine running sequential batches with parallel steps inside
- [ ] Fan-out/fan-in (map-reduce) for code review agents
- [ ] Concurrency limiting with semaphores for rate limit safety
- [ ] A runnable demo comparing sequential vs parallel latency

---

## Step 1: Build a Dependency Analyzer

Before choosing execution mode, map **task dependencies**.

| Dependency Type | Execution | Example |
|-----------------|-----------|---------|
| **Data dependency** | Sequential | Writer needs research output |
| **Independent analysis** | Parallel | Three reviewers on same input |
| **Partial overlap** | Mixed | Parallel research, sequential synthesis |
| **Resource conflict** | Sequential | Two agents writing same file |

```python
# src/execution/dependencies.py
from dataclasses import dataclass

@dataclass
class AgentTask:
    name: str
    depends_on: list[str]

def execution_order(tasks: list[AgentTask]) -> list[list[str]]:
    """Return batches: each batch can run in parallel internally."""
    completed: set[str] = set()
    batches: list[list[str]] = []
    task_map = {t.name: t for t in tasks}

    while len(completed) < len(tasks):
        ready = [
            t.name for t in tasks
            if t.name not in completed
            and all(d in completed for d in t.depends_on)
        ]
        if not ready:
            raise ValueError("Circular dependency detected")
        batches.append(ready)
        completed.update(ready)

    return batches

# Example: research -> parallel reviews -> synthesis
tasks = [
    AgentTask("research", []),
    AgentTask("security_review", ["research"]),
    AgentTask("style_review", ["research"]),
    AgentTask("synthesis", ["security_review", "style_review"]),
]
# Result: [["research"], ["security_review", "style_review"], ["synthesis"]]
```

---

## Step 2: Implement Fan-Out / Fan-In

**Fan-out** distributes the same input to multiple agents. **Fan-in** merges their results.

```python
# src/execution/parallel.py
import asyncio
import time
from typing import Any, Callable, Awaitable

AgentFn = Callable[[str], Awaitable[dict]]

async def fan_out(agents: list[AgentFn], input_data: str) -> list[dict]:
    return await asyncio.gather(*[agent(input_data) for agent in agents])

async def fan_in_merger(reviews: list[dict]) -> str:
    total_issues = sum(r.get("issues", 0) for r in reviews)
    agents = ", ".join(r.get("agent", "unknown") for r in reviews)
    return f"Merged {len(reviews)} reviews ({agents}): {total_issues} total issues"

async def security_agent(code: str) -> dict:
    await asyncio.sleep(0.5)
    return {"agent": "security", "issues": 2}

async def style_agent(code: str) -> dict:
    await asyncio.sleep(0.5)
    return {"agent": "style", "issues": 5}

async def perf_agent(code: str) -> dict:
    await asyncio.sleep(0.5)
    return {"agent": "performance", "issues": 1}

async def parallel_code_review(code: str) -> str:
    reviews = await fan_out(
        [security_agent, style_agent, perf_agent], code
    )
    return await fan_in_merger(reviews)
```

---

## Step 3: Build the Mixed Execution Engine

The engine runs batches sequentially but executes agents within each batch in parallel.

```python
# src/execution/engine.py
import asyncio
import time
from typing import Callable, Awaitable

AgentFn = Callable[[str], Awaitable[str]]

class ExecutionEngine:
    def __init__(
        self,
        agents: dict[str, AgentFn],
        max_concurrency: int = 3,
    ):
        self.agents = agents
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def run(
        self,
        plan: list[list[str]],
        initial_input: str,
        input_resolver: Callable[[str, dict[str, str]], str] | None = None,
    ) -> dict[str, str]:
        results: dict[str, str] = {"input": initial_input}
        total_start = time.time()

        for batch_idx, batch in enumerate(plan):
            print(f"Batch {batch_idx + 1}: {batch}")
            batch_start = time.time()

            tasks = []
            for agent_name in batch:
                input_data = (
                    input_resolver(agent_name, results)
                    if input_resolver
                    else results.get("research", results["input"])
                )
                tasks.append(self._run_limited(agent_name, input_data))

            batch_outputs = await asyncio.gather(*tasks)
            for name, output in zip(batch, batch_outputs):
                results[name] = output

            elapsed = (time.time() - batch_start) * 1000
            print(f"  Batch completed in {elapsed:.0f}ms")

        results["_total_ms"] = (time.time() - total_start) * 1000
        return results

    async def _run_limited(self, name: str, input_data: str) -> str:
        async with self.semaphore:
            return await self.agents[name](input_data)
```

---

## Step 4: Compare Sequential vs Parallel Latency

```python
# demo_execution.py
import asyncio
import time
from src.execution.engine import ExecutionEngine
from src.execution.dependencies import AgentTask, execution_order
from src.execution.parallel import (
    security_agent, style_agent, perf_agent, fan_in_merger, fan_out,
)

async def mock_research(inp: str) -> str:
    await asyncio.sleep(0.4)
    return f"Research findings for: {inp}"

async def mock_synthesis(inp: str) -> str:
    await asyncio.sleep(0.3)
    return f"Synthesis of: {inp[:100]}"

async def wrap_agent(fn, name: str):
    async def agent(inp: str) -> str:
        result = await fn(inp)
        return str(result)
    agent.__name__ = name
    return agent

async def main():
    # Sequential: run reviewers one at a time
    code = "def authenticate(user): pass"
    seq_start = time.time()
    r1 = await security_agent(code)
    r2 = await style_agent(code)
    r3 = await perf_agent(code)
    seq_result = await fan_in_merger([r1, r2, r3])
    seq_ms = (time.time() - seq_start) * 1000

    # Parallel: run reviewers concurrently
    par_start = time.time()
    reviews = await fan_out([security_agent, style_agent, perf_agent], code)
    par_result = await fan_in_merger(reviews)
    par_ms = (time.time() - par_start) * 1000

    print(f"Sequential review: {seq_ms:.0f}ms")
    print(f"Parallel review:   {par_ms:.0f}ms")
    print(f"Speedup:           {seq_ms / par_ms:.1f}x")
    print(f"\nResult: {par_result}")

    # Mixed engine demo
    async def security_review_str(inp: str) -> str:
        return str(await security_agent(inp))

    async def style_review_str(inp: str) -> str:
        return str(await style_agent(inp))

    plan = execution_order([
        AgentTask("research", []),
        AgentTask("security_review", ["research"]),
        AgentTask("style_review", ["research"]),
        AgentTask("synthesis", ["security_review", "style_review"]),
    ])
    engine = ExecutionEngine(
        agents={
            "research": mock_research,
            "security_review": security_review_str,
            "style_review": style_review_str,
            "synthesis": mock_synthesis,
        },
        max_concurrency=3,
    )

    def resolve_input(agent_name: str, results: dict[str, str]) -> str:
        if agent_name == "research":
            return results["input"]
        if agent_name in ("security_review", "style_review"):
            return results["research"]
        deps = f"Security: {results.get('security_review', '')}\nStyle: {results.get('style_review', '')}"
        return deps

    mixed = await engine.run(plan, "Review auth module", resolve_input)
    print(f"\nMixed engine total: {mixed['_total_ms']:.0f}ms")
    print(f"Synthesis preview: {mixed.get('synthesis', '')[:80]}...")

if __name__ == "__main__":
    asyncio.run(main())
```

The sequential vs parallel comparison above shows typical speedup when three 500ms reviewers run concurrently instead of back-to-back.

---

## Framework Patterns

### LangGraph Send API

LangGraph's `Send` primitive dispatches parallel branches dynamically. Each branch runs as a separate graph invocation and results merge at a reducer node.

```python
from langgraph.types import Send

def fan_out_node(state: dict) -> list[Send]:
    code = state["code"]
    return [
        Send("security_reviewer", {"code": code}),
        Send("style_reviewer", {"code": code}),
        Send("perf_reviewer", {"code": code}),
    ]

def merge_node(state: dict) -> dict:
    reviews = state.get("reviews", [])
    return {"summary": f"Merged {len(reviews)} reviews"}
```

### CrewAI Process Modes

CrewAI offers `Process.sequential` and `Process.hierarchical`. Tasks with `context=[prior_task]` enforce sequencing. Independent tasks without context dependencies can run concurrently.

### Sequential Pipeline

The simplest pattern: Agent 1 output becomes Agent 2 input.

```python
async def sequential_pipeline(query: str) -> str:
    research = await research_agent(query)
    analysis = await analysis_agent(research)
    report = await report_agent(analysis)
    return report
```

---

## Tradeoff Matrix

| Factor | Sequential | Parallel |
|--------|------------|----------|
| **Latency** | Sum of all agents | Max of parallel batch |
| **Token cost** | Lower (no duplication) | Higher (repeated context) |
| **Correctness** | Safe for dependencies | Risk of races on shared state |
| **Debugging** | Easier (linear trace) | Harder (concurrent logs) |
| **Rate limits** | One call at a time | May hit API limits |

**Rule of thumb:** Parallelize when agents are **read-only** on shared input. Serialize when agents **write** to shared state or depend on each other's output.

---

## Failure Modes

**False parallelism:** Running agents "in parallel" that both modify shared state causes conflicts. Use section ownership (Lesson 8) or keep parallel agents read-only.

**Over-parallelization:** Ten concurrent LLM calls may hit rate limits. Cap concurrency with semaphores (default 3-5).

**Premature synthesis:** The merge agent runs before all parallel branches finish. Always fan-in with explicit synchronization (`asyncio.gather` or LangGraph reducer).

**Hidden dependencies:** Two tasks appear independent but both write to the same blackboard key. Map dependencies explicitly — do not assume.

**Cost blindness:** Parallel review of the same 10K-token document three times triples input token cost. Budget for parallel duplication.

**Circular dependencies:** Task A depends on B, B depends on A. The dependency analyzer must detect and reject cycles.

---

## Production Notes

| Concern | Development | Production |
|---------|-------------|------------|
| Execution planning | Manual batches | Dependency graph from task metadata |
| Concurrency | Unlimited | Semaphore capped to provider rate limits |
| Timeouts | None | Per-agent timeout; cancel slow branches |
| Partial failure | Fail all | Fan-in with partial results + retry |
| Observability | Print batches | Trace spans per batch and per agent |
| Cost | Ignored | Log tokens per batch; alert on duplication |

**Concurrency semaphores:**

```python
semaphore = asyncio.Semaphore(3)

async def limited_agent(fn, input_data):
    async with semaphore:
        return await fn(input_data)
```

**Partial failure in fan-in:** If one of three reviewers fails, decide policy upfront: fail the whole batch, retry the failed branch, or merge available results with a warning. Do not silently drop failed branches.

**Dynamic parallelism:** LangGraph `Send` enables runtime fan-out (e.g., one branch per file in a repo). Cap maximum branches to prevent runaway cost.

---

## Key Takeaways

- Map dependencies first — sequential when B needs A's output, parallel when agents are independent
- Fan-out/fan-in (map-reduce) is the core parallel pattern for multi-agent systems
- Most production workflows are mixed: sequential batches with parallel steps inside each batch
- LangGraph Send, asyncio.gather, and CrewAI task context implement execution control
- Parallel cuts latency but increases token cost and debugging complexity
- Cap concurrency with semaphores, detect circular dependencies, and never merge before all branches complete

---

## Next Lesson

**[Lesson 8: Shared Memory and Blackboards](./08-shared-memory-and-blackboards.md)** — Learn how agents read and write shared state safely, and when blackboard architectures beat message passing.
