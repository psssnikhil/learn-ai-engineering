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

## 🎯 Learning Objectives

| What You'll Learn | Time | Difficulty |
|-------------------|------|------------|
| Compare parallel and sequential multi-agent execution | 40 min | Intermediate |
| Build dependency graphs to determine execution order | | |
| Implement fan-out/fan-in patterns for concurrent agents | | |
| Balance latency, cost, and correctness tradeoffs | | |

---

## 📚 Why Execution Order Matters

In multi-agent systems, **how** agents run affects latency, cost, and correctness. Run agents in the wrong order and you get race conditions or wasted tokens. Run everything sequentially when parallelism is safe and users wait unnecessarily.

**Sequential execution** runs agents one after another. Each step depends on the previous output. Research before writing. Design before implementation.

**Parallel execution** runs independent agents at the same time. Security review, style review, and performance review on the same codebase can happen concurrently.

```
Sequential:                         Parallel (fan-out/fan-in):

  A ──→ B ──→ C                       ┌──→ B ──┐
                                      A         ├──→ D (merge)
                                      └──→ C ──┘
```

The [agents-towards-production](https://github.com/NirDiamant/agents-towards-production) repository demonstrates both modes: sequential pipelines for data dependencies and parallel map-reduce for independent analysis. Microsoft's [AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners) course teaches map-reduce as a core multi-agent pattern.

---

## 🔗 Dependency Analysis

Before choosing execution mode, map **task dependencies**:

| Dependency Type | Execution | Example |
|-----------------|-----------|---------|
| **Data dependency** | Sequential | Writer needs research output |
| **Independent analysis** | Parallel | Three reviewers on same input |
| **Partial overlap** | Mixed | Parallel research, sequential synthesis |
| **Resource conflict** | Sequential | Two agents writing same file |

Build a simple dependency graph. If task B reads output from task A, they must be sequential. If tasks B and C both read only the original input, they can run in parallel.

```python
from dataclasses import dataclass

@dataclass
class AgentTask:
    name: str
    depends_on: list[str]

def execution_order(tasks: list[AgentTask]) -> list[list[str]]:
    """Return batches: each batch can run in parallel."""
    completed = set()
    batches = []

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

tasks = [
    AgentTask("research", []),
    AgentTask("security_review", ["research"]),
    AgentTask("style_review", ["research"]),
    AgentTask("synthesis", ["security_review", "style_review"]),
]
# Result: [["research"], ["security_review", "style_review"], ["synthesis"]]
```

---

## ⚡ Parallel Patterns

### Fan-Out / Fan-In (Map-Reduce)

**Fan-out:** Distribute the same input to multiple agents. **Fan-in:** A merger agent combines results.

```python
import asyncio
from typing import Any

async def fan_out(agent_fn, inputs: list[str]) -> list[Any]:
    return await asyncio.gather(*[agent_fn(inp) for inp in inputs])

async def security_agent(code: str) -> dict:
    await asyncio.sleep(0.5)
    return {"agent": "security", "issues": 2}

async def style_agent(code: str) -> dict:
    await asyncio.sleep(0.5)
    return {"agent": "style", "issues": 5}

async def merge_reviews(reviews: list[dict]) -> str:
    total = sum(r["issues"] for r in reviews)
    return f"Found {total} total issues across {len(reviews)} reviewers"

async def parallel_review(code: str) -> str:
    reviews = await asyncio.gather(
        security_agent(code),
        style_agent(code),
    )
    return await merge_reviews(reviews)
```

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

CrewAI offers `Process.sequential` and `Process.hierarchical`. For parallel work within a crew, define independent tasks without `context` dependencies and use async execution. Tasks with `context=[prior_task]` enforce sequencing.

---

## 🔗 Sequential Patterns

### Pipeline

The simplest pattern: Agent 1 output becomes Agent 2 input. LangGraph chains nodes with directed edges. CrewAI uses task `context` arrays.

```python
async def sequential_pipeline(query: str) -> str:
    research = await research_agent(query)
    analysis = await analysis_agent(research)
    report = await report_agent(analysis)
    return report
```

### AutoGen Sequential Chat

AutoGen's round-robin or custom speaker selection can enforce order. A group chat manager picks agents in a fixed sequence for pipeline workflows.

---

## 📊 Tradeoff Matrix

| Factor | Sequential | Parallel |
|--------|------------|----------|
| **Latency** | Sum of all agents | Max of parallel batch |
| **Token cost** | Lower (no duplication) | Higher (repeated context) |
| **Correctness** | Safe for dependencies | Risk of races on shared state |
| **Debugging** | Easier (linear trace) | Harder (concurrent logs) |
| **Rate limits** | One call at a time | May hit API limits |

**Rule of thumb:** Parallelize when agents are **read-only** on shared input. Serialize when agents **write** to shared state or depend on each other's output.

---

## 💻 Mixed Execution Engine

```python
import asyncio
from typing import Callable, Awaitable

AgentFn = Callable[[str], Awaitable[str]]

class ExecutionEngine:
    def __init__(self, agents: dict[str, AgentFn]):
        self.agents = agents

    async def run_batch(self, plan: list[list[str]], initial_input: str) -> dict[str, str]:
        results = {"input": initial_input}

        for batch in plan:
            print(f"  ⚡ Running batch: {batch}")
            tasks = []
            for agent_name in batch:
                deps = self._resolve_input(agent_name, results)
                tasks.append(self._run_agent(agent_name, deps))

            batch_results = await asyncio.gather(*tasks)
            for name, output in zip(batch, batch_results):
                results[name] = output

        return results

    def _resolve_input(self, agent_name: str, results: dict) -> str:
        return results.get("research", results["input"])

    async def _run_agent(self, name: str, input_data: str) -> str:
        return await self.agents[name](input_data)

# Plan: research first, then parallel reviews, then synthesis
plan = [["research"], ["security_review", "style_review"], ["synthesis"]]
```

---

## ⚠️ Pitfalls

**False parallelism:** Running agents "in parallel" that both modify shared state causes conflicts. Use shared memory with locks (Lesson 8) or keep parallel agents read-only.

**Over-parallelization:** Ten concurrent LLM calls may hit rate limits. Add semaphores to cap concurrency.

```python
semaphore = asyncio.Semaphore(3)

async def limited_agent(fn, input_data):
    async with semaphore:
        return await fn(input_data)
```

**Premature synthesis:** The merge agent runs before all parallel branches finish. Always fan-in with explicit synchronization.

---

## 🎓 Key Takeaways

```
✅ Map dependencies first — sequential when B needs A's output, parallel when independent
✅ Fan-out/fan-in (map-reduce) is the core parallel pattern for multi-agent systems
✅ LangGraph Send, asyncio.gather, and CrewAI task context implement execution control
✅ Parallel cuts latency but increases token cost and debugging complexity
✅ Cap concurrency with semaphores to respect rate limits
```

---

## 🚀 Next Lesson

**Lesson 8: Shared Memory and Blackboards** — Learn how agents read and write shared state safely.

You'll learn:
- 🗂️ Blackboard architectures for collaborative problem-solving
- 🔒 Avoiding race conditions in shared state
- 📋 LangGraph state reducers and checkpointing
- 🔄 When shared memory beats message passing

---

## 📚 Additional Resources

- 📄 [LangGraph Map-Reduce](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/)
- 💻 [CrewAI Processes](https://docs.crewai.com/core-concepts/Processes/)
- 📖 [Microsoft AI Agents for Beginners — Parallel Agents](https://github.com/microsoft/ai-agents-for-beginners)
- 🔧 [agents-towards-production concurrent patterns](https://github.com/NirDiamant/agents-towards-production)

---

*⏱️ Estimated time: 40 minutes | 📊 Difficulty: Intermediate | 💻 Includes code examples*
