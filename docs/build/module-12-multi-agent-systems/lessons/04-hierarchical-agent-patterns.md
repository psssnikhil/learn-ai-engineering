---
title: Hierarchical Agent Patterns
description: >-
  Design multi-tier agent hierarchies where managers delegate to specialists,
  inspired by LangGraph subgraphs and CrewAI crew structures
duration: 40 min
difficulty: intermediate
has_code: true
module: module-12
---
# Hierarchical Agent Patterns

## What You'll Learn

| What You'll Learn | Time | Difficulty |
|-------------------|------|------------|
| Understand hierarchical multi-agent architectures | 40 min | Intermediate |
| Design manager-worker and nested delegation trees | | |
| Implement hierarchical patterns with LangGraph and CrewAI | | |
| Choose when hierarchy beats flat orchestration | | |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python 3.10+** | With `asyncio` and basic typing |
| **Module 12, Lesson 3** | Orchestrator-worker pattern and task delegation |
| **Module 5: AI Agents** | Agent fundamentals, system prompts, tool use |
| **Optional: LangGraph or CrewAI** | Helpful for framework examples below |

```bash
pip install openai pydantic python-dotenv
# Optional for framework sections:
pip install langgraph crewai
```

---

## Intuition First

Imagine building a software project at a company. The CEO does not write every line of code or read every research paper. They set direction, assign domains to VPs, and expect consolidated reports back. Each VP manages a team with its own internal workflow. Research might split into web search, document parsing, and citation checking — all without the CEO micromanaging each step.

**Hierarchical agent patterns** apply the same idea to multi-agent systems. A root orchestrator receives the user request and delegates to domain managers. Each manager breaks work into subtasks for leaf workers, then synthesizes results before returning upward. Control flows down; results bubble up.

This differs from the flat orchestrator-worker pattern (Lesson 3), where one coordinator directly manages all workers. Hierarchy adds **nested delegation**: managers have autonomy within their domain, narrower context windows per agent, and clearer accountability boundaries. The tradeoff is more hops (latency, tokens) and more layers to debug.

Use hierarchy when your system has natural **domain splits** — research, engineering, compliance — each needing internal coordination. Stick with flat orchestration when you have fewer than six specialists and a linear workflow.

---

## What Are Hierarchical Agent Patterns?

**Hierarchical agent patterns** organize agents into layers of authority. A top-level agent receives the user request, decomposes it, and delegates to mid-level managers or specialists. Each layer narrows scope until leaf agents execute concrete work. Results bubble back up for synthesis.

This mirrors how organizations operate: executives set direction, managers coordinate teams, and individual contributors produce output. In production systems, hierarchy reduces cognitive load per agent and creates clear accountability boundaries.

### Architecture Overview

```
                    ┌─────────────────┐
                    │  Root Orchestrator │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ↓              ↓              ↓
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │ Manager  │  │ Manager  │  │ Manager  │
        │ Research │  │  Code    │  │  QA      │
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             │             │             │
        ┌────┴────┐   ┌────┴────┐   ┌────┴────┐
        ↓         ↓   ↓         ↓   ↓         ↓
     Worker   Worker Worker  Worker Worker  Worker
```

Unlike a flat orchestrator-worker setup, hierarchy supports **nested delegation**. A research manager can spawn sub-agents for web search, document parsing, and citation validation without involving the root orchestrator in every decision.

---

## When to Use Hierarchy

### Perfect For

| Use Case | Why Hierarchy Works |
|----------|----------------------|
| **Large domain-split systems** | Each domain has its own manager and workers |
| **Deep task decomposition** | Subtasks need further breakdown before execution |
| **Context window limits** | Each agent sees only its slice of the problem |
| **Team-style workflows** | Research crew + code crew + QA crew in parallel |

### Not Ideal For

- Fewer than six agents with a linear pipeline (use flat orchestrator)
- Ultra-low latency requirements (each tier adds hops)
- Tasks where every decision must go through one brain
- Simple single-domain workflows with no internal sub-structure

---

## What You'll Build

By the end of this lesson, you will have:

- [ ] A three-tier hierarchy: root orchestrator, domain managers, leaf workers
- [ ] LLM-powered managers that decompose tasks and synthesize sub-results
- [ ] Parallel execution within each manager's team
- [ ] Context summarization at tier boundaries
- [ ] A runnable demo that produces a consolidated multi-domain report

---

## Step 1: Define Leaf Workers and Managers

Leaf workers execute concrete subtasks. Managers decompose, delegate, and synthesize within their domain.

```python
# src/hierarchy/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import asyncio
import time

@dataclass
class TaskResult:
    agent: str
    output: str
    success: bool
    latency_ms: float = 0.0

class LeafWorker(ABC):
    def __init__(self, name: str, capability: str):
        self.name = name
        self.capability = capability

    @abstractmethod
    async def execute(self, subtask: str) -> TaskResult:
        pass

class SimpleLeafWorker(LeafWorker):
    """Mock worker for demonstration — replace with LLM calls in production."""

    async def execute(self, subtask: str) -> TaskResult:
        start = time.time()
        await asyncio.sleep(0.3)
        latency = (time.time() - start) * 1000
        return TaskResult(
            agent=self.name,
            output=f"[{self.capability}] Completed: {subtask[:60]}",
            success=True,
            latency_ms=latency,
        )

class ManagerAgent:
    def __init__(self, name: str, workers: list[LeafWorker]):
        self.name = name
        self.workers = workers

    def decompose(self, task: str) -> list[str]:
        """Split task into one subtask per worker."""
        return [f"{task} — {w.capability} aspect" for w in self.workers]

    async def delegate(self, task: str) -> TaskResult:
        print(f"  [{self.name}] decomposing: {task[:50]}...")
        subtasks = self.decompose(task)
        results = await asyncio.gather(
            *[w.execute(st) for w, st in zip(self.workers, subtasks)]
        )
        combined = "\n".join(r.output for r in results if r.success)
        total_latency = sum(r.latency_ms for r in results)
        return TaskResult(
            agent=self.name,
            output=self.synthesize(combined),
            success=all(r.success for r in results),
            latency_ms=total_latency,
        )

    def synthesize(self, worker_outputs: str) -> str:
        """Compress worker outputs before passing upward."""
        lines = worker_outputs.strip().split("\n")
        return f"**{self.name} summary ({len(lines)} workers):**\n" + worker_outputs
```

---

## Step 2: Build the Root Orchestrator

The root orchestrator routes to domain managers, collects their summaries, and produces a final deliverable.

```python
# src/hierarchy/root.py
from dataclasses import dataclass, field
from src.hierarchy.base import ManagerAgent, TaskResult

@dataclass
class HierarchyLog:
    tier: str
    agent: str
    latency_ms: float
    success: bool

class RootOrchestrator:
    def __init__(self, managers: dict[str, ManagerAgent]):
        self.managers = managers
        self.log: list[HierarchyLog] = []

    async def run(self, request: str) -> dict:
        print(f"Root received: {request[:60]}...")
        domain_results: dict[str, TaskResult] = {}

        # Domain managers can run in parallel when independent
        tasks = {domain: mgr.delegate(request) for domain, mgr in self.managers.items()}
        for domain, coro in tasks.items():
            result = await coro
            domain_results[domain] = result
            self.log.append(HierarchyLog("manager", result.agent, result.latency_ms, result.success))

        final = self.synthesize_final(request, domain_results)
        return {
            "status": "success" if all(r.success for r in domain_results.values()) else "partial",
            "final_output": final,
            "domains": {k: v.output for k, v in domain_results.items()},
            "log": [{"tier": e.tier, "agent": e.agent, "latency_ms": e.latency_ms} for e in self.log],
        }

    def synthesize_final(self, request: str, results: dict[str, TaskResult]) -> str:
        sections = "\n\n".join(
            f"## {domain.title()}\n{result.output}"
            for domain, result in results.items()
        )
        return f"# Final Report: {request[:80]}\n\n{sections}"
```

---

## Step 3: Wire Up a Three-Tier Hierarchy

```python
# demo_hierarchy.py
import asyncio
from src.hierarchy.base import SimpleLeafWorker, ManagerAgent
from src.hierarchy.root import RootOrchestrator

async def main():
    research_mgr = ManagerAgent("Research Manager", [
        SimpleLeafWorker("Searcher", "web_search"),
        SimpleLeafWorker("Summarizer", "summarize"),
        SimpleLeafWorker("Citer", "citation_check"),
    ])
    code_mgr = ManagerAgent("Code Manager", [
        SimpleLeafWorker("Architect", "design"),
        SimpleLeafWorker("Developer", "implement"),
    ])
    qa_mgr = ManagerAgent("QA Manager", [
        SimpleLeafWorker("Tester", "test"),
        SimpleLeafWorker("Reviewer", "review"),
    ])

    orchestrator = RootOrchestrator({
        "research": research_mgr,
        "code": code_mgr,
        "qa": qa_mgr,
    })

    result = await orchestrator.run(
        "Build a REST API for user authentication with OAuth2 support"
    )

    print(f"\nStatus: {result['status']}")
    print(f"Log entries: {len(result['log'])}")
    print(f"\n--- Final Output ---\n{result['final_output'][:600]}...")

if __name__ == "__main__":
    asyncio.run(main())
```

Run with:

```bash
python demo_hierarchy.py
```

---

## Framework Patterns

### LangGraph: Subgraphs as Hierarchy

LangGraph models hierarchy through **parent graphs** and **subgraphs**. The parent graph routes to a subgraph node; the subgraph runs its own state machine and returns consolidated output.

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class TeamState(TypedDict):
    task: str
    research: str
    code: str
    final_report: str

def research_subgraph_node(state: TeamState) -> TeamState:
    raw = web_search_agent(state["task"])
    summary = summarizer_agent(raw)
    return {"research": summary}

def code_subgraph_node(state: TeamState) -> TeamState:
    plan = architect_agent(state["task"])
    implementation = developer_agent(plan)
    return {"code": implementation}

def root_synthesizer(state: TeamState) -> TeamState:
    report = editor_agent(state["research"], state["code"])
    return {"final_report": report}

graph = StateGraph(TeamState)
graph.add_node("research_team", research_subgraph_node)
graph.add_node("code_team", code_subgraph_node)
graph.add_node("synthesize", root_synthesizer)
graph.set_entry_point("research_team")
graph.add_edge("research_team", "code_team")
graph.add_edge("code_team", "synthesize")
graph.add_edge("synthesize", END)
```

### CrewAI: Nested Crews

CrewAI expresses hierarchy through **Crew** composition with `Process.hierarchical` and a `manager_agent` that coordinates sub-crews.

```python
from crewai import Agent, Task, Crew, Process

research_crew = Crew(
    agents=[search_agent, analyst_agent],
    tasks=[search_task, analyze_task],
    process=Process.sequential,
)

master_manager = Agent(
    role="Project Director",
    goal="Coordinate research and engineering crews",
    backstory="Senior technical lead who delegates and synthesizes",
    allow_delegation=True,
)

master_crew = Crew(
    agents=[master_manager],
    tasks=[coordinate_task],
    process=Process.hierarchical,
    manager_agent=master_manager,
)
```

---

## Hierarchy vs Flat Orchestration

| Aspect | Flat Orchestrator | Hierarchical |
|--------|-------------------|--------------|
| **Complexity ceiling** | ~5-8 workers | Scales to dozens |
| **Delegation depth** | Single level | Multi-level |
| **Context per agent** | Broad | Narrow, focused |
| **Debugging** | Simpler | More layers to trace |
| **Latency** | Lower | Higher (more hops) |
| **Best for** | Small teams | Large, domain-split systems |

---

## Failure Modes

**Over-nesting:** More than three tiers rarely helps. Each hop adds latency and token cost. If a manager only wraps one worker, collapse the tier.

**Unclear boundaries:** Managers must know what to keep versus delegate. Vague role prompts cause duplicate work across domains or gaps where nobody owns a subtask.

**Bottleneck managers:** If every decision routes through one manager, you recreate a single-agent problem. Give managers autonomy within their domain and pre-defined decomposition strategies.

**Lost context:** Passing raw transcripts between tiers exhausts context windows. Summarize at each boundary — managers should return compressed summaries, not full worker logs.

**Silent failures in sub-trees:** A leaf worker failure buried three levels deep may never reach the root. Propagate `success` flags upward and include partial results with clear error annotations.

**Unbounded fan-out:** Managers that spawn unlimited sub-agents per request blow up cost. Cap subtask count and worker pool size per manager.

---

## Production Notes

| Concern | Development | Production |
|---------|-------------|------------|
| Tier depth | 2-3 levels | Hard cap at 3; audit for redundant tiers |
| Manager prompts | Generic | Domain-specific with explicit delegation rules |
| Parallelism | Sequential managers | Parallel independent domain managers |
| Context passing | Full worker output | Summarized at each tier boundary |
| Observability | Print statements | Structured logs with `tier`, `agent`, `parent` fields |
| Cost control | No limits | Token budget per tier; timeout per manager |

**Tracing hierarchy:** Log every delegation with `{root → manager → worker}` path. Tools like Langfuse or LangSmith let you visualize nested spans. Without this, debugging a failed three-tier workflow is guesswork.

**Idempotent subtasks:** Leaf workers should be safe to retry. Managers should detect duplicate subtask results before synthesizing.

**Graceful degradation:** If one domain manager fails, the root orchestrator should still synthesize partial output from successful domains rather than failing entirely.

---

## Key Takeaways

- Hierarchical patterns scale multi-agent systems through nested delegation — root orchestrator, domain managers, leaf workers
- Use hierarchy when tasks decompose into domains that each need internal coordination; use flat orchestration for small linear pipelines
- LangGraph subgraphs, CrewAI hierarchical process, and AutoGen nested chats are the main framework implementations
- Limit depth to 2-3 tiers and summarize results at each boundary before passing upward
- Match manager prompts to clear delegation authority — what they own, what they delegate, and how they synthesize
- Propagate success/failure flags up the tree and cap subtask fan-out to control cost

---

## Next Lesson

**[Lesson 5: Supervisor and Router Patterns](./05-supervisor-and-router-patterns.md)** — Learn how agents classify requests and route them to the right specialist, and how supervisor loops monitor worker output for quality.
