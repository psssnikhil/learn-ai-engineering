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

## 🎯 Learning Objectives

| What You'll Learn | Time | Difficulty |
|-------------------|------|------------|
| Understand hierarchical multi-agent architectures | 40 min | Intermediate |
| Design manager-worker and nested delegation trees | | |
| Implement hierarchical patterns with LangGraph and CrewAI | | |
| Choose when hierarchy beats flat orchestration | | |

---

## 📚 What Are Hierarchical Agent Patterns?

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

Unlike a flat orchestrator-worker setup (Lesson 3), hierarchy supports **nested delegation**. A research manager can spawn sub-agents for web search, document parsing, and citation validation without involving the root orchestrator in every decision.

---

## 🏗️ Framework Patterns

### LangGraph: Subgraphs as Hierarchy

LangGraph models hierarchy through **parent graphs** and **subgraphs**. The parent graph routes to a subgraph node; the subgraph runs its own state machine and returns consolidated output. Microsoft's [AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners) course uses this pattern for multi-agent workflows where a planner delegates to specialist subgraphs.

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

class TeamState(TypedDict):
    task: str
    research: str
    code: str
    final_report: str

def research_subgraph_node(state: TeamState) -> TeamState:
    # Subgraph: search agent → summarizer agent
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

# Parent graph delegates to subgraph nodes
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

CrewAI expresses hierarchy through **Crew** composition. A master crew's manager agent coordinates sub-crews, each with its own agents and tasks. The [agents-towards-production](https://github.com/NirDiamant/agents-towards-production) repository demonstrates this for production pipelines where a lead agent oversees domain-specific crews.

```python
from crewai import Agent, Task, Crew, Process

research_crew = Crew(
    agents=[search_agent, analyst_agent],
    tasks=[search_task, analyze_task],
    process=Process.sequential,
)

code_crew = Crew(
    agents=[architect_agent, developer_agent],
    tasks=[design_task, implement_task],
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

### AutoGen: Nested Group Chats

AutoGen supports hierarchy via **nested chats** where a group chat manager spawns child conversations for subtasks, then reports results to the parent group. This pattern suits debate-and-refine workflows at each tier.

---

## 📊 Hierarchy vs Flat Orchestration

| Aspect | Flat Orchestrator | Hierarchical |
|--------|-------------------|--------------|
| **Complexity ceiling** | ~5-8 workers | Scales to dozens |
| **Delegation depth** | Single level | Multi-level |
| **Context per agent** | Broad | Narrow, focused |
| **Debugging** | Simpler | More layers to trace |
| **Latency** | Lower | Higher (more hops) |
| **Best for** | Small teams | Large, domain-split systems |

Use hierarchy when tasks naturally decompose into **domains** (research, engineering, compliance) that each need their own internal coordination. Stick with flat orchestration when you have fewer than six specialists and a linear workflow.

---

## 💻 Implementation: Three-Tier Hierarchy

```python
from dataclasses import dataclass
from typing import Any
import asyncio

@dataclass
class TaskResult:
    agent: str
    output: Any
    status: str

class LeafWorker:
    def __init__(self, name: str, capability: str):
        self.name = name
        self.capability = capability

    async def execute(self, subtask: str) -> TaskResult:
        await asyncio.sleep(0.5)
        return TaskResult(self.name, f"[{self.capability}] {subtask}", "done")

class ManagerAgent:
    def __init__(self, name: str, workers: list[LeafWorker]):
        self.name = name
        self.workers = workers

    async def delegate(self, task: str) -> list[TaskResult]:
        print(f"  📋 {self.name} breaking down: {task}")
        subtasks = [f"{task} — part {i+1}" for i in range(len(self.workers))]
        return await asyncio.gather(
            *[w.execute(st) for w, st in zip(self.workers, subtasks)]
        )

class RootOrchestrator:
    def __init__(self, managers: dict[str, ManagerAgent]):
        self.managers = managers

    async def run(self, request: str) -> dict[str, list[TaskResult]]:
        print(f"🎯 Root received: {request}")
        results = {}
        for domain, manager in self.managers.items():
            results[domain] = await manager.delegate(request)
        return results

# Build hierarchy
research_mgr = ManagerAgent("Research Manager", [
    LeafWorker("Searcher", "web_search"),
    LeafWorker("Summarizer", "summarize"),
])
code_mgr = ManagerAgent("Code Manager", [
    LeafWorker("Architect", "design"),
    LeafWorker("Developer", "implement"),
])

orchestrator = RootOrchestrator({
    "research": research_mgr,
    "code": code_mgr,
})
```

---

## ⚠️ Common Pitfalls

**Over-nesting:** More than three tiers rarely helps. Each hop adds latency and token cost.

**Unclear boundaries:** Managers must know what to keep versus delegate. Vague role prompts cause duplicate work.

**Bottleneck managers:** If every decision routes through one manager, you recreate a single-agent problem. Give managers autonomy within their domain.

**Lost context:** Summarize results at each tier before passing upward. Raw transcripts exhaust context windows.

---

## 🎓 Key Takeaways

```
✅ Hierarchical patterns scale multi-agent systems through nested delegation
✅ LangGraph subgraphs, CrewAI hierarchical process, and AutoGen nested chats implement hierarchy
✅ Use hierarchy when domains need internal coordination beyond a flat orchestrator
✅ Limit depth to 2-3 tiers and summarize at each boundary
✅ Match manager prompts to clear delegation authority
```

---

## 🚀 Next Lesson

**Lesson 5: Supervisor and Router Patterns** — Learn how agents classify requests and route them to the right specialist.

You'll learn:
- 🧭 Intent classification and dynamic routing
- 👁️ Supervisor loops that monitor worker output
- 🔀 LangGraph conditional edges for routing
- ⚡ When to route vs when to orchestrate

---

## 📚 Additional Resources

- 📄 [LangGraph Multi-Agent Concepts](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
- 💻 [CrewAI Hierarchical Process](https://docs.crewai.com/core-concepts/Processes/)
- 📖 [Microsoft AI Agents for Beginners — Multi-Agent](https://github.com/microsoft/ai-agents-for-beginners)
- 🔧 [agents-towards-production](https://github.com/NirDiamant/agents-towards-production)

---

*⏱️ Estimated time: 40 minutes | 📊 Difficulty: Intermediate | 💻 Includes code examples*
