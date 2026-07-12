---
title: Building a Multi-Agent System
description: >-
  Assemble a complete multi-agent system end-to-end — architecture, implementation,
  testing, and deployment using LangGraph, CrewAI, and production best practices
duration: 50 min
difficulty: intermediate
has_code: true
module: module-12
---
# Building a Multi-Agent System

## 🎯 Learning Objectives

| What You'll Learn | Time | Difficulty |
|-------------------|------|------------|
| Design a complete multi-agent architecture from requirements | 50 min | Intermediate |
| Assemble patterns from Lessons 1-9 into one system | | |
| Test, observe, and debug multi-agent workflows | | |
| Deploy with a production readiness checklist | | |

---

## 📚 From Patterns to Production

Throughout this module, you learned individual patterns: orchestration, communication, hierarchy, routing, handoffs, parallel execution, shared memory, and consensus. This lesson **assembles them** into a deployable system.

We will build a **Research & Report Pipeline** — a common production use case featured in [agents-towards-production](https://github.com/NirDiamant/agents-towards-production) and Microsoft's [AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners). The system researches a topic, analyzes findings, writes a report, and has a reviewer approve or send revisions.

### Target Architecture

```
User Query
    ↓
┌──────────┐
│  Router  │ ──→ off-topic → General Agent → END
└────┬─────┘
     ↓ (research topic)
┌──────────────┐
│ Orchestrator │
└──────┬───────┘
       ↓
  ┌────┴────┐ (parallel)
  ↓         ↓
Web      Doc
Search   Parser
  └────┬────┘
       ↓ (fan-in to blackboard)
┌──────────────┐
│  Blackboard  │ ← shared state
└──────┬───────┘
       ↓
┌──────────────┐
│   Analyst    │
└──────┬───────┘
       ↓
┌──────────────┐
│   Writer     │
└──────┬───────┘
       ↓
┌──────────────┐
│   Reviewer   │ ──→ reject → Writer (loop, max 2)
└──────┬───────┘
       ↓ (approved)
     Output
```

---

## 🏗️ Step 1: Define Requirements

Before writing code, answer these questions:

| Question | Our Answer |
|----------|------------|
| **What is the input?** | User research topic (string) |
| **What is the output?** | Approved markdown report |
| **Which agents are needed?** | Router, researchers, analyst, writer, reviewer |
| **Sequential or parallel?** | Parallel research, sequential analysis → write → review |
| **Shared state or messages?** | Blackboard for research facts |
| **Conflict resolution?** | Reviewer loop (judge pattern) |
| **Failure handling?** | Retries on research agents, max 2 review cycles |

Document agent roles in a config file. Every agent gets a precise `role`, `goal`, `tools`, and `constraints`.

---

## 💻 Step 2: Core Implementation

### State Schema

```python
from typing import TypedDict, Annotated
import operator

class PipelineState(TypedDict):
    query: str
    route: str
    facts: Annotated[list[str], operator.add]
    analysis: str
    draft: str
    final_report: str
    review_feedback: str
    review_count: int
    approved: bool
    trace: Annotated[list[str], operator.add]
```

### Agent Nodes

```python
async def router_node(state: PipelineState) -> dict:
    query = state["query"].lower()
    if any(w in query for w in ["weather", "joke", "hello"]):
        route = "general"
    else:
        route = "research"
    return {"route": route, "trace": [f"router → {route}"]}

async def web_research_node(state: PipelineState) -> dict:
    facts = [f"[web] Finding about {state['query']} from source A"]
    return {"facts": facts, "trace": ["web_research complete"]}

async def doc_research_node(state: PipelineState) -> dict:
    facts = [f"[doc] Finding about {state['query']} from source B"]
    return {"facts": facts, "trace": ["doc_research complete"]}

async def analyst_node(state: PipelineState) -> dict:
    summary = f"Analysis of {len(state['facts'])} facts on {state['query']}"
    return {"analysis": summary, "trace": ["analyst complete"]}

async def writer_node(state: PipelineState) -> dict:
    feedback = state.get("review_feedback", "")
    draft = f"Report on {state['query']}: {state['analysis']}"
    if feedback:
        draft += f" (revised per: {feedback})"
    return {"draft": draft, "trace": ["writer complete"]}

async def reviewer_node(state: PipelineState) -> dict:
    count = state.get("review_count", 0) + 1
    draft = state["draft"]

    if len(draft) > 30 and count > 1:
        return {
            "approved": True,
            "final_report": draft,
            "review_count": count,
            "trace": ["reviewer approved"],
        }
    return {
        "approved": False,
        "review_feedback": "Add more detail and depth.",
        "review_count": count,
        "trace": [f"reviewer rejected (round {count})"],
    }
```

### Graph Assembly (LangGraph)

```python
from langgraph.graph import StateGraph, END

def build_pipeline():
    graph = StateGraph(PipelineState)

    graph.add_node("router", router_node)
    graph.add_node("web_research", web_research_node)
    graph.add_node("doc_research", doc_research_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviewer", reviewer_node)

    graph.set_entry_point("router")

    def route_decision(state):
        return "general" if state["route"] == "general" else "research"

    graph.add_conditional_edges("router", route_decision, {
        "general": END,
        "research": "web_research",
    })

    # Parallel research: both run, then analyst
    graph.add_edge("web_research", "analyst")
    graph.add_edge("doc_research", "analyst")
    # Entry also triggers doc research
    graph.add_edge("router", "doc_research")  # simplified; use Send in production

    graph.add_edge("analyst", "writer")
    graph.add_edge("writer", "reviewer")

    def review_decision(state):
        if state["approved"] or state.get("review_count", 0) >= 2:
            return "done"
        return "revise"

    graph.add_conditional_edges("reviewer", review_decision, {
        "done": END,
        "revise": "writer",
    })

    return graph.compile()
```

### CrewAI Alternative

For teams preferring CrewAI, define agents and tasks matching the same flow. Use `Process.sequential` for the main pipeline and independent tasks without `context` for parallel research. The reviewer task feeds back via a revision task with `context=[draft_task, review_task]`.

---

## 🧪 Step 3: Testing

Multi-agent systems need **layered tests**:

**Unit tests** — Each agent node in isolation with mock state.

```python
import asyncio

async def test_reviewer_rejects_short_draft():
    state = {"draft": "short", "review_count": 0}
    result = await reviewer_node(state)
    assert result["approved"] is False
    assert "review_feedback" in result

asyncio.run(test_reviewer_rejects_short_draft())
```

**Integration tests** — Full pipeline with recorded LLM responses (use mocks or VCR cassettes).

**Trace assertions** — Verify the `trace` field shows expected agent order: router → research → analyst → writer → reviewer.

**Chaos tests** — Kill one research agent and verify the pipeline still completes with partial facts.

---

## 📊 Step 4: Observability

Production multi-agent systems require visibility:

| Signal | What to Track |
|--------|---------------|
| **Trace spans** | Per-agent latency, token usage |
| **Handoff log** | Agent transitions and packages |
| **Blackboard snapshots** | State at each step |
| **Review cycles** | How often reviewers reject |
| **Cost per run** | Total tokens across all agents |

LangGraph integrates with LangSmith for tracing. Log the `trace` list from state as a lightweight alternative. Structure logs as JSON for queryability.

```python
import logging
import json

logger = logging.getLogger("multi_agent")

def log_step(state: PipelineState, step: str):
    logger.info(json.dumps({
        "step": step,
        "query": state.get("query"),
        "facts_count": len(state.get("facts", [])),
        "approved": state.get("approved"),
        "trace": state.get("trace", []),
    }))
```

---

## 🚀 Step 5: Deployment Checklist

```
Architecture
  ✅ Agent roles documented with clear boundaries
  ✅ Execution plan (parallel vs sequential) defined
  ✅ Handoff and state schemas specified
  ✅ Max iteration limits on all loops

Reliability
  ✅ Retries on transient failures
  ✅ Graceful degradation (partial research results)
  ✅ Checkpointing for long workflows
  ✅ Timeout per agent call

Security
  ✅ Input validation on user queries
  ✅ Tool access scoped per agent
  ✅ Output filtering before returning to user

Operations
  ✅ Structured logging and tracing
  ✅ Cost monitoring and alerts
  ✅ A/B testing for prompt changes
  ✅ Runbook for debugging failed traces
```

---

## 📋 Pattern Map: What We Used

| Lesson | Pattern Applied |
|--------|-----------------|
| Lesson 1 | Multi-agent vs single-agent decision |
| Lesson 2 | Blackboard as shared memory |
| Lesson 3 | Orchestrator coordinates workflow |
| Lesson 4 | Research sub-teams as hierarchy |
| Lesson 5 | Router classifies; reviewer supervises |
| Lesson 6 | Review feedback as handoff package |
| Lesson 7 | Parallel web + doc research |
| Lesson 8 | Blackboard state with reducers |
| Lesson 9 | Reviewer judge pattern with retry loop |

---

## 🎓 Key Takeaways

```
✅ Start with requirements: input, output, agents, execution order, failure policy
✅ Combine patterns — no single architecture fits every task
✅ LangGraph graphs and CrewAI crews both assemble patterns into pipelines
✅ Test agents individually, then integration, then chaos scenarios
✅ Observability and iteration limits are non-negotiable for production
✅ Use the deployment checklist before going live
```

---

## 🚀 Module Complete

Congratulations! You have completed **Module 12: Multi-Agent Systems**. You can now:

- 🏗️ Choose the right multi-agent architecture for your use case
- 🔄 Implement communication, orchestration, and delegation
- ⚡ Optimize parallel and sequential execution
- ⚖️ Resolve conflicts and deploy with confidence

**Continue your journey** — Apply these patterns to your own projects and reference the [agents-towards-production](https://github.com/NirDiamant/agents-towards-production) repo for production-ready examples.

---

## 📚 Additional Resources

- 📄 [LangGraph Full Tutorial](https://langchain-ai.github.io/langgraph/tutorials/)
- 💻 [CrewAI Getting Started](https://docs.crewai.com/introduction)
- 📖 [Microsoft AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners)
- 🔧 [agents-towards-production](https://github.com/NirDiamant/agents-towards-production)
- 📺 [Anthropic: Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)

---

*⏱️ Estimated time: 50 minutes | 📊 Difficulty: Intermediate | 💻 Includes code examples*
