---
title: Shared Memory and Blackboards
description: >-
  Implement shared state, blackboard architectures, and safe concurrent access
  for collaborating agents using LangGraph state and production patterns
duration: 40 min
difficulty: intermediate
has_code: true
module: module-12
---
# Shared Memory and Blackboards

## 🎯 Learning Objectives

| What You'll Learn | Time | Difficulty |
|-------------------|------|------------|
| Understand shared memory and blackboard architectures | 40 min | Intermediate |
| Implement safe read/write patterns for agent state | | |
| Use LangGraph state reducers and checkpointing | | |
| Choose between shared memory and message passing | | |

---

## 📚 Shared Memory in Multi-Agent Systems

In Lesson 2, we compared message passing with shared memory. Now we go deeper. **Shared memory** gives all agents access to a common data store: a database, in-memory dict, vector store, or structured state object. Agents read what others wrote and contribute their own updates.

The **blackboard architecture** is a classic AI pattern from the 1980s, still highly relevant for LLM agents. A shared "blackboard" holds hypotheses, partial solutions, and facts. Specialist agents watch the blackboard, contribute when they can add value, and react to others' contributions.

```
┌─────────────────────────────────────────────┐
│              BLACKBOARD (Shared State)       │
│  ┌─────────┐ ┌─────────┐ ┌─────────────┐   │
│  │ Facts   │ │Hypotheses│ │ Partial     │   │
│  │         │ │          │ │ Solutions   │   │
│  └─────────┘ └─────────┘ └─────────────┘   │
└───────┬──────────┬──────────┬───────────────┘
        ↓          ↓          ↓
   Agent A    Agent B    Agent C
   (reads/    (reads/    (reads/
    writes)    writes)    writes)
```

Production systems in [agents-towards-production](https://github.com/NirDiamant/agents-towards-production) use shared state for agent checkpoints and audit trails. Microsoft's [AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners) teaches stateful agent workflows as a foundation for multi-agent collaboration.

---

## 🗂️ Blackboard vs Generic Shared Memory

| Feature | Blackboard | Generic Shared Memory |
|---------|------------|----------------------|
| **Structure** | Typed sections (facts, goals, plans) | Flat key-value |
| **Triggering** | Agents activate on new postings | Agents poll or subscribe |
| **Purpose** | Collaborative problem-solving | State persistence |
| **Control** | Often a controller selects next agent | Any agent reads/writes freely |

A blackboard adds **structure and discipline**. A generic shared dict is simpler but prone to chaos without conventions.

---

## 🏗️ LangGraph Shared State

LangGraph treats shared memory as first-class. The graph `State` is a `TypedDict` that every node reads and updates. **Reducers** define how concurrent updates merge.

```python
from typing import TypedDict, Annotated
import operator

class BlackboardState(TypedDict):
    task: str
    facts: Annotated[list[str], operator.add]
    hypotheses: Annotated[list[str], operator.add]
    solution: str
    contributors: Annotated[list[str], operator.add]

def research_agent(state: BlackboardState) -> dict:
    return {
        "facts": [f"Market size is $4.2B for: {state['task']}"],
        "contributors": ["research_agent"],
    }

def analyst_agent(state: BlackboardState) -> dict:
    facts_summary = "; ".join(state["facts"][:2])
    return {
        "hypotheses": [f"Growth likely given: {facts_summary}"],
        "contributors": ["analyst_agent"],
    }

def writer_agent(state: BlackboardState) -> dict:
    report = f"Report based on {len(state['facts'])} facts, "
    report += f"{len(state['hypotheses'])} hypotheses."
    return {"solution": report, "contributors": ["writer_agent"]}
```

The `Annotated[list, operator.add]` reducer appends rather than overwrites, making it safe for parallel contributors.

### Checkpointing

LangGraph persists state to a checkpointer (SQLite, Postgres). Agents can resume after failures without losing blackboard contents. This is critical for long-running multi-agent workflows.

---

## 🔒 Safe Concurrent Access

Shared memory introduces **race conditions**. Two agents read the same value, both modify it, and the last write wins — losing one agent's contribution.

### Strategies

**1. Reducer-based merge (LangGraph):** Framework handles concurrent writes with defined merge logic.

**2. Optimistic locking:** Attach a version number. Writes fail if version changed since read.

```python
import asyncio
from dataclasses import dataclass, field

@dataclass
class VersionedState:
    data: dict = field(default_factory=dict)
    version: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

class SafeBlackboard:
    def __init__(self):
        self.state = VersionedState()

    async def read(self, key: str, expected_version: int | None = None):
        async with self.state._lock:
            if expected_version and self.state.version != expected_version:
                raise ConcurrentModificationError("State changed since read")
            return self.state.data.get(key), self.state.version

    async def write(self, key: str, value, expected_version: int):
        async with self.state._lock:
            if self.state.version != expected_version:
                raise ConcurrentModificationError("Write conflict")
            self.state.data[key] = value
            self.state.version += 1
```

**3. Section ownership:** Each agent owns specific blackboard sections. Research agent writes only to `facts`. Analyst writes only to `hypotheses`. No conflicts if sections do not overlap.

**4. Append-only logs:** Instead of mutating values, append events. Agents replay the log to reconstruct current state.

---

## 💻 Blackboard Controller

A **controller** (or scheduler) decides which agent runs next based on blackboard contents. This prevents every agent from running on every update.

```python
from enum import Enum

class BoardSection(Enum):
    FACTS = "facts"
    HYPOTHESES = "hypotheses"
    SOLUTION = "solution"

class BlackboardController:
    def __init__(self, blackboard: SafeBlackboard):
        self.blackboard = blackboard
        self.agents = {
            BoardSection.FACTS: research_agent,
            BoardSection.HYPOTHESES: analyst_agent,
            BoardSection.SOLUTION: writer_agent,
        }

    async def select_next_agent(self, state: dict) -> callable | None:
        if not state.get("facts"):
            return self.agents[BoardSection.FACTS]
        if not state.get("hypotheses"):
            return self.agents[BoardSection.HYPOTHESES]
        if not state.get("solution"):
            return self.agents[BoardSection.SOLUTION]
        return None

    async def run_until_complete(self, task: str, max_steps: int = 10):
        state = {"task": task, "facts": [], "hypotheses": [], "solution": ""}

        for _ in range(max_steps):
            agent_fn = await self.select_next_agent(state)
            if agent_fn is None:
                break
            update = agent_fn(state)
            for key, value in update.items():
                if isinstance(value, list):
                    state.setdefault(key, []).extend(value)
                else:
                    state[key] = value

        return state
```

---

## 📊 Shared Memory vs Message Passing

| Criterion | Shared Memory | Message Passing |
|-----------|---------------|-----------------|
| **Speed** | Fast reads | Serialization overhead |
| **Debugging** | Harder (implicit flow) | Easier (explicit messages) |
| **Distributed systems** | Needs external store | Natural fit |
| **Concurrent writes** | Needs coordination | Inherently isolated |
| **Audit trail** | Append logs help | Message history |

**Use shared memory** when agents need frequent access to the same evolving state (collaborative research, iterative refinement). **Use message passing** when interactions are discrete requests between loosely coupled agents.

---

## 💡 Best Practices

1. **Define schema upfront** — Typed sections prevent agents from writing to random keys.
2. **Assign section ownership** — Minimize write conflicts.
3. **Use append-only updates** — Reducers and event logs preserve history.
4. **Checkpoint state** — Enable recovery in long workflows.
5. **Cap blackboard size** — Summarize or prune old entries to manage context windows.
6. **Log who wrote what** — The `contributors` field enables accountability.

---

## 🎓 Key Takeaways

```
✅ Shared memory and blackboards let agents collaborate through common state
✅ LangGraph TypedDict state with reducers is production-grade shared memory
✅ Prevent race conditions with section ownership, versioning, or append-only logs
✅ Blackboard controllers schedule which agent acts next
✅ Choose shared memory for collaborative iteration; messages for discrete requests
```

---

## 🚀 Next Lesson

**Lesson 9: Conflict Resolution and Consensus** — Handle disagreements when agents produce conflicting outputs.

You'll learn:
- ⚖️ Voting, debate, and judge patterns
- 🔄 Merge strategies for conflicting agent results
- 🏛️ AutoGen group debate and CrewAI review loops
- 📏 When consensus is worth the extra cost

---

## 📚 Additional Resources

- 📄 [LangGraph State Management](https://langchain-ai.github.io/langgraph/concepts/low_level/#state)
- 💻 [CrewAI Memory and Context](https://docs.crewai.com/core-concepts/Memory/)
- 📖 [Microsoft AI Agents for Beginners — State Management](https://github.com/microsoft/ai-agents-for-beginners)
- 🔧 [agents-towards-production state patterns](https://github.com/NirDiamant/agents-towards-production)

---

*⏱️ Estimated time: 40 minutes | 📊 Difficulty: Intermediate | 💻 Includes code examples*
