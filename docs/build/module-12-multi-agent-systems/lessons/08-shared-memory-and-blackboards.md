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

## What You'll Learn

| What You'll Learn | Time | Difficulty |
|-------------------|------|------------|
| Understand shared memory and blackboard architectures | 40 min | Intermediate |
| Implement safe read/write patterns for agent state | | |
| Use LangGraph state reducers and checkpointing | | |
| Choose between shared memory and message passing | | |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python 3.10+** | With `asyncio`, `dataclasses`, and typing |
| **Module 12, Lessons 2 and 7** | Communication patterns and parallel execution |
| **Module 5: AI Agents** | Stateful agents and tool use |
| **Optional: LangGraph** | For reducer and checkpointing examples |

```bash
pip install openai pydantic python-dotenv
# Optional:
pip install langgraph
```

---

## Intuition First

Picture a war room with a large whiteboard. Intelligence officers post facts on the left, analysts write hypotheses in the center, and the strategist updates the battle plan on the right. Nobody passes sticky notes hand-to-hand — everyone looks at the same board, adds what they know, and reacts to what others posted. A coordinator decides who speaks next based on what is still missing.

That whiteboard is a **blackboard**. The **shared memory** is the physical board itself. Agents (officers) read and write structured sections rather than messaging each other directly.

This pattern excels when multiple specialists iteratively refine a solution — research, analysis, writing — and need frequent access to evolving state. It struggles when agents are loosely coupled or distributed across services, where explicit message passing is cleaner.

The main risk is chaos: two agents overwriting each other's work, unbounded board size blowing up context windows, or every agent running on every small update. Production blackboards need schema, section ownership, append-only updates, and a controller that schedules who acts next.

---

## Shared Memory in Multi-Agent Systems

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

Production systems in [agents-towards-production](https://github.com/NirDiamant/agents-towards-production) use shared state for agent checkpoints and audit trails.

---

## Blackboard vs Generic Shared Memory

| Feature | Blackboard | Generic Shared Memory |
|---------|------------|----------------------|
| **Structure** | Typed sections (facts, goals, plans) | Flat key-value |
| **Triggering** | Controller selects next agent | Agents poll or subscribe |
| **Purpose** | Collaborative problem-solving | State persistence |
| **Control** | Scheduled activation | Any agent reads/writes freely |

A blackboard adds **structure and discipline**. A generic shared dict is simpler but prone to chaos without conventions.

---

## What You'll Build

By the end of this lesson, you will have:

- [ ] A typed blackboard schema with facts, hypotheses, and solution sections
- [ ] A `SafeBlackboard` with optimistic locking for concurrent writes
- [ ] A controller that selects the next agent based on board state
- [ ] Append-only updates with contributor tracking
- [ ] A runnable demo that iterates until a solution is produced

---

## Step 1: Define the Blackboard Schema

```python
# src/blackboard/schema.py
from dataclasses import dataclass, field
from typing import Any

@dataclass
class BlackboardState:
    task: str = ""
    facts: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    solution: str = ""
    contributors: list[str] = field(default_factory=list)
    version: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "facts": self.facts,
            "hypotheses": self.hypotheses,
            "solution": self.solution,
            "contributors": self.contributors,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BlackboardState":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
```

---

## Step 2: Implement Safe Concurrent Access

Shared memory introduces **race conditions**. Two agents read the same value, both modify it, and the last write wins — losing one agent's contribution.

```python
# src/blackboard/safe.py
import asyncio
from dataclasses import dataclass, field
from src.blackboard.schema import BlackboardState

class ConcurrentModificationError(Exception):
    pass

@dataclass
class VersionedBlackboard:
    state: BlackboardState = field(default_factory=BlackboardState)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def read(self) -> tuple[BlackboardState, int]:
        async with self._lock:
            return BlackboardState.from_dict(self.state.to_dict()), self.state.version

    async def write_append(
        self,
        expected_version: int,
        agent: str,
        updates: dict,
    ) -> None:
        async with self._lock:
            if self.state.version != expected_version:
                raise ConcurrentModificationError(
                    f"Version mismatch: expected {expected_version}, "
                    f"got {self.state.version}"
                )
            for key, value in updates.items():
                if key == "contributors":
                    self.state.contributors.append(agent)
                elif isinstance(getattr(self.state, key, None), list):
                    getattr(self.state, key).extend(
                        value if isinstance(value, list) else [value]
                    )
                elif hasattr(self.state, key):
                    setattr(self.state, key, value)
            self.state.version += 1
```

**Alternative strategies:**

- **Reducer-based merge (LangGraph):** Framework handles concurrent writes with defined merge logic
- **Section ownership:** Each agent writes only to its designated section — no conflicts
- **Append-only event log:** Agents append events; state is reconstructed by replay

---

## Step 3: Build Agent Functions and Controller

```python
# src/blackboard/agents.py
from src.blackboard.schema import BlackboardState

def research_agent(state: BlackboardState) -> dict:
    return {
        "facts": [f"Market size is $4.2B for: {state.task}"],
    }

def analyst_agent(state: BlackboardState) -> dict:
    facts_summary = "; ".join(state.facts[:3])
    return {
        "hypotheses": [f"Growth likely given: {facts_summary}"],
    }

def writer_agent(state: BlackboardState) -> dict:
    report = (
        f"Report for '{state.task}': "
        f"{len(state.facts)} facts, {len(state.hypotheses)} hypotheses."
    )
    return {"solution": report}
```

```python
# src/blackboard/controller.py
import asyncio
from src.blackboard.safe import VersionedBlackboard, ConcurrentModificationError
from src.blackboard.agents import research_agent, analyst_agent, writer_agent

class BlackboardController:
    AGENTS = [
        ("research_agent", research_agent, lambda s: not s.facts),
        ("analyst_agent", analyst_agent, lambda s: s.facts and not s.hypotheses),
        ("writer_agent", writer_agent, lambda s: s.hypotheses and not s.solution),
    ]

    def __init__(self, blackboard: VersionedBlackboard):
        self.blackboard = blackboard

    def select_next(self, state) -> tuple[str, callable] | None:
        for name, fn, condition in self.AGENTS:
            if condition(state):
                return name, fn
        return None

    async def run_until_complete(self, task: str, max_steps: int = 10) -> dict:
        state, version = await self.blackboard.read()
        state.task = task
        self.blackboard.state = state

        for step in range(max_steps):
            state, version = await self.blackboard.read()
            selected = self.select_next(state)
            if selected is None:
                print(f"Complete after {step} steps")
                break

            agent_name, agent_fn = selected
            print(f"Step {step + 1}: activating {agent_name}")
            updates = agent_fn(state)

            try:
                await self.blackboard.write_append(version, agent_name, updates)
            except ConcurrentModificationError:
                print(f"  Conflict on step {step + 1}, retrying...")
                continue

        final, _ = await self.blackboard.read()
        return final.to_dict()
```

---

## Step 4: Run the Demo

```python
# demo_blackboard.py
import asyncio
from src.blackboard.safe import VersionedBlackboard
from src.blackboard.controller import BlackboardController

async def main():
    board = VersionedBlackboard()
    controller = BlackboardController(board)

    result = await controller.run_until_complete(
        "AI coding tools market analysis"
    )

    print(f"\nFacts: {result['facts']}")
    print(f"Hypotheses: {result['hypotheses']}")
    print(f"Solution: {result['solution']}")
    print(f"Contributors: {result['contributors']}")
    print(f"Final version: {result['version']}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## LangGraph Shared State

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

def research_node(state: BlackboardState) -> dict:
    return {
        "facts": [f"Market size is $4.2B for: {state['task']}"],
        "contributors": ["research_agent"],
    }
```

The `Annotated[list, operator.add]` reducer appends rather than overwrites, making it safe for parallel contributors.

### Checkpointing

LangGraph persists state to a checkpointer (SQLite, Postgres). Agents can resume after failures without losing blackboard contents. This is critical for long-running multi-agent workflows.

---

## Shared Memory vs Message Passing

| Criterion | Shared Memory | Message Passing |
|-----------|---------------|-----------------|
| **Speed** | Fast reads | Serialization overhead |
| **Debugging** | Harder (implicit flow) | Easier (explicit messages) |
| **Distributed systems** | Needs external store | Natural fit |
| **Concurrent writes** | Needs coordination | Inherently isolated |
| **Audit trail** | Append logs help | Message history |

**Use shared memory** when agents need frequent access to the same evolving state (collaborative research, iterative refinement). **Use message passing** when interactions are discrete requests between loosely coupled agents.

---

## Failure Modes

**Last-write-wins races:** Two parallel agents read version 5, both write, one update is lost. Use versioning, locks, or append-only reducers.

**Unbounded board growth:** Facts and hypotheses accumulate until they exceed context limits. Summarize or prune old entries at defined thresholds.

**Schema drift:** Agents write to ad-hoc keys outside the schema. Enforce typed sections and reject unknown keys.

**Controller starvation:** One agent condition is never satisfied, looping forever. Set `max_steps` and detect stale state (no version change across N steps).

**Stale reads:** An agent plans action based on outdated board state. Pass version numbers with reads and validate before writes.

**Over-shared state:** Putting everything on one blackboard when agents are independent adds coupling. Use shared memory only where collaboration is real.

---

## Production Notes

| Concern | Development | Production |
|---------|-------------|------------|
| Storage | In-memory dict | Redis, Postgres, or LangGraph checkpointer |
| Concurrency | Single process | Distributed locks or CRDT/reducer merge |
| Board size | Unlimited | Prune/summarize at 50-80% context budget |
| Audit | Contributors list | Full event log with timestamps |
| Recovery | Restart from scratch | Checkpoint after each agent step |
| Access control | Open | Section ownership per agent role |

**Pruning strategy:** When facts exceed N entries, run a summarizer agent that replaces the list with a compressed paragraph. Keep the raw event log in storage for audit.

**External stores:** For distributed deployments, back the blackboard with Redis (fast, TTL support) or Postgres (durable, queryable). In-memory boards do not survive process restarts.

**Testing concurrent writes:** Simulate two agents writing simultaneously in tests. Verify that optimistic locking retries or reducers preserve both contributions.

---

## Key Takeaways

- Shared memory and blackboards let agents collaborate through common structured state
- Blackboards add typed sections and a controller; generic shared memory needs the same discipline manually
- LangGraph TypedDict state with reducers is production-grade shared memory for parallel contributors
- Prevent race conditions with section ownership, versioning, locks, or append-only logs
- Blackboard controllers schedule which agent acts next — preventing every agent from running on every update
- Choose shared memory for collaborative iteration; message passing for discrete, loosely coupled interactions

---

## Next Lesson

**[Lesson 9: Conflict Resolution and Consensus](./09-conflict-resolution-and-consensus.md)** — Handle disagreements when agents produce conflicting outputs using debate, voting, and judge patterns.
