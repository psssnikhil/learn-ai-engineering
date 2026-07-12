---
title: Agent Handoffs and Delegation
description: >-
  Master context transfer, responsibility handoffs, and delegation protocols
  between agents in LangGraph, CrewAI, and AutoGen workflows
duration: 45 min
difficulty: intermediate
has_code: true
module: module-12
---
# Agent Handoffs and Delegation

## 🎯 Learning Objectives

| What You'll Learn | Time | Difficulty |
|-------------------|------|------------|
| Understand agent handoffs vs simple message passing | 45 min | Intermediate |
| Package context for clean delegation | | |
| Implement handoff protocols in production frameworks | | |
| Prevent context loss and responsibility gaps | | |

---

## 📚 What Is an Agent Handoff?

An **agent handoff** transfers **control**, **context**, and **accountability** from one agent to another. This is more than sending a message. The receiving agent becomes the active executor, inherits relevant state, and owns the next steps until it hands off again or completes the task.

Think of a hospital shift change: the outgoing doctor does not just say "patient in room 4." They transfer the chart, highlight open issues, state what's been tried, and explicitly pass responsibility to the incoming doctor.

```
Agent A (active)                    Agent B (active)
     │                                   │
     │  1. Summarize progress            │
     │  2. Package open items            │
     │  3. Transfer control  ──────────> │
     │  4. Go idle                       │  5. Resume work
     │                                   │  6. Own next decisions
```

Microsoft's [AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners) emphasizes that handoffs are the backbone of multi-agent customer service flows. The [agents-towards-production](https://github.com/NirDiamant/agents-towards-production) repo implements handoffs with explicit state schemas so no agent operates on stale context.

---

## 🔄 Handoff vs Delegation vs Messaging

| Concept | What Transfers | Control Change |
|---------|----------------|----------------|
| **Message** | Data only | Sender may remain active |
| **Delegation** | Subtask + context | Delegator may supervise |
| **Handoff** | Full active role | Receiver becomes primary |

**Delegation** is "please handle this subtask and report back." **Handoff** is "you are now in charge of this conversation." Customer support systems hand off from triage to billing. Research pipelines delegate sub-queries but keep a lead agent in control.

---

## 📦 The Handoff Package

Every handoff should include a structured **handoff package**:

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

@dataclass
class HandoffPackage:
    from_agent: str
    to_agent: str
    task_summary: str
    completed_steps: list[str]
    open_items: list[str]
    artifacts: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_prompt(self) -> str:
        return f"""
HANDOFF from {self.from_agent} to {self.to_agent}

Task: {self.task_summary}
Completed: {', '.join(self.completed_steps)}
Open: {', '.join(self.open_items)}
Constraints: {', '.join(self.constraints)}
Artifacts: {list(self.artifacts.keys())}
"""
```

Key fields:
- **task_summary** — One-paragraph goal statement
- **completed_steps** — What not to redo
- **open_items** — Explicit next actions
- **artifacts** — Files, data, code produced so far
- **constraints** — Budget, tone, deadlines, policies

---

## 🏗️ Framework Implementations

### LangGraph: Command-Based Handoffs

LangGraph supports handoffs through graph **commands** that update state and jump to a different node. The handoff node writes the package into shared state and sets `active_agent`.

```python
from langgraph.types import Command
from typing import TypedDict

class HandoffState(TypedDict):
    active_agent: str
    handoff_log: list[dict]
    messages: list[dict]
    artifacts: dict

def triage_agent(state: HandoffState) -> Command:
    query = state["messages"][-1]["content"]
    if "legal" in query.lower():
        target = "legal_agent"
    else:
        target = "support_agent"

    package = {
        "from": "triage_agent",
        "to": target,
        "summary": f"User query classified for {target}",
        "open_items": [query],
    }
    return Command(
        goto=target,
        update={
            "active_agent": target,
            "handoff_log": state.get("handoff_log", []) + [package],
        },
    )
```

### CrewAI: allow_delegation

CrewAI agents with `allow_delegation=True` can assign subtasks to peers. The framework passes context through the manager agent. For true handoffs, define explicit tasks with `context` from prior task outputs.

```python
from crewai import Task

research_task = Task(
    description="Research {topic}",
    agent=researcher,
    expected_output="Summary with 5 key findings",
)

writing_task = Task(
    description="Write a report based on research",
    agent=writer,
    context=[research_task],  # Handoff: prior output flows in
    expected_output="500-word report",
)
```

### AutoGen: Agent Transfer

AutoGen 0.4+ supports agent-to-agent transfer where the active speaker changes and conversation history is preserved. The receiving agent sees the full thread plus a transfer notice.

---

## 💻 Handoff Manager Implementation

```python
import asyncio
from enum import Enum

class AgentRole(Enum):
    TRIAGE = "triage"
    TECHNICAL = "technical"
    BILLING = "billing"
    DONE = "done"

class HandoffManager:
    def __init__(self):
        self.active_agent = AgentRole.TRIAGE
        self.history: list[HandoffPackage] = []
        self.artifacts: dict = {}

    def handoff(self, package: HandoffPackage, new_role: AgentRole):
        self.history.append(package)
        self.artifacts.update(package.artifacts)
        self.active_agent = new_role
        print(f"🤝 Handoff: {package.from_agent} → {package.to_agent}")

    async def run(self, user_query: str) -> str:
        context = user_query

        # Triage decides handoff target
        if "refund" in user_query.lower():
            target = AgentRole.BILLING
        elif "error" in user_query.lower():
            target = AgentRole.TECHNICAL
        else:
            target = AgentRole.DONE

        if target != AgentRole.DONE:
            pkg = HandoffPackage(
                from_agent="triage",
                to_agent=target.value,
                task_summary=user_query,
                completed_steps=["classified intent"],
                open_items=[f"Resolve: {user_query}"],
            )
            self.handoff(pkg, target)
            context = pkg.to_prompt()

        return await self.execute_active(context)

    async def execute_active(self, context: str) -> str:
        role = self.active_agent.value
        await asyncio.sleep(0.2)
        return f"[{role}] Resolved with context: {context[:80]}..."
```

---

## ⚠️ Common Handoff Failures

**Context dumping:** Passing entire chat logs instead of a summary wastes tokens and confuses the receiver.

**Orphaned tasks:** Agent A hands off but nobody tracks whether Agent B finished. Always log handoffs in a shared `handoff_log`.

**Circular handoffs:** Agent A sends to B, B sends back to A indefinitely. Cap handoff count (typically 3-5).

**Missing constraints:** The billing agent does not know the user is on a free trial. Include policy constraints in every package.

---

## 💡 Best Practices

1. **Summarize, don't forward** — Compress history into the handoff package.
2. **Log every handoff** — Auditable trails are essential for debugging.
3. **Set handoff limits** — Prevent infinite loops between agents.
4. **Validate receiver readiness** — Confirm the target agent has required tools.
5. **Test handoff edges** — Most multi-agent bugs live at transfer boundaries.

---

## 🎓 Key Takeaways

```
✅ Handoffs transfer control, context, and accountability — not just data
✅ Use structured handoff packages with summary, open items, and artifacts
✅ LangGraph commands, CrewAI task context, and AutoGen transfers implement handoffs
✅ Summarize at boundaries; never dump full conversation history
✅ Log handoffs and cap transfer count to prevent loops
```

---

## 🚀 Next Lesson

**Lesson 7: Parallel vs Sequential Execution** — Decide when agents should run concurrently or in strict order.

You'll learn:
- ⚡ Parallel fan-out and fan-in patterns
- 🔗 Dependency graphs for sequencing
- 📊 Cost-latency tradeoffs
- 🛠️ LangGraph `Send` API and asyncio gathering

---

## 📚 Additional Resources

- 📄 [LangGraph Command and Handoffs](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
- 💻 [CrewAI Task Context](https://docs.crewai.com/core-concepts/Tasks/)
- 📖 [Microsoft AI Agents for Beginners — Agentic Patterns](https://github.com/microsoft/ai-agents-for-beginners)
- 🔧 [agents-towards-production handoff patterns](https://github.com/NirDiamant/agents-towards-production)

---

*⏱️ Estimated time: 45 minutes | 📊 Difficulty: Intermediate | 💻 Includes code examples*
