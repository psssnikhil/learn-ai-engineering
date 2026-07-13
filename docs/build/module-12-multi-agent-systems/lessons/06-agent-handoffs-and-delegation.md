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

## What You'll Learn

| What You'll Learn | Time | Difficulty |
|-------------------|------|------------|
| Understand agent handoffs vs simple message passing | 45 min | Intermediate |
| Package context for clean delegation | | |
| Implement handoff protocols in production frameworks | | |
| Prevent context loss and responsibility gaps | | |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python 3.10+** | With `asyncio`, `dataclasses`, and typing |
| **Module 12, Lessons 3-5** | Orchestrator, hierarchy, and routing patterns |
| **Module 5: AI Agents** | Agent roles, system prompts, tool use |
| **Optional: LangGraph** | For command-based handoff examples |

```bash
pip install openai pydantic python-dotenv
# Optional:
pip install langgraph
```

---

## Intuition First

Think of a hospital shift change. The outgoing doctor does not just say "patient in room 4." They transfer the chart, highlight open issues, state what has been tried, note allergies and constraints, and explicitly pass responsibility to the incoming doctor. The incoming doctor becomes the **active decision-maker** until the next handoff.

In multi-agent systems, the same discipline applies. Sending a message is like leaving a note — the sender may still be in charge. **Delegation** is asking a colleague to handle a subtask and report back while you supervise. **Handoff** is transferring full control: the receiving agent owns the conversation until it completes or hands off again.

Customer support flows demonstrate this clearly. A triage bot classifies intent, packages context, and hands off to billing or technical support. The billing agent does not re-ask questions the user already answered. The handoff package carries forward everything the next agent needs — and nothing they do not.

Getting handoffs wrong is the most common source of multi-agent bugs: duplicated work, lost constraints, infinite loops between agents, and orphaned tasks nobody tracks to completion.

---

## What Is an Agent Handoff?

An **agent handoff** transfers **control**, **context**, and **accountability** from one agent to another. This is more than sending a message. The receiving agent becomes the active executor, inherits relevant state, and owns the next steps until it hands off again or completes the task.

```
Agent A (active)                    Agent B (active)
     │                                   │
     │  1. Summarize progress            │
     │  2. Package open items            │
     │  3. Transfer control  ──────────> │
     │  4. Go idle                       │  5. Resume work
     │                                   │  6. Own next decisions
```

Microsoft's [AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners) emphasizes that handoffs are the backbone of multi-agent customer service flows. Production repos like [agents-towards-production](https://github.com/NirDiamant/agents-towards-production) implement handoffs with explicit state schemas so no agent operates on stale context.

---

## Handoff vs Delegation vs Messaging

| Concept | What Transfers | Control Change |
|---------|----------------|----------------|
| **Message** | Data only | Sender may remain active |
| **Delegation** | Subtask + context | Delegator may supervise |
| **Handoff** | Full active role | Receiver becomes primary |

**Delegation** is "please handle this subtask and report back." **Handoff** is "you are now in charge of this conversation." Research pipelines often delegate sub-queries but keep a lead agent in control. Support systems hand off from triage to billing with full control transfer.

---

## What You'll Build

By the end of this lesson, you will have:

- [ ] A structured `HandoffPackage` dataclass for context transfer
- [ ] A `HandoffManager` that routes queries and tracks active agent
- [ ] Handoff logging with audit trail and loop prevention
- [ ] LLM-powered triage that produces handoff packages
- [ ] A runnable support-flow demo with triage, technical, and billing agents

---

## Step 1: Define the Handoff Package

Every handoff should include a structured **handoff package** — never raw chat logs.

```python
# src/handoffs/package.py
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
    handoff_count: int = 0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_prompt(self) -> str:
        return f"""HANDOFF from {self.from_agent} to {self.to_agent}
Timestamp: {self.timestamp}

Task: {self.task_summary}

Completed steps:
{chr(10).join(f'  - {s}' for s in self.completed_steps)}

Open items:
{chr(10).join(f'  - {s}' for s in self.open_items)}

Constraints:
{chr(10).join(f'  - {c}' for c in self.constraints)}

Available artifacts: {list(self.artifacts.keys())}
"""

    def validate(self) -> list[str]:
        errors = []
        if not self.task_summary.strip():
            errors.append("task_summary is required")
        if not self.open_items:
            errors.append("open_items must list at least one next action")
        if not self.to_agent:
            errors.append("to_agent is required")
        return errors
```

Key fields:

- **task_summary** — One-paragraph goal statement
- **completed_steps** — What not to redo
- **open_items** — Explicit next actions for the receiving agent
- **artifacts** — Files, data, code produced so far
- **constraints** — Budget, tone, deadlines, policies (e.g., "user is on free trial")

---

## Step 2: Build the Handoff Manager

The manager tracks active agent, enforces handoff limits, and logs every transfer.

```python
# src/handoffs/manager.py
import asyncio
from enum import Enum
from dataclasses import dataclass, field
from src/handoffs.package import HandoffPackage

MAX_HANDOFFS = 5

class AgentRole(Enum):
    TRIAGE = "triage"
    TECHNICAL = "technical"
    BILLING = "billing"
    DONE = "done"

@dataclass
class HandoffState:
    active_agent: AgentRole = AgentRole.TRIAGE
    handoff_log: list[HandoffPackage] = field(default_factory=list)
    artifacts: dict = field(default_factory=dict)
    handoff_count: int = 0

class HandoffManager:
    def __init__(self, max_handoffs: int = MAX_HANDOFFS):
        self.state = HandoffState()
        self.max_handoffs = max_handoffs

    def classify_intent(self, query: str) -> AgentRole:
        q = query.lower()
        if any(w in q for w in ("refund", "invoice", "payment", "charge")):
            return AgentRole.BILLING
        if any(w in q for w in ("error", "bug", "crash", "500", "login")):
            return AgentRole.TECHNICAL
        if any(w in q for w in ("hello", "thanks", "bye")):
            return AgentRole.DONE
        return AgentRole.TECHNICAL  # default fallback

    def handoff(self, package: HandoffPackage, new_role: AgentRole) -> None:
        errors = package.validate()
        if errors:
            raise ValueError(f"Invalid handoff package: {errors}")

        if self.state.handoff_count >= self.max_handoffs:
            raise RuntimeError(
                f"Handoff limit reached ({self.max_handoffs}). Possible loop."
            )

        self.state.handoff_log.append(package)
        self.state.artifacts.update(package.artifacts)
        self.state.active_agent = new_role
        self.state.handoff_count += 1
        print(f"Handoff #{self.state.handoff_count}: "
              f"{package.from_agent} -> {package.to_agent}")

    async def run(self, user_query: str) -> dict:
        target = self.classify_intent(user_query)

        if target == AgentRole.DONE:
            return {"agent": "triage", "response": "Happy to help! Anything else?"}

        pkg = HandoffPackage(
            from_agent="triage",
            to_agent=target.value,
            task_summary=user_query,
            completed_steps=["Received user query", "Classified intent"],
            open_items=[f"Resolve: {user_query}"],
            constraints=["Be concise", "Do not ask for info already provided"],
            artifacts={"original_query": user_query},
            handoff_count=self.state.handoff_count + 1,
        )
        self.handoff(pkg, target)

        response = await self.execute_active(pkg.to_prompt())
        return {
            "agent": self.state.active_agent.value,
            "response": response,
            "handoffs": len(self.state.handoff_log),
            "log": [
                {"from": h.from_agent, "to": h.to_agent, "time": h.timestamp}
                for h in self.state.handoff_log
            ],
        }

    async def execute_active(self, context: str) -> str:
        role = self.state.active_agent.value
        await asyncio.sleep(0.2)  # Replace with LLM call
        return f"[{role}] Handled request using packaged context."
```

---

## Step 3: Add LLM-Powered Agents

Replace mock execution with real agents that consume the handoff prompt.

```python
# src/handoffs/agents.py
from openai import OpenAI

client = OpenAI()

AGENT_PROMPTS = {
    "triage": "You classify support requests and prepare handoff packages.",
    "technical": "You are a technical support specialist. Use the handoff context. "
                 "Do not re-ask questions already answered.",
    "billing": "You are a billing specialist. Check constraints for plan/trial info. "
               "Process refund and invoice requests.",
}

async def run_agent(role: str, context: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": AGENT_PROMPTS[role]},
            {"role": "user", "content": context},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content
```

---

## Step 4: Run the Demo

```python
# demo_handoffs.py
import asyncio
from src.handoffs.manager import HandoffManager

async def main():
    manager = HandoffManager(max_handoffs=5)

    queries = [
        "I was charged twice for my subscription this month",
        "The app crashes when I click login with Google",
        "Hello, just checking in",
    ]

    for query in queries:
        print(f"\nUser: {query}")
        result = await manager.run(query)
        print(f"Agent: {result['agent']}")
        print(f"Response: {result['response']}")
        print(f"Handoffs so far: {result['handoffs']}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Framework Implementations

### LangGraph: Command-Based Handoffs

LangGraph supports handoffs through graph **commands** that update state and jump to a different node.

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
    target = "legal_agent" if "legal" in query.lower() else "support_agent"

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

### CrewAI: Task Context Chains

CrewAI passes prior task output through `context` arrays — a form of delegation handoff.

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
    context=[research_task],
    expected_output="500-word report",
)
```

### AutoGen: Agent Transfer

AutoGen 0.4+ supports agent-to-agent transfer where the active speaker changes and conversation history is preserved, with an explicit transfer notice prepended for the receiving agent.

---

## Failure Modes

**Context dumping:** Passing entire chat logs instead of a summary wastes tokens and confuses the receiver. Always compress history into the handoff package.

**Orphaned tasks:** Agent A hands off but nobody tracks whether Agent B finished. Log every handoff in a shared `handoff_log` and require completion acknowledgments.

**Circular handoffs:** Agent A sends to B, B sends back to A indefinitely. Cap handoff count (typically 3-5) and detect repeated `(from, to)` pairs.

**Missing constraints:** The billing agent does not know the user is on a free trial. Include policy constraints in every package — they are not optional metadata.

**Stale artifacts:** Handoff references files or data that were updated after the package was created. Include timestamps and version numbers on artifacts.

**Validation skipped:** Empty `open_items` or missing `task_summary` cause the receiving agent to guess next steps. Validate packages before transfer.

---

## Production Notes

| Concern | Development | Production |
|---------|-------------|------------|
| Handoff format | Ad-hoc strings | Validated schema (Pydantic/dataclass) |
| Loop prevention | Manual | Hard cap + repeated-pair detection |
| Audit trail | Print log | Persistent store (DB or event log) |
| Agent readiness | Assume available | Check target agent has required tools |
| Escalation | None | Human handoff when limit reached |
| Testing | Happy path | Test every handoff edge in routing matrix |

**Human escalation:** When `handoff_count` exceeds the cap, route to a human operator with the full handoff log — not another agent.

**Idempotent handoffs:** If a handoff fails mid-transfer, the system should be able to retry with the same package without duplicating work. Store packages before activating the receiving agent.

**Metrics to track:** Handoffs per conversation, time-to-first-response after handoff, re-handoff rate (indicates misrouting), and constraint-violation rate.

---

## Key Takeaways

- Handoffs transfer control, context, and accountability — not just data
- Use structured handoff packages with summary, completed steps, open items, artifacts, and constraints
- Delegation keeps a supervisor in control; handoff makes the receiver the primary active agent
- LangGraph commands, CrewAI task context, and AutoGen transfers are the main framework implementations
- Summarize at boundaries; never dump full conversation history
- Log every handoff, validate packages, and cap transfer count to prevent loops

---

## Next Lesson

**[Lesson 7: Parallel vs Sequential Execution](./07-parallel-vs-sequential-execution.md)** — Decide when agents should run concurrently or in strict order, and implement fan-out/fan-in patterns for optimal latency and cost.
