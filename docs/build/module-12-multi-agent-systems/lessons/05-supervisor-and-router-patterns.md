---
title: Supervisor and Router Patterns
description: >-
  Build agents that classify intent, route tasks to specialists, and supervise
  worker output using LangGraph supervisors and router architectures
duration: 40 min
difficulty: intermediate
has_code: true
module: module-12
---
# Supervisor and Router Patterns

## 🎯 Learning Objectives

| What You'll Learn | Time | Difficulty |
|-------------------|------|------------|
| Distinguish supervisor and router architectures | 40 min | Intermediate |
| Implement intent-based routing to specialist agents | | |
| Build supervisor loops that review and redirect work | | |
| Apply patterns from LangGraph, CrewAI, and AutoGen | | |

---

## 📚 Supervisor vs Router: Two Sides of Control

Multi-agent systems need something to decide **who does what**. Two closely related patterns handle this:

**Router:** Classifies the incoming request and sends it to exactly one specialist. Think of a customer support triage bot that picks billing, technical, or sales agents. Routing is typically a **one-shot decision** at the start of a workflow.

**Supervisor:** An ongoing overseer that assigns work, reviews results, and decides whether to accept output, retry, or reassign. The supervisor runs in a **loop** until the task meets quality criteria.

```
Router Pattern:                    Supervisor Pattern:

  Request                            Request
     ↓                                  ↓
 ┌─────────┐                      ┌────────────┐
 │ Router  │                      │ Supervisor │
 └────┬────┘                      └─────┬──────┘
      │ (one decision)                    │ (loop)
 ┌────┼────┐                         ┌────┴────┐
 ↓    ↓    ↓                         ↓         ↓
Spec Spec Spec                     Worker    Review
 A    B    C                         ↑         │
                                      └─ retry ─┘
```

Microsoft's [AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners) module on agent design patterns treats routing as the entry point and supervision as the quality gate. Production repos like [agents-towards-production](https://github.com/NirDiamant/agents-towards-production) combine both: route first, then supervise within the chosen domain.

---

## 🧭 Router Pattern in Practice

### When to Use Routers

| Scenario | Router Fit |
|----------|------------|
| Mutually exclusive tasks | ✅ Strong |
| Specialist pools with clear domains | ✅ Strong |
| Sequential pipelines | ❌ Use orchestrator |
| Tasks needing iteration | ❌ Add supervisor |

Routers minimize token spend by activating only the relevant agent. A billing question never invokes the code-generation agent.

### LangGraph Conditional Routing

LangGraph uses **conditional edges** to implement routers. A routing function inspects state and returns the next node name.

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

class RouterState(TypedDict):
    query: str
    route: str
    response: str

def classify_intent(state: RouterState) -> RouterState:
    query = state["query"].lower()
    if "invoice" in query or "payment" in query:
        route = "billing"
    elif "error" in query or "bug" in query:
        route = "technical"
    else:
        route = "general"
    return {"route": route}

def route_decision(state: RouterState) -> Literal["billing", "technical", "general"]:
    return state["route"]

def billing_agent(state: RouterState) -> RouterState:
    return {"response": f"Billing help for: {state['query']}"}

def technical_agent(state: RouterState) -> RouterState:
    return {"response": f"Tech support for: {state['query']}"}

def general_agent(state: RouterState) -> RouterState:
    return {"response": f"General answer for: {state['query']}"}

graph = StateGraph(RouterState)
graph.add_node("classify", classify_intent)
graph.add_node("billing", billing_agent)
graph.add_node("technical", technical_agent)
graph.add_node("general", general_agent)

graph.set_entry_point("classify")
graph.add_conditional_edges("classify", route_decision, {
    "billing": "billing",
    "technical": "technical",
    "general": "general",
})
for node in ["billing", "technical", "general"]:
    graph.add_edge(node, END)
```

### CrewAI Task Routing

In CrewAI, routing often appears as a **manager agent** that assigns tasks to crew members based on role descriptions. The manager reads each agent's `role` and `goal` fields to pick the best match.

---

## 👁️ Supervisor Pattern in Practice

Supervisors add a **feedback loop**. After a worker completes a task, the supervisor evaluates quality and either approves or sends revision instructions.

### LangGraph Supervisor Architecture

LangGraph's documented supervisor pattern places a central LLM node that calls worker nodes as tools. The supervisor decides which worker to invoke next and when the job is done.

```python
from typing import TypedDict, Annotated
import operator

class SupervisorState(TypedDict):
    messages: Annotated[list, operator.add]
    next_agent: str
    iteration: int
    approved: bool

def supervisor_node(state: SupervisorState) -> SupervisorState:
    last_output = state["messages"][-1]["content"]
    iteration = state.get("iteration", 0)

    if "APPROVED" in last_output or iteration >= 3:
        return {"approved": True, "next_agent": "FINISH"}

    if iteration == 0:
        return {"next_agent": "researcher", "iteration": iteration + 1}
    if "incomplete" in last_output.lower():
        return {"next_agent": "researcher", "iteration": iteration + 1}
    return {"next_agent": "writer", "iteration": iteration + 1}

def worker_node(state: SupervisorState) -> SupervisorState:
    agent = state["next_agent"]
    result = f"[{agent}] processed: {state['messages'][0]['content']}"
    return {"messages": [{"role": "assistant", "content": result}]}
```

### AutoGen Group Chat Manager

AutoGen's `GroupChatManager` is a classic supervisor. It reads the conversation history, selects the next speaker, and terminates when the manager decides the task is complete. This maps directly to human team standups where a lead assigns the next speaker.

---

## 📊 Router vs Supervisor Comparison

| Dimension | Router | Supervisor |
|-----------|--------|------------|
| **Decision frequency** | Once at entry | Repeated in loop |
| **Token cost** | Lower | Higher |
| **Quality control** | Minimal | Strong |
| **Latency** | Fast | Slower |
| **Failure recovery** | Weak | Retries, reassignment |
| **Implementation** | Conditional edges | Loop with exit condition |

**Combined pattern:** Route to the right domain, then supervise within that domain. A support system routes to the technical agent, and a QA supervisor reviews the technical response before sending it to the user.

---

## 💻 Combined Router + Supervisor

```python
import asyncio
from enum import Enum

class Domain(Enum):
    BILLING = "billing"
    TECHNICAL = "technical"

class Router:
    def classify(self, query: str) -> Domain:
        if any(w in query.lower() for w in ["invoice", "refund", "charge"]):
            return Domain.BILLING
        return Domain.TECHNICAL

class Supervisor:
    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries

    async def review(self, output: str, attempt: int) -> tuple[bool, str]:
        if len(output) > 50 and "error" not in output.lower():
            return True, output
        if attempt >= self.max_retries:
            return True, output + " [escalated]"
        return False, "Please add more detail and remove errors."

class Specialist:
    def __init__(self, domain: Domain):
        self.domain = domain

    async def handle(self, query: str, feedback: str = "") -> str:
        await asyncio.sleep(0.3)
        prefix = f"[{self.domain.value}]"
        if feedback:
            return f"{prefix} Revised answer for '{query}' per: {feedback}"
        return f"{prefix} Draft answer for '{query}'"

async def handle_request(query: str) -> str:
    domain = Router().classify(query)
    worker = Specialist(domain)
    supervisor = Supervisor()

    feedback = ""
    for attempt in range(3):
        output = await worker.handle(query, feedback)
        approved, result = await supervisor.review(output, attempt)
        if approved:
            return result
        feedback = result
    return output
```

---

## 💡 Best Practices

1. **Keep routing rules explicit** — Log why each route was chosen for debugging.
2. **Cap supervisor loops** — Set a maximum iteration count to prevent runaway costs.
3. **Use structured output** — Have the supervisor return JSON with `approved`, `feedback`, and `next_agent` fields.
4. **Separate routing from execution** — The router should not also write the final answer.
5. **Measure misroutes** — Track how often users escalate or rephrase after routing.

---

## 🎓 Key Takeaways

```
✅ Routers classify once and dispatch to the right specialist
✅ Supervisors loop until output meets quality standards
✅ LangGraph conditional edges implement routers; supervisor nodes implement review loops
✅ Combine both: route to domain, supervise within domain
✅ Always cap iterations and log routing decisions
```

---

## 🚀 Next Lesson

**Lesson 6: Agent Handoffs and Delegation** — Master how agents transfer control, context, and responsibility mid-workflow.

You'll learn:
- 🤝 Handoff protocols between agents
- 📦 Context packaging for clean transfers
- 🔄 LangGraph `Command` and AutoGen agent transfers
- ⚠️ Avoiding context loss during delegation

---

## 📚 Additional Resources

- 📄 [LangGraph Supervisor Tutorial](https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/)
- 💻 [CrewAI Manager Agent](https://docs.crewai.com/core-concepts/Agents/)
- 📖 [AutoGen Group Chat](https://microsoft.github.io/autogen/docs/user-guide/core-user-guide/design-patterns/group-chat)
- 🔧 [agents-towards-production routing examples](https://github.com/NirDiamant/agents-towards-production)

---

*⏱️ Estimated time: 40 minutes | 📊 Difficulty: Intermediate | 💻 Includes code examples*
