---
title: 'Agent Frameworks (LangGraph, CrewAI)'
description: >-
  Compare popular agent frameworks including LangGraph, CrewAI, and Autogen, and
  learn when to use a framework vs building from scratch
duration: 35 min
difficulty: advanced
has_code: false
module: module-11
objectives:
  - Compare the design philosophy of LangGraph vs CrewAI
  - Build a simple agent using LangGraph's graph-based approach
  - Set up a multi-agent team using CrewAI
  - Explain when to use a framework vs building from scratch
  - Describe the strengths and limitations of each framework
---
# Agent Frameworks (LangGraph, CrewAI)

## What You'll Learn

By the end of this lesson, you'll understand:
- The landscape of AI agent frameworks
- LangGraph: graph-based agent workflows
- CrewAI: role-based multi-agent teams
- AutoGen: conversation-driven agent collaboration
- When to use a framework and when to build from scratch

**Time to Complete**: 35 minutes
**Difficulty**: Advanced

---

## The Agent Framework Landscape

Building agents from scratch gives you full control but means reimplementing common patterns. Frameworks handle the boilerplate and provide battle-tested abstractions.

| Framework | Philosophy | Best For |
|-----------|-----------|----------|
| LangGraph | Agents as state machines (graphs) | Complex workflows with branching logic |
| CrewAI | Agents as team members with roles | Multi-agent collaboration |
| AutoGen | Agents as conversational participants | Research and experimentation |
| OpenAI Agents SDK | Tool-calling with handoffs | Production OpenAI-based agents |

---

## LangGraph

LangGraph models agents as **directed graphs** where nodes are actions and edges are decisions. This makes complex control flow explicit and debuggable.

### Core Concepts

- **State**: A shared data object passed between nodes
- **Nodes**: Functions that transform state (LLM calls, tool use, logic)
- **Edges**: Connections between nodes (conditional or unconditional)
- **Graph**: The complete workflow definition

### Building a LangGraph Agent

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# Define the state that flows through the graph
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    next_action: str

# Define node functions
def reasoning_node(state: AgentState) -> AgentState:
    """LLM decides what to do next."""
    messages = state["messages"]
    response = llm.invoke(messages)

    if should_use_tool(response):
        return {"messages": [response], "next_action": "tool"}
    else:
        return {"messages": [response], "next_action": "end"}

def tool_node(state: AgentState) -> AgentState:
    """Execute the tool the LLM requested."""
    last_message = state["messages"][-1]
    tool_result = execute_tool(last_message)
    return {"messages": [tool_result], "next_action": "reason"}

# Build the graph
graph = StateGraph(AgentState)
graph.add_node("reason", reasoning_node)
graph.add_node("tool", tool_node)

# Add edges
graph.set_entry_point("reason")
graph.add_conditional_edges("reason", lambda s: s["next_action"], {
    "tool": "tool",
    "end": END
})
graph.add_edge("tool", "reason")  # After tool use, reason again

# Compile and run
agent = graph.compile()
result = agent.invoke({"messages": [user_message], "next_action": ""})
```

### LangGraph Strengths

- **Explicit control flow**: You can see exactly how the agent decides what to do
- **Persistence**: Built-in checkpointing for long-running agents
- **Human-in-the-loop**: Easy to add approval steps between nodes
- **Debugging**: Graph visualization shows execution path

---

## CrewAI

CrewAI models agents as **team members** with specific roles, goals, and backstories. Multiple agents collaborate to complete complex tasks.

### Core Concepts

- **Agent**: An individual with a role, goal, and backstory
- **Task**: A specific piece of work assigned to an agent
- **Crew**: A team of agents working together
- **Process**: How the crew coordinates (sequential or hierarchical)

### Building a CrewAI Team

```python
from crewai import Agent, Task, Crew, Process

# Define agents with distinct roles
researcher = Agent(
    role="Research Analyst",
    goal="Find accurate, up-to-date information on the given topic",
    backstory="You are an expert researcher with 10 years of experience in technology analysis.",
    verbose=True,
    llm="gpt-4o"
)

writer = Agent(
    role="Technical Writer",
    goal="Transform research findings into clear, engaging content",
    backstory="You are a skilled technical writer who makes complex topics accessible.",
    verbose=True,
    llm="gpt-4o"
)

editor = Agent(
    role="Editor",
    goal="Ensure content is accurate, well-structured, and free of errors",
    backstory="You are a detail-oriented editor with high standards for quality.",
    verbose=True,
    llm="gpt-4o-mini"
)

# Define tasks
research_task = Task(
    description="Research the current state of AI agents in 2025. Cover key frameworks, patterns, and trends.",
    expected_output="A detailed research summary with citations",
    agent=researcher
)

writing_task = Task(
    description="Write a 1000-word article based on the research findings.",
    expected_output="A polished article in markdown format",
    agent=writer,
    context=[research_task]  # Depends on research
)

editing_task = Task(
    description="Review and edit the article for accuracy and clarity.",
    expected_output="The final edited article",
    agent=editor,
    context=[writing_task]
)

# Assemble the crew
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editing_task],
    process=Process.sequential,  # Tasks run in order
    verbose=True
)

# Run the crew
result = crew.kickoff()
```

### CrewAI Strengths

- **Intuitive mental model**: Agents as team members is easy to reason about
- **Role specialization**: Each agent focuses on what it does best
- **Built-in delegation**: Agents can delegate to each other
- **Quick prototyping**: Get multi-agent systems running fast

---

## AutoGen

AutoGen (by Microsoft) models agents as participants in a conversation. Agents take turns speaking and can include human participants.

```python
from autogen import AssistantAgent, UserProxyAgent

# Create agents
assistant = AssistantAgent(
    name="assistant",
    llm_config={"model": "gpt-4o"},
    system_message="You are a helpful AI assistant."
)

user_proxy = UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",  # Automated mode
    code_execution_config={"work_dir": "output"}
)

# Start a conversation
user_proxy.initiate_chat(
    assistant,
    message="Write a Python function to find prime numbers up to N."
)
```

---

## Framework Comparison

| Feature | LangGraph | CrewAI | AutoGen |
|---------|-----------|--------|---------|
| Control flow | Explicit graphs | Role-based sequential/hierarchical | Conversation turns |
| Multi-agent | Via separate nodes | Native crew concept | Native multi-agent chat |
| Persistence | Built-in checkpoints | Limited | Limited |
| Learning curve | Medium-high | Low-medium | Low |
| Debugging | Graph visualization | Verbose logs | Chat transcripts |
| Production readiness | High | Medium | Medium |
| Customization | Very flexible | Moderate | Moderate |

---

## When to Use a Framework vs. Build From Scratch

### Use a Framework When:
- You need multi-agent collaboration
- Your workflow has complex branching logic
- You want persistence and checkpointing out of the box
- You are prototyping and need to move fast
- The framework's abstractions match your use case

### Build From Scratch When:
- You need full control over the agent loop
- Your use case is simple (single agent, few tools)
- Framework overhead is too much for your latency budget
- You need deep customization of every step
- You want to minimize dependencies

---

## Resources

- **LangGraph Documentation** -- Graph-based agent framework by LangChain
- **CrewAI Documentation** -- Multi-agent framework with role-based design
- **AutoGen Documentation** -- Microsoft's multi-agent conversation framework
- **OpenAI Agents SDK** -- OpenAI's production agent framework

---

## Key Takeaways

1. **LangGraph** excels at explicit, debuggable workflows with complex control flow
2. **CrewAI** makes multi-agent collaboration intuitive with role-based design
3. **AutoGen** is great for conversational multi-agent experimentation
4. **No framework is universally best** -- choose based on your specific requirements
5. **Start without a framework** for simple agents, adopt one when complexity demands it
