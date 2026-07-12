---
title: Introduction to AI Agents
description: >-
  Understand what AI agents are, how they differ from simple LLM calls, and the
  core components that make agents autonomous
duration: 40 min
difficulty: intermediate
has_code: false
module: module-11
---
# Introduction to AI Agents

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand what makes an AI agent different from a chatbot | 40 min | Intermediate |
| Learn the core agent loop: Perceive, Reason, Act | | |
| Explore the key components of an agent system | | |
| See real-world agent applications | | |

---

## What is an AI Agent?

An **AI agent** is a system that uses an LLM as its reasoning engine to autonomously decide what actions to take, execute those actions, observe results, and iterate until a goal is achieved.

### Chatbot vs Agent

```
CHATBOT (Simple LLM Call):
  User asks question -> LLM generates answer -> Done

AGENT (Autonomous System):
  User sets goal -> LLM reasons about steps needed
    -> Takes action (search, code, API call)
    -> Observes result
    -> Decides next action
    -> Repeats until goal is met
    -> Returns final answer
```

The key difference: **agents take actions in a loop**, not just generate text once.

```python
# Simple chatbot - one LLM call
def chatbot(question):
    return llm.generate(question)

# Agent - iterative reasoning and action loop
def agent(goal):
    context = []
    while not is_goal_achieved(context):
        # 1. Reason about what to do next
        thought = llm.generate(
            f"Goal: {goal}
Context so far: {context}
"
            f"What should I do next?"
        )
        
        # 2. Choose and execute an action
        action = parse_action(thought)
        result = execute_action(action)
        
        # 3. Add observation to context
        context.append({"thought": thought, "action": action, "result": result})
    
    return synthesize_answer(context)
```

---

## The Agent Loop

Every agent follows the same fundamental cycle:

```
         ┌──────────────────────────┐
         |        USER GOAL         |
         └────────────┬─────────────┘
                      v
              ┌───────────────┐
              |   PERCEIVE    |  <-- Observe environment, read context
              └───────┬───────┘
                      v
              ┌───────────────┐
              |    REASON     |  <-- LLM decides what to do next
              └───────┬───────┘
                      v
              ┌───────────────┐
              |     ACT       |  <-- Execute tool, API call, code
              └───────┬───────┘
                      v
              ┌───────────────┐
         ┌----|   EVALUATE    |  <-- Check: is the goal met?
         |    └───────────────┘
         |          |
         | No       | Yes
         |          v
         |    ┌───────────┐
         +--->|  RETURN    |  --> Final answer to user
              └───────────┘
```

### In Code

```python
from openai import OpenAI

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute Python code and return the output",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to run"}
                },
                "required": ["code"]
            }
        }
    }
]

def run_agent(goal, max_steps=10):
    messages = [
        {"role": "system", "content": 
            "You are a helpful agent. Use the provided tools to accomplish "
            "the user's goal. Think step by step. When you have enough "
            "information to answer, provide your final response."},
        {"role": "user", "content": goal}
    ]
    
    for step in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            tools=tools
        )
        
        message = response.choices[0].message
        messages.append(message)
        
        # If no tool calls, agent is done
        if not message.tool_calls:
            return message.content
        
        # Execute each tool call
        for tool_call in message.tool_calls:
            result = execute_tool(tool_call.function.name, 
                                  tool_call.function.arguments)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })
    
    return "Agent reached maximum steps without completing the goal."
```

---

## Core Components of an Agent

### 1. The Brain (LLM)

The LLM serves as the reasoning engine. It decides:
- What information is needed
- Which tool to use
- How to interpret results
- When the task is complete

### 2. Tools

Tools give the agent the ability to interact with the world beyond text generation:

| Tool Type | Examples | Purpose |
|-----------|----------|---------|
| **Search** | Web search, document retrieval | Get current information |
| **Code execution** | Python sandbox, shell commands | Compute, transform data |
| **API calls** | REST APIs, databases | Read/write external systems |
| **File operations** | Read, write, edit files | Manage documents |
| **Communication** | Email, Slack, notifications | Interact with humans |

### 3. Memory

Agents need memory to maintain context across their reasoning steps:

- **Short-term memory**: The conversation/context within a single task (usually the message history)
- **Long-term memory**: Persistent storage across sessions (vector databases, files, key-value stores)

```python
# Short-term: conversation history (built-in)
messages = [...]  # grows as agent works

# Long-term: vector store for persistent knowledge
def store_memory(text, metadata):
    embedding = embed(text)
    vector_db.upsert(embedding, text, metadata)

def recall_memory(query, top_k=5):
    embedding = embed(query)
    return vector_db.search(embedding, top_k=top_k)
```

### 4. Planning

More advanced agents plan before acting:

```python
# Simple agent: act immediately
"Search for X" -> execute search -> "Search for Y" -> ...

# Planning agent: create plan first
"Goal: Write a market analysis report"
  Plan:
  1. Search for industry trends
  2. Search for competitor data
  3. Search for market size estimates
  4. Analyze and synthesize findings
  5. Write structured report
  -> Execute plan step by step
```

---

## When to Use Agents vs Simple LLM Calls

| Scenario | Approach | Why |
|----------|----------|-----|
| Answer a factual question | Simple LLM call | One-step, no tools needed |
| Summarize a document | Simple LLM call | Input/output, no iteration |
| Research a topic from multiple sources | **Agent** | Needs search + synthesis loop |
| Debug and fix code | **Agent** | Needs code execution + iteration |
| Book a flight based on preferences | **Agent** | Needs API calls + decision making |
| Generate a static template | Simple LLM call | No external interaction needed |
| Monitor a system and respond to alerts | **Agent** | Ongoing, reactive, uses tools |

**Rule of thumb**: If the task needs external tools or iterative reasoning, use an agent. If it's a single input-to-output transformation, a simple LLM call is better.

---

## Real-World Agent Applications

### 1. Coding Assistants
Agents that read code, run tests, fix bugs, and submit changes. Examples: GitHub Copilot Workspace, Claude Code, Cursor.

### 2. Research Assistants
Agents that search the web, read papers, synthesize findings, and produce reports.

### 3. Customer Support Agents
Agents that classify issues, look up account information, execute actions (refunds, account changes), and escalate when needed.

### 4. Data Analysis Agents
Agents that write and execute SQL/Python, create visualizations, and iterate on analysis based on findings.

---

## Key Considerations

### Safety and Guardrails

Agents act autonomously, so they need guardrails:

```python
# Always limit agent capabilities
ALLOWED_ACTIONS = ["search", "read_file", "run_python"]
MAX_STEPS = 15
REQUIRE_HUMAN_APPROVAL_FOR = ["send_email", "delete_file", "make_purchase"]

def execute_action(action):
    if action.name not in ALLOWED_ACTIONS:
        return "Action not permitted"
    if action.name in REQUIRE_HUMAN_APPROVAL_FOR:
        approval = get_human_approval(action)
        if not approval:
            return "Action rejected by human"
    return action.execute()
```

### Cost Awareness

Each agent step = an LLM call. A 10-step agent task costs 10x a simple call. Design agents to be efficient.

---

## Key Takeaways

- AI agents use LLMs as reasoning engines within an action loop
- The core cycle is: Perceive, Reason, Act, Evaluate
- Agents need tools, memory, and planning capabilities
- Use agents when tasks require iteration and external interaction
- Always implement safety guardrails and cost controls

---

## Next Lesson

**Lesson 2: Agent Architectures** - Explore different architectural patterns for building agents, from simple ReAct to advanced planning agents.
