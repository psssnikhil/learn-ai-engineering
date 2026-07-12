---
title: Agent Architectures
description: >-
  Explore the major architectural patterns for building AI agents, from simple
  ReAct loops to advanced planning and reflection systems
duration: 40 min
difficulty: advanced
has_code: false
---
# Agent Architectures

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Compare major agent architecture patterns | 40 min | Advanced |
| Understand when to use each pattern | | |
| Learn the tradeoffs between simplicity and capability | | |
| See how popular frameworks implement these patterns | | |

---

## Overview of Agent Architectures

Different tasks require different agent architectures. Here is the landscape from simplest to most complex:

```
Simplicity                                          Capability
|------------------------------------------------------>|
|                                                       |
| Simple       ReAct       Plan-and-     Reflection    |
| Tool Use     Loop        Execute       Agents        |
|                                                       |
| 1 LLM call   Loop until  Plan first,  Generate,     |
| + 1 tool     done        then execute  critique,     |
|                                        refine        |
```

---

## Pattern 1: Simple Tool Use (Router)

The simplest agent: the LLM decides which tool to call, calls it once, and returns.

```python
# Simple router - one decision, one action
def simple_agent(query, tools):
    response = llm.generate(
        messages=[
            {"role": "system", "content": 
                f"You have access to these tools: {describe_tools(tools)}. "
                f"Call the appropriate tool to answer the user's question."},
            {"role": "user", "content": query}
        ],
        tools=tools
    )
    
    if response.tool_calls:
        tool_result = execute_tool(response.tool_calls[0])
        # One more LLM call to synthesize the answer
        final = llm.generate(
            messages=[..., {"role": "tool", "content": tool_result}]
        )
        return final.content
    
    return response.content
```

**When to use**: Simple lookups, single-step operations, FAQ bots.

**Limitations**: Cannot handle multi-step tasks or recover from mistakes.

---

## Pattern 2: ReAct (Reason + Act) Loop

The workhorse pattern. The agent reasons about what to do, takes action, observes the result, and loops until done.

```
┌─────────┐     ┌────────┐     ┌─────────┐
| Thought  | --> | Action | --> | Observe |
| "I need  |    | Search |    | Got 3    |
|  to find"|    | for X  |    | results  |
└─────────┘     └────────┘     └────┬────┘
     ^                              |
     |          Is goal met?        |
     |          No ─────────────────┘
     |          Yes ──> Return final answer
```

```python
def react_agent(goal, tools, max_steps=10):
    messages = [
        {"role": "system", "content": 
            "You are a helpful agent. For each step:
"
            "1. Think about what you need to do next
"
            "2. Use a tool if needed
"
            "3. When you have enough information, give your final answer
"
            f"Available tools: {describe_tools(tools)}"},
        {"role": "user", "content": goal}
    ]
    
    for step in range(max_steps):
        response = llm.generate(messages=messages, tools=tools)
        messages.append(response)
        
        # No tool calls means agent is done
        if not response.tool_calls:
            return response.content
        
        # Execute tools and add observations
        for tool_call in response.tool_calls:
            result = execute_tool(tool_call)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })
    
    return "Reached maximum steps."
```

**When to use**: Research tasks, data gathering, debugging, most general-purpose agents.

**Limitations**: Can get stuck in loops, no upfront planning, context window fills up.

---

## Pattern 3: Plan-and-Execute

The agent creates a plan first, then executes each step. Better for complex, multi-step tasks.

```
User Goal
    |
    v
┌──────────────────┐
| PLANNER (LLM)    |
| Creates step-by- |
| step plan        |
└────────┬─────────┘
         v
Plan: [Step 1, Step 2, Step 3, Step 4]
         |
         v
┌──────────────────┐
| EXECUTOR (LLM)   |  For each step:
| Executes each    |  - Execute with tools
| step with tools  |  - Validate result
└────────┬─────────┘  - Update plan if needed
         v
Final Result
```

```python
def plan_and_execute_agent(goal, tools):
    # Phase 1: Planning
    plan = llm.generate(
        messages=[{
            "role": "system",
            "content": "Create a step-by-step plan to accomplish the goal. "
                       "Output as a numbered list. Be specific about what "
                       "information is needed at each step."
        }, {
            "role": "user",
            "content": goal
        }]
    )
    
    steps = parse_plan(plan.content)
    results = []
    
    # Phase 2: Execution
    for i, step in enumerate(steps):
        step_result = react_agent(
            goal=f"Execute this step: {step}

"
                 f"Previous results: {results}",
            tools=tools,
            max_steps=5
        )
        results.append({"step": step, "result": step_result})
        
        # Optional: re-plan if step failed
        if step_failed(step_result):
            revised_plan = llm.generate(
                f"Step '{step}' failed with: {step_result}
"
                f"Revise the remaining plan."
            )
            steps = steps[:i+1] + parse_plan(revised_plan.content)
    
    # Phase 3: Synthesis
    final = llm.generate(
        f"Goal: {goal}
Results from all steps: {results}
"
        f"Synthesize a final answer."
    )
    return final.content
```

**When to use**: Complex multi-step tasks, project planning, report generation, tasks where order matters.

**Limitations**: Planning takes extra tokens/time, plan may become stale, re-planning adds complexity.

---

## Pattern 4: Reflection / Self-Critique

The agent generates output, then critiques its own work, and iterates to improve.

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
| Generate    | --> | Critique     | --> | Is it good   |
| initial     |    | "What's      |    | enough?      |
| response    |    |  wrong with  |    |              |
|             |    |  this?"      |    | No -> Revise |
└─────────────┘    └──────────────┘    | Yes -> Return|
                                       └──────────────┘
```

```python
def reflection_agent(task, max_iterations=3):
    # Generate initial response
    draft = llm.generate(f"Complete this task: {task}")
    
    for i in range(max_iterations):
        # Critique
        critique = llm.generate(
            f"Task: {task}

"
            f"Current draft:
{draft}

"
            f"Critique this draft. What is missing, incorrect, or could "
            f"be improved? Be specific and actionable."
        )
        
        # Check if good enough
        if "no major issues" in critique.lower() or "looks good" in critique.lower():
            return draft
        
        # Revise based on critique
        draft = llm.generate(
            f"Task: {task}

"
            f"Previous draft:
{draft}

"
            f"Critique:
{critique}

"
            f"Revise the draft to address all critique points."
        )
    
    return draft
```

**When to use**: Writing tasks, code generation, any task where iterative refinement improves quality.

**Limitations**: Multiple LLM calls (expensive), may over-optimize, needs good stopping criteria.

---

## Choosing the Right Architecture

| Task Type | Recommended Pattern | Why |
|-----------|--------------------|-----|
| Simple Q&A with tools | Simple Tool Use | One step is enough |
| Research and information gathering | ReAct | Needs iterative search |
| Complex multi-step workflow | Plan-and-Execute | Benefits from upfront planning |
| Content creation | Reflection | Iterative refinement improves quality |
| Code generation + testing | ReAct + Reflection | Generate, test, fix loop |
| Autonomous long-running tasks | Plan-and-Execute + ReAct | Plan the strategy, execute adaptively |

---

## Key Takeaways

- Start with the simplest architecture that can solve your task
- ReAct is the most versatile general-purpose pattern
- Plan-and-Execute adds structure for complex multi-step tasks
- Reflection improves quality through self-critique loops
- Production agents often combine patterns (e.g., Plan-and-Execute with ReAct execution and Reflection review)

---

## Next Lesson

**Lesson 3: The ReAct Pattern** - Deep dive into implementing the ReAct pattern with real tools and error handling.
