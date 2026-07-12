---
title: The ReAct Pattern - Reasoning and Acting
description: >-
  Implement the ReAct pattern from scratch with real tools, error handling, and
  production considerations
duration: 45 min
difficulty: advanced
has_code: false
module: module-11
---
# The ReAct Pattern - Reasoning and Acting

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand the ReAct paper and pattern in depth | 45 min | Advanced |
| Implement a ReAct agent from scratch | | |
| Add error handling and recovery | | |
| Optimize for cost and reliability | | |

---

## What is ReAct?

**ReAct** (Reasoning + Acting) is an agent pattern from the 2022 paper by Yao et al. The key insight: interleaving **reasoning traces** (thinking about what to do) with **actions** (tool calls) produces much better results than either alone.

```
Without ReAct (Act only):
  Query -> Call tool -> Call tool -> Answer
  (No reasoning about which tool or why)

Without ReAct (Reason only):
  Query -> Think -> Think -> Think -> Answer
  (No access to external information)

ReAct (Reason + Act):
  Query -> Think "I need to search for X"
        -> Search for X
        -> Think "The results show Y, but I also need Z"
        -> Search for Z
        -> Think "Now I have enough to answer"
        -> Answer
```

---

## Building a ReAct Agent Step by Step

### Step 1: Define Your Tools

```python
import json
import requests
from datetime import datetime

# Tool definitions for the LLM
tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the web for current information. Use for facts, news, or data you don't know.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression. Use for any arithmetic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate, e.g. '(15 * 3) + 42'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_date",
            "description": "Get today's date. Use when you need to know the current date.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]

# Tool implementations
def execute_tool(name, arguments):
    args = json.loads(arguments) if isinstance(arguments, str) else arguments
    
    if name == "search":
        # In production, connect to a real search API
        return f"Search results for '{args['query']}': [simulated results]"
    
    elif name == "calculate":
        try:
            # Safety: only allow math operations
            allowed_chars = set("0123456789+-*/(). ")
            expr = args["expression"]
            if all(c in allowed_chars for c in expr):
                return str(eval(expr))
            else:
                return "Error: Invalid expression. Use only numbers and +, -, *, /, ()"
        except Exception as e:
            return f"Calculation error: {str(e)}"
    
    elif name == "get_current_date":
        return datetime.now().strftime("%Y-%m-%d")
    
    else:
        return f"Error: Unknown tool '{name}'"
```

### Step 2: The ReAct Loop

```python
from openai import OpenAI

client = OpenAI()

def react_agent(user_query, max_steps=8, model="gpt-4.1"):
    """
    A complete ReAct agent implementation.
    """
    system_prompt = """You are a helpful assistant with access to tools.

For each step, think carefully about what you need to do:
1. Consider what information you have and what you still need
2. Use a tool if you need external information or computation
3. When you have enough information, provide your final answer

Be efficient - don't use tools unnecessarily. If you can answer 
directly from your knowledge, do so."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query}
    ]
    
    for step in range(max_steps):
        # Get LLM response (may include tool calls)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            temperature=0.2,  # Low temperature for reliable reasoning
        )
        
        message = response.choices[0].message
        messages.append(message)
        
        # If no tool calls, agent has reached its answer
        if not message.tool_calls:
            return {
                "answer": message.content,
                "steps": step + 1,
                "messages": messages
            }
        
        # Execute each tool call
        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            
            print(f"  Step {step + 1}: Calling {tool_name}({tool_args})")
            
            result = execute_tool(tool_name, tool_args)
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })
    
    # If we hit max steps, ask for a final answer
    messages.append({
        "role": "user",
        "content": "Please provide your best answer with the information gathered so far."
    })
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2
    )
    
    return {
        "answer": response.choices[0].message.content,
        "steps": max_steps,
        "hit_max_steps": True,
        "messages": messages
    }

# Example usage
result = react_agent("What is the population of Tokyo divided by the area in square miles?")
print(f"Answer: {result['answer']}")
print(f"Steps taken: {result['steps']}")
```

### Step 3: Add Error Handling

```python
def react_agent_robust(user_query, max_steps=8, max_retries=2):
    """ReAct agent with error handling and retry logic."""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
    
    consecutive_errors = 0
    
    for step in range(max_steps):
        try:
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=messages,
                tools=tools,
                temperature=0.2,
                timeout=30,  # 30 second timeout
            )
        except Exception as e:
            consecutive_errors += 1
            if consecutive_errors >= max_retries:
                return {"answer": "I encountered repeated errors. Please try again.", 
                        "error": str(e)}
            # Add error context so the agent can adapt
            messages.append({
                "role": "user",
                "content": f"[System: API error occurred: {str(e)}. Please try a different approach.]"
            })
            continue
        
        consecutive_errors = 0  # Reset on success
        message = response.choices[0].message
        messages.append(message)
        
        if not message.tool_calls:
            return {"answer": message.content, "steps": step + 1}
        
        for tool_call in message.tool_calls:
            try:
                result = execute_tool(tool_call.function.name, 
                                      tool_call.function.arguments)
            except Exception as e:
                result = f"Tool error: {str(e)}. Try a different approach."
            
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })
    
    return {"answer": "Could not complete the task within the step limit.",
            "hit_max_steps": True}
```

---

## ReAct Optimization Tips

### 1. Be Specific in Tool Descriptions

```python
# BAD: Vague description
{"description": "Search for stuff"}

# GOOD: Specific with examples
{"description": "Search the web for current information. Use for: "
                "factual questions, recent events, statistics, prices. "
                "Returns top 3 results with snippets."}
```

### 2. Control Step Count

- Simple lookups: max_steps=3
- Research tasks: max_steps=8
- Complex analysis: max_steps=12
- Never set max_steps > 15 (cost and context window concerns)

### 3. Monitor Token Usage

Each step adds to the conversation history. A 10-step agent sends the FULL history with each call, so tokens grow quadratically.

```
Step 1:  ~500 tokens sent
Step 5:  ~3,000 tokens sent
Step 10: ~8,000 tokens sent  (expensive!)
```

**Mitigation**: Summarize earlier steps if the conversation gets long.

---

## Key Takeaways

- ReAct interleaves reasoning (Thought) with action (Tool use) and observation
- Implement with the standard chat completions API + tool calling
- Always add error handling and maximum step limits
- Be specific in tool descriptions to guide the agent
- Monitor token usage -- costs grow with each step

---

## Next Lesson

**Lesson 4: Tool Use & Function Calling** - Master the details of giving agents tools, including parameter validation, authorization, and building custom tool libraries.
