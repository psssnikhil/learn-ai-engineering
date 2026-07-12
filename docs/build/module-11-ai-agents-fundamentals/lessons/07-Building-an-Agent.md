---
title: Building an Agent from Scratch
description: >-
  Build a complete AI agent step-by-step with tools, memory, and error handling
  -- no frameworks required
duration: 45 min
difficulty: advanced
has_code: false
module: module-11
objectives:
  - Build a complete agent loop with tool calling
  - Add error handling and retry logic
  - Implement conversation memory in the agent
  - Test the agent with multi-step tasks
---
# Building an Agent from Scratch

## Learning Objectives

By the end of this lesson, you will be able to:
- Build a fully functional agent from scratch using only the OpenAI SDK
- Implement the complete agent loop: prompt, reason, act, observe
- Add robust error handling and guardrails
- Test your agent on real multi-step tasks

---

## Why Build from Scratch?

Frameworks like LangChain and CrewAI are useful, but building from scratch teaches you:
- Exactly how agents work under the hood
- How to debug agent failures
- How to customize behavior that frameworks do not support
- When you actually need a framework vs. when raw code is simpler

---

## The Complete Agent

Here is a full agent implementation in ~100 lines:

```python
from openai import OpenAI
import json

client = OpenAI()

# ── Tool Definitions ──

def search_web(query: str) -> str:
    """Search the web for information. Returns a summary of results."""
    # In production, call a real search API (Tavily, Serper, etc.)
    return f"Search results for '{query}': [simulated results]"

def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: Invalid characters in expression"
    try:
        result = eval(expression)  # Safe because we validated input
        return str(result)
    except Exception as e:
        return f"Error: {e}"

def get_current_date() -> str:
    """Get today's date."""
    from datetime import date
    return date.today().isoformat()

# Tool registry -- maps names to functions and schemas
TOOLS = {
    "search_web": {
        "function": search_web,
        "schema": {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for current information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"],
                },
            },
        },
    },
    "calculate": {
        "function": calculate,
        "schema": {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Evaluate a math expression (e.g., '2 + 2', '100 * 0.15')",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression"}
                    },
                    "required": ["expression"],
                },
            },
        },
    },
    "get_current_date": {
        "function": get_current_date,
        "schema": {
            "type": "function",
            "function": {
                "name": "get_current_date",
                "description": "Get today's date",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    },
}


# ── The Agent ──

class Agent:
    def __init__(self, system_prompt: str, model: str = "gpt-4.1",
                 max_iterations: int = 10):
        self.system_prompt = system_prompt
        self.model = model
        self.max_iterations = max_iterations
        self.messages = [{"role": "system", "content": system_prompt}]
        self.tool_schemas = [t["schema"] for t in TOOLS.values()]

    def run(self, user_input: str) -> str:
        """Run the agent loop until the task is complete."""
        self.messages.append({"role": "user", "content": user_input})

        for i in range(self.max_iterations):
            # Call the LLM
            response = client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.tool_schemas,
            )
            message = response.choices[0].message

            # If no tool calls, the agent is done
            if not message.tool_calls:
                self.messages.append(message)
                return message.content

            # Execute each tool call
            self.messages.append(message)
            for tool_call in message.tool_calls:
                result = self._execute_tool(tool_call)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        return "Agent reached maximum iterations without completing the task."

    def _execute_tool(self, tool_call) -> str:
        """Execute a tool call and return the result."""
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            return f"Error: Could not parse arguments for {name}"

        if name not in TOOLS:
            return f"Error: Unknown tool '{name}'"

        try:
            func = TOOLS[name]["function"]
            result = func(**args)
            return str(result)
        except Exception as e:
            return f"Error executing {name}: {e}"


# ── Usage ──

agent = Agent(
    system_prompt=(
        "You are a helpful research assistant. "
        "Use your tools to find information and perform calculations. "
        "Always verify your answers with tools rather than guessing."
    )
)

# Single-turn
# response = agent.run("What is 15% tip on a $85.50 dinner bill?")

# Multi-turn (agent remembers previous messages)
# response = agent.run("Now calculate 20% tip on the same bill")
```

---

## Adding Error Handling

Production agents need to handle failures gracefully:

```python
import time

class RobustAgent(Agent):
    def _execute_tool(self, tool_call) -> str:
        """Execute with retry logic and timeout."""
        name = tool_call.function.name

        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            return f"Error: Invalid JSON arguments for {name}"

        if name not in TOOLS:
            return f"Error: Tool '{name}' not found. Available: {list(TOOLS.keys())}"

        # Retry up to 2 times on failure
        for attempt in range(3):
            try:
                func = TOOLS[name]["function"]
                result = func(**args)
                return str(result)
            except Exception as e:
                if attempt < 2:
                    time.sleep(1)
                    continue
                return f"Error: {name} failed after 3 attempts: {e}"

        return f"Error: {name} failed unexpectedly"
```

---

## Adding Streaming Output

For a better user experience, stream the agent's reasoning:

```python
def run_streaming(agent, user_input: str):
    """Run agent with streaming output."""
    agent.messages.append({"role": "user", "content": user_input})

    for i in range(agent.max_iterations):
        stream = client.chat.completions.create(
            model=agent.model,
            messages=agent.messages,
            tools=agent.tool_schemas,
            stream=True,
        )

        # Collect the streamed response
        content_parts = []
        tool_calls_data = {}

        for chunk in stream:
            delta = chunk.choices[0].delta

            if delta.content:
                print(delta.content, end="", flush=True)
                content_parts.append(delta.content)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_data:
                        tool_calls_data[idx] = {
                            "id": tc.id or "",
                            "name": tc.function.name if tc.function and tc.function.name else "",
                            "arguments": "",
                        }
                    if tc.function and tc.function.arguments:
                        tool_calls_data[idx]["arguments"] += tc.function.arguments
                    if tc.id:
                        tool_calls_data[idx]["id"] = tc.id

        full_content = "".join(content_parts)

        if not tool_calls_data:
            print()  # newline after streaming
            return full_content

        # Execute tools and continue the loop
        for idx, tc_data in sorted(tool_calls_data.items()):
            print(f"
  [Tool: {tc_data['name']}({tc_data['arguments']})]")
            # ... execute tool and append results
```

---

## Testing Your Agent

Write test cases that exercise multi-step reasoning:

```python
def test_agent():
    agent = Agent(
        system_prompt="You are a helpful assistant with access to tools."
    )

    # Test 1: Simple tool use
    response = agent.run("What is today's date?")
    assert "202" in response, "Should contain a date"

    # Test 2: Multi-step reasoning
    agent2 = Agent(
        system_prompt="You are a helpful assistant with access to tools."
    )
    response = agent2.run(
        "Calculate 18% tip on $124.50, then add the tip to get the total"
    )
    assert "146" in response or "147" in response, "Should calculate correctly"

    # Test 3: Error recovery
    agent3 = Agent(
        system_prompt="You are a helpful assistant with access to tools."
    )
    response = agent3.run("What is the square root of -1?")
    # Agent should handle the math error gracefully

    print("All tests passed!")

# test_agent()
```

---

## Key Takeaways

- A complete agent needs only ~100 lines: tool definitions, a message loop, and tool execution
- The agent loop is: prompt -> LLM -> check for tool calls -> execute tools -> repeat
- Always add error handling for tool execution failures and argument parsing
- Test with multi-step tasks that require tool chaining
- Start without a framework, then adopt one only when you need features you cannot build quickly

## Resources

- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling) -- Official documentation
- [Anthropic Tool Use Guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) -- Claude's approach to tool calling
- [Building effective agents (Anthropic blog)](https://www.anthropic.com/engineering/building-effective-agents) -- Production agent patterns

---

Next: Agent Frameworks
