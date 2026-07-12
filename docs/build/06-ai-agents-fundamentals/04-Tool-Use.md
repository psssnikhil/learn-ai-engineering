---
title: Tool Use & Function Calling
description: >-
  Master giving AI agents tools through function calling APIs, including
  parameter design, validation, and building reusable tool libraries
duration: 40 min
difficulty: advanced
has_code: false
---
# Tool Use & Function Calling

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand how function calling works in LLM APIs | 40 min | Advanced |
| Design effective tool schemas | | |
| Handle tool execution safely | | |
| Build reusable tool libraries | | |

---

## How Function Calling Works

Function calling lets an LLM output structured tool invocations instead of (or alongside) text. The LLM does NOT execute the tools -- it decides WHICH tool to call and WITH WHAT arguments. Your code executes the actual function.

```
User: "What's the weather in Tokyo?"
          |
          v
    ┌───────────┐
    |    LLM    |  "I should call get_weather with city=Tokyo"
    └─────┬─────┘
          | Outputs: tool_call(name="get_weather", args={"city": "Tokyo"})
          v
    ┌───────────┐
    | YOUR CODE |  Calls actual weather API
    └─────┬─────┘
          | Returns: {"temp": 22, "condition": "sunny"}
          v
    ┌───────────┐
    |    LLM    |  "It's 22C and sunny in Tokyo."
    └───────────┘
```

---

## Defining Tools with JSON Schema

Tools are described using JSON Schema, which tells the LLM what each tool does and what parameters it accepts.

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city. "
                           "Returns temperature in Celsius and conditions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'Tokyo' or 'New York'"
                    },
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature units. Defaults to celsius."
                    }
                },
                "required": ["city"]
            }
        }
    }
]
```

### Tool Description Best Practices

| Do | Don't |
|----|-------|
| Explain what the tool returns | Leave description vague |
| Include example parameter values | Assume the LLM knows your API |
| Specify when to use vs not use | Give overly technical descriptions |
| Use enum for fixed choices | Allow free-form when options are limited |
| Mark required vs optional clearly | Make everything required |

---

## Building a Tool Registry

For production agents, organize tools in a registry:

```python
from typing import Callable, Any
import json

class ToolRegistry:
    def __init__(self):
        self._tools = {}
    
    def register(self, name: str, description: str, 
                 parameters: dict, handler: Callable):
        self._tools[name] = {
            "schema": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters
                }
            },
            "handler": handler
        }
    
    def get_schemas(self) -> list:
        """Return tool schemas for the LLM API."""
        return [t["schema"] for t in self._tools.values()]
    
    def execute(self, name: str, arguments: str) -> str:
        """Execute a tool by name with JSON arguments."""
        if name not in self._tools:
            return f"Error: Unknown tool '{name}'"
        
        try:
            args = json.loads(arguments)
            result = self._tools[name]["handler"](**args)
            return json.dumps(result) if not isinstance(result, str) else result
        except json.JSONDecodeError:
            return "Error: Invalid arguments format"
        except TypeError as e:
            return f"Error: Invalid parameters - {str(e)}"
        except Exception as e:
            return f"Error executing {name}: {str(e)}"

# Usage
registry = ToolRegistry()

# Register a weather tool
def get_weather(city: str, units: str = "celsius") -> dict:
    # Call actual weather API here
    return {"city": city, "temp": 22, "units": units, "condition": "sunny"}

registry.register(
    name="get_weather",
    description="Get current weather for a city",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
            "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["city"]
    },
    handler=get_weather
)

# Register a calculator tool
def calculate(expression: str) -> str:
    allowed = set("0123456789+-*/(). ")
    if not all(c in allowed for c in expression):
        return "Error: Only numbers and basic math operators allowed"
    return str(eval(expression))

registry.register(
    name="calculate",
    description="Evaluate a math expression. Only supports +, -, *, /, and parentheses.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "e.g. '(15 * 3) + 42'"}
        },
        "required": ["expression"]
    },
    handler=calculate
)

# Use in agent
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=messages,
    tools=registry.get_schemas()  # Pass all tool schemas
)

# Execute tool calls
for tool_call in response.choices[0].message.tool_calls:
    result = registry.execute(tool_call.function.name, 
                               tool_call.function.arguments)
```

---

## Safety: Validating Tool Calls

Never blindly execute what the LLM asks for. Validate inputs and restrict dangerous operations.

```python
# Input validation layer
def safe_execute(registry, tool_name, arguments):
    # 1. Check if tool exists
    if tool_name not in registry._tools:
        return "Tool not found"
    
    # 2. Parse and validate arguments
    try:
        args = json.loads(arguments)
    except json.JSONDecodeError:
        return "Invalid JSON arguments"
    
    # 3. Sanitize string inputs (prevent injection)
    for key, value in args.items():
        if isinstance(value, str) and len(value) > 10000:
            return f"Argument '{key}' exceeds maximum length"
    
    # 4. Check authorization for sensitive tools
    SENSITIVE_TOOLS = {"send_email", "delete_record", "execute_sql"}
    if tool_name in SENSITIVE_TOOLS:
        if not get_user_approval(tool_name, args):
            return "Action requires user approval. Please confirm."
    
    # 5. Execute with timeout
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Tool execution timed out")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)  # 30 second timeout
    
    try:
        result = registry.execute(tool_name, arguments)
    finally:
        signal.alarm(0)  # Cancel timeout
    
    return result
```

---

## Parallel vs Sequential Tool Calls

Modern APIs support **parallel tool calls** where the LLM requests multiple tools at once:

```python
# The LLM might return multiple tool calls in one response:
# tool_calls = [
#     call("get_weather", {"city": "Tokyo"}),
#     call("get_weather", {"city": "London"}),
#     call("get_current_date", {})
# ]

# Execute in parallel for speed
import asyncio

async def execute_tools_parallel(tool_calls, registry):
    tasks = []
    for tc in tool_calls:
        tasks.append(asyncio.to_thread(
            registry.execute, tc.function.name, tc.function.arguments
        ))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    tool_messages = []
    for tc, result in zip(tool_calls, results):
        if isinstance(result, Exception):
            result = f"Error: {str(result)}"
        tool_messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": str(result)
        })
    
    return tool_messages
```

---

## Key Takeaways

- Function calling lets LLMs decide WHICH tools to call; your code executes them
- Write clear, specific tool descriptions with examples and enums
- Use a tool registry pattern for production agents
- Always validate inputs, sanitize arguments, and add timeouts
- Support parallel tool execution for faster agent responses
- Require human approval for sensitive or destructive operations

---

## Next Lesson

**Lesson 5: Agent Memory Systems** - Learn to give agents short-term and long-term memory for maintaining context across interactions.
