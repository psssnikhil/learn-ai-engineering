---
title: Tool Use & Function Calling
description: >-
  Master giving AI agents tools through function calling APIs, including
  parameter design, validation, and building reusable tool libraries
duration: 60 min
difficulty: advanced
has_code: true
module: module-11
---

# Tool Use & Function Calling

## Prerequisites

- **Lesson 01 — Introduction to Agents** — understand the tool execution boundary
- **Lesson 03 — ReAct Pattern** — tool schemas and `execute_tool` pattern
- **Python intermediate** — type hints, decorators, `asyncio`, JSON
- **OpenAI API** — familiar with `chat.completions.create` and the `tools` parameter

---

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand the full function-calling protocol at the API level | 10 min | Advanced |
| Design JSON Schema tool descriptions that guide the LLM accurately | 10 min | Advanced |
| Build a production-grade ToolRegistry with validation and error handling | 15 min | Advanced |
| Implement parallel tool execution with asyncio | 10 min | Advanced |
| Secure tool execution against injection and abuse | 15 min | Advanced |

---

## Intuition First: The Tool Execution Contract

The most important mental model for tool use is the **boundary of responsibility**:

```
┌──────────────────────────────────────────────────────────┐
│                        LLM                               │
│  - Reads tool descriptions                               │
│  - Decides which tool to call and with what arguments    │
│  - DOES NOT execute tools                                │
│  - DOES NOT validate inputs before deciding              │
└───────────────────────────┬──────────────────────────────┘
                            │ outputs tool_call(name, args)
                            │
┌───────────────────────────▼──────────────────────────────┐
│                     YOUR CODE                            │
│  - Receives the tool call request                        │
│  - Validates the arguments                               │
│  - Checks authorization                                  │
│  - Executes the actual function                          │
│  - Returns result as a string back to the LLM            │
└──────────────────────────────────────────────────────────┘
```

The LLM is the director; your code is the executor. This means **all security, validation, authorization, and rate limiting must live in your code** — never rely on the LLM to restrain itself.

---

## How Function Calling Works at the API Level

When you call `chat.completions.create` with a `tools` parameter, the API does the following:

1. Appends tool schemas to the end of the system context.
2. Allows the model to respond with a special `tool_calls` field (instead of or alongside `content`).
3. Returns a response with `finish_reason="tool_calls"` when the model wants to invoke a tool.

The model does NOT call your function. It outputs a structured JSON description:

```json
{
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\"city\": \"Tokyo\", \"units\": \"celsius\"}"
      }
    }
  ]
}
```

Your code extracts `name` and `arguments`, executes the real function, and returns results as a `role: "tool"` message with the matching `tool_call_id`.

```python
from openai import OpenAI
import json

client = OpenAI()

def function_calling_demo():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "units": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature units. Default: celsius.",
                        },
                    },
                    "required": ["city"],
                },
            },
        }
    ]

    messages = [
        {"role": "user", "content": "What's the weather in Tokyo?"}
    ]

    # Step 1: LLM decides to call get_weather
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
    )
    message = response.choices[0].message
    print(f"finish_reason: {response.choices[0].finish_reason}")
    # → "tool_calls"
    print(f"tool_calls: {message.tool_calls}")
    # → [ChatCompletionMessageToolCall(id='call_abc123', function=Function(name='get_weather', arguments='{"city": "Tokyo"}'))]

    messages.append(message)

    # Step 2: YOUR CODE executes the tool
    tc = message.tool_calls[0]
    args = json.loads(tc.function.arguments)
    weather_result = {"city": args["city"], "temp_c": 26, "condition": "Sunny"}

    # Step 3: Return result as tool message
    messages.append({
        "role": "tool",
        "tool_call_id": tc.id,                     # must match the original id
        "content": json.dumps(weather_result),
    })

    # Step 4: LLM synthesizes a natural-language response
    final = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    print(final.choices[0].message.content)
    # → "The current weather in Tokyo is 26°C and sunny."
```

---

## Designing Effective Tool Schemas

The JSON Schema description is the contract between you and the LLM. It determines whether the model calls the right tool with the right arguments.

### Tool Description Best Practices

```python
# ❌ BAD: Vague, no examples, unclear scope
{
    "name": "database_query",
    "description": "Query the database",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        },
        "required": ["query"]
    }
}

# ✅ GOOD: Specific, examples included, scope and limitations clear
{
    "name": "search_orders",
    "description": (
        "Search for customer orders by order ID, customer email, or date range. "
        "Returns a list of orders with status, items, and total amount. "
        "Use this for: 'find order #12345', 'orders for john@example.com', "
        "'orders placed last week'. "
        "Do NOT use for: payment processing, refunds, or order modifications."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "Order ID to look up, e.g. 'ORD-2024-001234'",
            },
            "customer_email": {
                "type": "string",
                "description": "Customer email to search orders for",
            },
            "date_from": {
                "type": "string",
                "description": "Start date in YYYY-MM-DD format",
            },
            "date_to": {
                "type": "string",
                "description": "End date in YYYY-MM-DD format",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return. Default: 10. Max: 50.",
                "minimum": 1,
                "maximum": 50,
                "default": 10,
            },
        },
        # No required fields — any combination of filters is valid
        "required": [],
    },
}
```

**Rules for good schemas:**

| Rule | Reason |
|------|--------|
| Say what the tool *returns* in the description | LLM needs to know what it will get back |
| Include 2–3 concrete examples in the description | Reduces misuse; helps the LLM map queries to tools |
| Say when NOT to use the tool | Prevents the LLM from using a tool for the wrong purpose |
| Use `enum` for fixed choices | Forces the LLM to pick from your valid options |
| Use `minimum`/`maximum` for numeric bounds | Makes schema self-documenting |
| Mark required vs optional correctly | Missing required args cause runtime errors |

---

## Building a Production ToolRegistry

For production agents with many tools, a registry pattern centralizes tool management:

```python
from typing import Callable, Any
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: dict
    handler: Callable
    requires_auth: bool = False     # whether this tool needs human approval
    timeout_seconds: int = 30       # max execution time


class ToolRegistry:
    """
    Centralized registry for agent tools.
    Handles schema exposure, execution, and safety checks.
    """

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        description: str,
        parameters: dict,
        handler: Callable,
        requires_auth: bool = False,
        timeout_seconds: int = 30,
    ) -> "ToolRegistry":
        """Register a tool. Returns self for method chaining."""
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            requires_auth=requires_auth,
            timeout_seconds=timeout_seconds,
        )
        logger.info(f"Registered tool: {name}")
        return self

    def get_schemas(self) -> list[dict]:
        """Return all tool schemas in the OpenAI function-calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    def execute(
        self,
        name: str,
        arguments: str,
        auth_callback: Callable[[str, dict], bool] | None = None,
    ) -> str:
        """
        Execute a tool by name with JSON arguments.
        Validates, checks auth, executes, and returns result as string.
        """
        # 1. Check tool exists
        if name not in self._tools:
            return f"Error: Unknown tool '{name}'. Available: {list(self._tools.keys())}"

        tool = self._tools[name]

        # 2. Parse arguments
        try:
            args = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError as e:
            return f"Error: Malformed JSON arguments: {e}"

        # 3. Validate argument values (basic sanity checks)
        validation_error = self._validate_args(tool, args)
        if validation_error:
            return f"Validation error: {validation_error}"

        # 4. Authorization check for sensitive tools
        if tool.requires_auth:
            if auth_callback is None or not auth_callback(name, args):
                return (
                    f"Authorization required for '{name}'. "
                    "A human reviewer must approve this action."
                )

        # 5. Execute with timeout
        import signal

        def _timeout_handler(signum, frame):
            raise TimeoutError(f"Tool '{name}' timed out after {tool.timeout_seconds}s")

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(tool.timeout_seconds)
        try:
            result = tool.handler(**args)
            result_str = json.dumps(result) if not isinstance(result, str) else result
        except TimeoutError as e:
            result_str = str(e)
        except TypeError as e:
            result_str = f"Error: Invalid parameters for '{name}': {e}"
        except Exception as e:
            result_str = f"Error executing '{name}': {type(e).__name__}: {e}"
        finally:
            signal.alarm(0)

        # 6. Truncate very long results to prevent context overflow
        max_len = 4000
        if len(result_str) > max_len:
            result_str = result_str[:max_len] + f"\n[Result truncated at {max_len} chars]"

        logger.info(f"Tool '{name}' returned: {result_str[:100]}")
        return result_str

    def _validate_args(self, tool: ToolDefinition, args: dict) -> str | None:
        """Basic argument validation. Returns error message or None."""
        required = tool.parameters.get("required", [])
        for field_name in required:
            if field_name not in args:
                return f"Missing required argument: '{field_name}'"

        properties = tool.parameters.get("properties", {})
        for arg_name, arg_value in args.items():
            if arg_name not in properties:
                continue    # Extra args: silently ignore (some APIs allow this)

            prop = properties[arg_name]

            # String length check
            if isinstance(arg_value, str) and len(arg_value) > 10_000:
                return f"Argument '{arg_name}' exceeds maximum length of 10,000 characters"

            # Enum check
            if "enum" in prop and arg_value not in prop["enum"]:
                return f"'{arg_value}' is not a valid value for '{arg_name}'. Must be one of: {prop['enum']}"

            # Numeric bounds
            if isinstance(arg_value, (int, float)):
                if "minimum" in prop and arg_value < prop["minimum"]:
                    return f"'{arg_name}' must be >= {prop['minimum']}"
                if "maximum" in prop and arg_value > prop["maximum"]:
                    return f"'{arg_name}' must be <= {prop['maximum']}"

        return None   # No error
```

---

## Example: Populating a Registry

```python
import httpx
from datetime import datetime

registry = ToolRegistry()

# ── Tool 1: Weather ──────────────────────────────────────────
def get_weather(city: str, units: str = "celsius") -> dict:
    # In production: call OpenWeatherMap, WeatherAPI, etc.
    return {"city": city, "temp": 22, "units": units, "condition": "Partly cloudy"}

registry.register(
    name="get_weather",
    description=(
        "Get current weather for a city. Returns temperature and conditions. "
        "Use for: 'weather in Tokyo', 'is it raining in London'. "
        "Not for: historical weather or forecasts."
    ),
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name, e.g. 'Tokyo'"},
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature units. Default: celsius.",
            },
        },
        "required": ["city"],
    },
    handler=get_weather,
)

# ── Tool 2: Calculator ───────────────────────────────────────
import math as _math

def calculate(expression: str) -> str:
    allowed = {
        "math": _math, "round": round, "abs": abs,
        "max": max, "min": min, "int": int, "float": float,
    }
    try:
        return str(eval(expression, {"__builtins__": {}}, allowed))
    except Exception as e:
        return f"Error: {e}"

registry.register(
    name="calculate",
    description="Evaluate a math expression. Examples: '2 ** 10', 'math.sqrt(144)', 'round(3.14159, 2)'.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Python math expression, e.g. '12000000 * 0.15'",
            }
        },
        "required": ["expression"],
    },
    handler=calculate,
)

# ── Tool 3: Send Email (requires authorization) ──────────────
def send_email(to: str, subject: str, body: str) -> str:
    # In production: call SendGrid, SES, etc.
    return f"Email queued to {to}: subject='{subject}'"

registry.register(
    name="send_email",
    description=(
        "Send an email to a specified recipient. "
        "IMPORTANT: Always confirm with the user before sending. "
        "Use for: notifications, reports, follow-ups."
    ),
    parameters={
        "type": "object",
        "properties": {
            "to":      {"type": "string", "description": "Recipient email address"},
            "subject": {"type": "string", "description": "Email subject line"},
            "body":    {"type": "string", "description": "Email body text"},
        },
        "required": ["to", "subject", "body"],
    },
    handler=send_email,
    requires_auth=True,    # Require human approval before sending
    timeout_seconds=15,
)

# Use in an agent
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's the weather in Paris in fahrenheit?"}],
    tools=registry.get_schemas(),
)
for tc in response.choices[0].message.tool_calls or []:
    result = registry.execute(tc.function.name, tc.function.arguments)
    print(f"{tc.function.name} → {result}")
```

---

## Parallel Tool Execution

Modern LLM APIs may return multiple tool calls in a single response. Execute them concurrently to minimize wall-clock latency:

```python
import asyncio
from openai.types.chat import ChatCompletionMessageToolCall

async def execute_tool_async(
    registry: ToolRegistry,
    tc: ChatCompletionMessageToolCall,
) -> dict:
    """Execute one tool call in a thread pool (for blocking I/O tools)."""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,   # default ThreadPoolExecutor
        registry.execute,
        tc.function.name,
        tc.function.arguments,
    )
    return {
        "role": "tool",
        "tool_call_id": tc.id,
        "content": result,
    }


async def execute_tool_calls_parallel(
    registry: ToolRegistry,
    tool_calls: list[ChatCompletionMessageToolCall],
) -> list[dict]:
    """Execute all tool calls concurrently. Return tool messages in order."""
    tasks = [execute_tool_async(registry, tc) for tc in tool_calls]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    tool_messages = []
    for tc, result in zip(tool_calls, results):
        if isinstance(result, Exception):
            tool_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": f"Error: {result}",
            })
        else:
            tool_messages.append(result)

    return tool_messages


# Integration in the agent loop
async def react_agent_async(query: str, registry: ToolRegistry) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful agent."},
        {"role": "user", "content": query},
    ]

    for _ in range(10):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=registry.get_schemas(),
        )
        message = response.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return message.content

        # Execute all tool calls in parallel
        tool_messages = await execute_tool_calls_parallel(registry, message.tool_calls)
        messages.extend(tool_messages)

    return "Agent reached maximum steps."
```

**Latency impact**: If an agent calls two web searches and a calculator in one step, sequential execution takes ~1,500 ms; parallel execution takes ~600 ms (the max of the individual tool latencies).

!!! warning "Parallel tool calls and data dependencies"
    Don't execute in parallel if tool calls are data-dependent (e.g., tool B needs the result of tool A). The LLM typically groups independent calls in one response — but verify this assumption by checking call semantics, not just that multiple calls appeared together.

---

## Securing Tool Execution

Tools are dangerous. The LLM may be manipulated by adversarial content in tool results (prompt injection). Always apply these defenses:

**Input sanitization**: Strip instruction-like patterns from tool arguments before logging or forwarding.

```python
import re

def sanitize_tool_input(text: str) -> str:
    """Remove patterns that look like injected instructions."""
    # Remove text that tries to override system instructions
    injection_patterns = [
        r"ignore (all |your )?(previous |prior )?instructions?",
        r"system:\s+",
        r"assistant:\s+",
        r"<\|.*?\|>",    # OpenAI-style special tokens
    ]
    for pattern in injection_patterns:
        text = re.sub(pattern, "[REMOVED]", text, flags=re.IGNORECASE)
    return text
```

**Output truncation**: Long tool outputs can overwhelm the context window and push earlier instructions out of the model's effective attention. Always truncate.

**Authorization tiers**: Classify tools by risk level.

```python
TOOL_TIERS = {
    "read":      {"search", "get_weather", "calculate", "get_date"},
    "write":     {"create_document", "update_record"},
    "sensitive": {"send_email", "make_purchase", "delete_record"},
}

def tier_for_tool(name: str) -> str:
    for tier, tools in TOOL_TIERS.items():
        if name in tools:
            return tier
    return "unknown"
```

**Rate limiting per tool**: Prevent runaway agents from making thousands of API calls.

```python
from collections import defaultdict
import time

class ToolRateLimiter:
    def __init__(self, calls_per_minute: int = 20):
        self._calls = defaultdict(list)
        self._limit = calls_per_minute

    def check(self, tool_name: str) -> bool:
        """Returns True if the call is allowed, False if rate limited."""
        now = time.time()
        window = [t for t in self._calls[tool_name] if now - t < 60]
        self._calls[tool_name] = window
        if len(window) >= self._limit:
            return False
        self._calls[tool_name].append(now)
        return True
```

---

## Edge Cases & Misconceptions

**Misconception: `required: []` means the tool is optional.**
`required` specifies which *parameters* are mandatory when the tool is called. An empty `required` array means the tool can be called with no arguments. It does not mean the tool is optional in the tool list.

**Misconception: The LLM validates schema compliance before calling tools.**
The LLM attempts to generate valid arguments based on the schema description, but it does not run a schema validator. Arguments may violate constraints (`maximum`, `minLength`, etc.) because those constraints are interpreted probabilistically, not enforced. Always validate in your `execute` function.

**Edge case: Tool calls with empty arguments.**
Some tools (like `get_current_date`) take no parameters. The API may return an empty string `""` or `"{}"` as arguments. Always handle both:
```python
args = json.loads(arguments) if arguments and arguments.strip() not in ("", "{}") else {}
```

**Edge case: The same tool called multiple times with the same arguments.**
If an agent calls `search("Paris population")` twice in the same session, it's wasting tokens. Cache tool results within a single agent session:

```python
import hashlib

class CachedToolRegistry(ToolRegistry):
    def __init__(self):
        super().__init__()
        self._cache: dict[str, str] = {}

    def execute(self, name: str, arguments: str, **kwargs) -> str:
        cache_key = hashlib.md5(f"{name}:{arguments}".encode()).hexdigest()
        if cache_key in self._cache:
            return f"[Cached] {self._cache[cache_key]}"
        result = super().execute(name, arguments, **kwargs)
        self._cache[cache_key] = result
        return result
```

---

## Production Connection

At production scale, the tool execution layer is a microservice boundary:

- **Tool server**: Tools run in isolated containers with their own rate limits, credentials, and monitoring. The agent sends tool call requests over HTTP/gRPC; the tool server executes and responds.
- **Tool versioning**: Tools evolve. Version your tool schemas (`search_v2` vs. `search_v1`) and migrate agents gradually.
- **Secrets management**: Tool handlers need API keys (weather API, CRM credentials). Never hardcode keys; inject via environment variables or a secrets manager (AWS Secrets Manager, HashiCorp Vault).
- **Circuit breakers**: If a downstream API is down, fail fast and return a clear error rather than hanging. Implement a circuit breaker pattern (e.g., `tenacity` library with exponential backoff).

---

## Key Takeaways

- Function calling is a two-phase protocol: the LLM outputs a structured tool call; your code executes it and returns the result as a `role: "tool"` message.
- Tool descriptions are the single highest-leverage quality lever — invest time in specificity, examples, and "do not use for" clauses.
- Build a `ToolRegistry` to centralize schema exposure, argument validation, authorization, and error handling.
- Execute multiple parallel tool calls concurrently with `asyncio.gather` — saves significant latency for independent tools.
- Always validate inputs: check required fields, enum values, and numeric bounds in your own code — the LLM is not a validator.
- Classify tools by risk tier; require human approval for write and sensitive operations.
- Cache tool results within a session to prevent redundant expensive calls.

---

## Related Papers

| Paper | What it contributes | Link |
|-------|-------------------|------|
| Schick et al. (2023) — *Toolformer: Language Models Can Teach Themselves to Use Tools* | Self-supervised fine-tuning to integrate tool APIs into generation | [arxiv.org/abs/2302.04761](https://arxiv.org/abs/2302.04761) |
| Patil et al. (2023) — *Gorilla: Large Language Model Connected with Massive APIs* | Fine-tuned LLM for accurate API call generation across 1,600+ APIs | [arxiv.org/abs/2305.15334](https://arxiv.org/abs/2305.15334) |
| Qin et al. (2023) — *Tool Learning with Foundation Models* | Comprehensive survey of tool learning, including grounding, planning, and execution | [arxiv.org/abs/2304.08354](https://arxiv.org/abs/2304.08354) |
| Perez & Ribeiro (2022) — *Ignore Previous Prompt: Attack Techniques For Language Models* | Documents prompt injection attacks; motivation for tool output sanitization | [arxiv.org/abs/2211.09527](https://arxiv.org/abs/2211.09527) |

---

## Further Reading

- [OpenAI Function Calling Documentation](https://platform.openai.com/docs/guides/function-calling) — official API reference with examples
- [Anthropic Tool Use Guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — Claude's tool use API with structured output
- [Pydantic for Schema Validation](https://docs.pydantic.dev/) — use Pydantic models as type-safe tool argument validators

---

## Next Lesson

**[Lesson 5: Agent Memory Systems](05-Agent-Memory.md)** — Learn how agents maintain context across steps and sessions using short-term, long-term, and episodic memory architectures.
