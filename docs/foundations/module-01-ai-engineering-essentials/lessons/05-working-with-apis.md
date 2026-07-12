---
title: Working with LLM APIs — Production Guide
description: >-
  Master production-grade patterns for LLM API integration — provider abstractions,
  function calling, structured outputs, robust error handling, rate limiting,
  cost tracking, and observability for multi-provider systems
duration: 90 min
difficulty: intermediate
has_code: true
module: module-01
---
# Working with LLM APIs — Production Guide

## Prerequisites

- [Lesson 02: Your First AI Application](02-first-ai-application.md) — Chat Completions API basics
- [Lesson 03: Understanding Tokens and Costs](03-tokens-and-costs.md) — token counting, pricing

## What You'll Learn

| Objective | Why It Matters |
|-----------|---------------|
| Build a provider-agnostic abstraction layer | APIs change; your application logic should not |
| Implement function calling correctly | Tool use is the foundation of AI agents |
| Handle all error categories with the right strategy | Silent failures and cascading retries are production disasters |
| Implement rate limiting that respects provider tiers | Prevent billing overages and 429 storms |
| Trace and observe LLM calls | You cannot debug what you cannot see |
| Build a fallback system | Provider outages are real; multi-provider routing provides resilience |

---

## Why You Need an Abstraction Layer

The LLM API landscape changes rapidly. Models are deprecated, pricing changes, providers go down. If your application code calls `openai.chat.completions.create(model="gpt-4-turbo", ...)` in 20 places, every model change requires 20 edits and full regression testing.

A thin abstraction layer insulates your application logic from provider details:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Optional
import time
import os

@dataclass
class Message:
    role:    str   # "system", "user", "assistant", "tool"
    content: str
    name:    Optional[str] = None   # for tool results

@dataclass
class CompletionResponse:
    content:           str
    input_tokens:      int
    output_tokens:     int
    model:             str
    stop_reason:       str   # "stop", "length", "tool_calls"
    tool_calls:        list = field(default_factory=list)

class LLMProvider(ABC):
    """Base class for all LLM provider implementations."""

    @abstractmethod
    def complete(self, messages: list[Message], **kwargs) -> CompletionResponse:
        """Synchronous completion."""
        ...

    @abstractmethod
    def stream(self, messages: list[Message], **kwargs) -> Iterator[str]:
        """Streaming completion, yields token strings."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging."""
        ...
```

### OpenAI Implementation

```python
from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError

class OpenAIProvider(LLMProvider):

    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        self.model  = model
        self.client = OpenAI(api_key=api_key or os.environ["OPENAI_API_KEY"])

    @property
    def name(self) -> str:
        return f"openai/{self.model}"

    def _to_openai_messages(self, messages: list[Message]) -> list[dict]:
        """Convert our Message format to OpenAI's dict format."""
        result = []
        for m in messages:
            msg = {"role": m.role, "content": m.content}
            if m.name:
                msg["name"] = m.name
            result.append(msg)
        return result

    def complete(self, messages: list[Message], **kwargs) -> CompletionResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._to_openai_messages(messages),
            **kwargs
        )
        choice = response.choices[0]
        return CompletionResponse(
            content      = choice.message.content or "",
            input_tokens = response.usage.prompt_tokens,
            output_tokens= response.usage.completion_tokens,
            model        = self.model,
            stop_reason  = choice.finish_reason,
            tool_calls   = choice.message.tool_calls or [],
        )

    def stream(self, messages: list[Message], **kwargs) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self._to_openai_messages(messages),
            stream=True,
            **kwargs
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

class AnthropicProvider(LLMProvider):

    def __init__(self, model: str = "claude-3-5-haiku-20241022", api_key: str = None):
        from anthropic import Anthropic
        self.model  = model
        self.client = Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    @property
    def name(self) -> str:
        return f"anthropic/{self.model}"

    def complete(self, messages: list[Message], **kwargs) -> CompletionResponse:
        # Anthropic separates system message from the messages list
        system_content = next(
            (m.content for m in messages if m.role == "system"), ""
        )
        conv_messages = [
            {"role": m.role, "content": m.content}
            for m in messages if m.role != "system"
        ]

        response = self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.pop("max_tokens", 4096),
            system=system_content,
            messages=conv_messages,
            **kwargs
        )

        return CompletionResponse(
            content      = response.content[0].text,
            input_tokens = response.usage.input_tokens,
            output_tokens= response.usage.output_tokens,
            model        = self.model,
            stop_reason  = response.stop_reason,
        )

    def stream(self, messages: list[Message], **kwargs) -> Iterator[str]:
        system_content = next((m.content for m in messages if m.role == "system"), "")
        conv_messages  = [{"role": m.role, "content": m.content}
                          for m in messages if m.role != "system"]
        with self.client.messages.stream(
            model=self.model, max_tokens=4096,
            system=system_content, messages=conv_messages, **kwargs
        ) as stream:
            yield from stream.text_stream
```

---

## Function Calling (Tool Use)

Function calling lets the model request external data or actions. The model outputs a structured tool call (JSON), your code executes it, and the result is fed back to the model. This is the mechanism behind AI agents.

```python
import json
from openai import OpenAI

client = OpenAI()

# Step 1: Define your tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol, e.g. AAPL, MSFT"
                    }
                },
                "required": ["symbol"],
                "additionalProperties": False,
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A Python-evaluable math expression, e.g. '12 * 1.08'"
                    }
                },
                "required": ["expression"],
            }
        }
    }
]

# Step 2: Implement the actual functions
def get_stock_price(symbol: str) -> dict:
    """Real implementation would call a market data API."""
    mock_prices = {"AAPL": 185.30, "MSFT": 415.70, "GOOGL": 175.80}
    price = mock_prices.get(symbol.upper())
    if price is None:
        return {"error": f"Unknown symbol: {symbol}"}
    return {"symbol": symbol.upper(), "price": price, "currency": "USD"}

def calculate(expression: str) -> dict:
    """Safely evaluate a math expression."""
    try:
        result = eval(expression, {"__builtins__": {}})  # no builtins for safety
        return {"result": result, "expression": expression}
    except Exception as e:
        return {"error": str(e)}

AVAILABLE_TOOLS = {
    "get_stock_price": get_stock_price,
    "calculate": calculate,
}

# Step 3: The tool use loop
def run_with_tools(user_message: str, verbose: bool = True) -> str:
    """
    Execute a user request that may require tool calls.
    The model decides which tools to call, in what order.
    """
    messages = [
        {"role": "user", "content": user_message}
    ]

    while True:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",   # model decides whether to call tools
        )

        choice = response.choices[0]

        # If model wants to call tools, execute them
        if choice.finish_reason == "tool_calls":
            # Add the model's tool call message to history
            messages.append(choice.message)

            # Execute each requested tool call
            for tool_call in choice.message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                if verbose:
                    print(f"→ Calling {func_name}({func_args})")

                func = AVAILABLE_TOOLS.get(func_name)
                result = func(**func_args) if func else {"error": f"Unknown tool: {func_name}"}

                if verbose:
                    print(f"← Result: {result}")

                # Add tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": json.dumps(result),
                })

        else:
            # Model gave a final response (not a tool call)
            return choice.message.content

# Test
result = run_with_tools("What is the total value of 5 shares of AAPL and 3 shares of MSFT?")
print(f"\nFinal answer: {result}")
# → Calling get_stock_price({'symbol': 'AAPL'})
# ← Result: {'symbol': 'AAPL', 'price': 185.3, 'currency': 'USD'}
# → Calling get_stock_price({'symbol': 'MSFT'})
# ← Result: {'symbol': 'MSFT', 'price': 415.7, 'currency': 'USD'}
# → Calling calculate({'expression': '5 * 185.3 + 3 * 415.7'})
# ← Result: {'result': 2173.6, 'expression': '5 * 185.3 + 3 * 415.7'}
# Final answer: The total value is $2,173.60 (5 AAPL at $185.30 + 3 MSFT at $415.70).
```

---

## Comprehensive Error Handling

LLM APIs fail in predictable ways. Each failure category requires a different response:

```python
import time
import logging
from openai import (
    OpenAI, RateLimitError, APIConnectionError,
    APIStatusError, APITimeoutError
)

logger = logging.getLogger(__name__)

class APICallError(Exception):
    """Raised when all retries are exhausted or the error is non-retryable."""
    def __init__(self, message: str, original_error: Exception):
        super().__init__(message)
        self.original_error = original_error

def call_with_retry(
    client: OpenAI,
    messages: list,
    model: str = "gpt-4o-mini",
    max_retries: int = 4,
    base_delay: float = 1.0,
    timeout: float = 30.0,
    **kwargs,
) -> str:
    """
    Call the OpenAI API with exponential backoff retry.

    Retry strategy:
    - RateLimitError (429): exponential backoff with jitter, up to max_retries
    - APIConnectionError: retry immediately, up to max_retries
    - APITimeoutError: retry with longer timeout, up to 2 attempts
    - APIStatusError 5xx: retry once (server error, possibly transient)
    - APIStatusError 4xx: raise immediately (client error, retry won't help)
    """
    import random

    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=timeout,
                **kwargs,
            )
            return response.choices[0].message.content

        except RateLimitError as e:
            last_error = e
            if attempt == max_retries - 1:
                break
            # Exponential backoff with jitter (avoids thundering herd)
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"Rate limited (attempt {attempt+1}/{max_retries}). Waiting {delay:.1f}s")
            time.sleep(delay)

        except APIConnectionError as e:
            last_error = e
            if attempt == max_retries - 1:
                break
            logger.warning(f"Connection error (attempt {attempt+1}/{max_retries}). Retrying...")
            time.sleep(1)

        except APITimeoutError as e:
            last_error = e
            if attempt >= 2:
                break
            logger.warning(f"Timeout (attempt {attempt+1}). Retrying with longer timeout...")
            timeout = min(timeout * 2, 120)  # double timeout, max 120s

        except APIStatusError as e:
            if e.status_code >= 500:
                # Server error — might be transient
                last_error = e
                if attempt == max_retries - 1:
                    break
                logger.warning(f"Server error {e.status_code} (attempt {attempt+1}). Retrying...")
                time.sleep(base_delay * (2 ** attempt))
            else:
                # 4xx — client error (bad request, invalid model, etc.)
                # Retrying is futile; raise immediately
                raise APICallError(
                    f"Non-retryable API error {e.status_code}: {e.message}",
                    original_error=e
                )

    raise APICallError(
        f"API call failed after {max_retries} attempts",
        original_error=last_error
    )
```

---

## Rate Limiting

API providers enforce rate limits at two levels:
- **RPM**: requests per minute (usually lower tier)
- **TPM**: tokens per minute (the harder constraint at scale)

```python
import threading
import time
from collections import deque

class TokenBucketRateLimiter:
    """
    Token bucket algorithm for rate limiting API calls.

    Allows short bursts while enforcing average rate limits.
    Thread-safe for use in concurrent applications.
    """

    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        self.rpm    = requests_per_minute
        self.tpm    = tokens_per_minute
        self.lock   = threading.Lock()

        # Track request timestamps in rolling window
        self.request_times: deque = deque()
        self.token_counts:  deque = deque()   # (timestamp, token_count) pairs

    def _clean_old_entries(self, window_seconds: float = 60.0):
        """Remove entries older than the window."""
        cutoff = time.monotonic() - window_seconds
        while self.request_times and self.request_times[0] < cutoff:
            self.request_times.popleft()
        while self.token_counts and self.token_counts[0][0] < cutoff:
            self.token_counts.popleft()

    def acquire(self, estimated_tokens: int = 1000, block: bool = True) -> bool:
        """
        Acquire permission to make an API call.

        estimated_tokens: estimated total tokens (input + output) for this call
        block: if True, wait until rate limit allows; if False, return False immediately
        """
        while True:
            with self.lock:
                self._clean_old_entries()

                current_requests = len(self.request_times)
                current_tokens   = sum(t for _, t in self.token_counts)

                if current_requests < self.rpm and current_tokens + estimated_tokens <= self.tpm:
                    # Approved — record this call
                    now = time.monotonic()
                    self.request_times.append(now)
                    self.token_counts.append((now, estimated_tokens))
                    return True

            if not block:
                return False

            time.sleep(0.1)  # wait and retry

# Usage
limiter = TokenBucketRateLimiter(
    requests_per_minute=100,   # tier 1 OpenAI limit
    tokens_per_minute=40_000,  # tier 1 limit
)

def rate_limited_call(messages: list, estimated_tokens: int = 500) -> str:
    limiter.acquire(estimated_tokens=estimated_tokens)
    return call_with_retry(client, messages)
```

---

## Observability and Logging

You cannot debug production AI systems without traces. At minimum, log: model, tokens, latency, cost, and a sampling of inputs/outputs.

```python
import time
import uuid
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field

@dataclass
class LLMTrace:
    """One complete LLM API call, with timing and cost."""
    trace_id:       str
    provider:       str
    model:          str
    input_tokens:   int    = 0
    output_tokens:  int    = 0
    latency_ms:     float  = 0
    cost_usd:       float  = 0
    error:          str    = ""
    tags:           dict   = field(default_factory=dict)

    def log(self):
        logging.info(
            "LLM call",
            extra={
                "trace_id":     self.trace_id,
                "model":        self.model,
                "input_tokens": self.input_tokens,
                "output_tokens":self.output_tokens,
                "latency_ms":   round(self.latency_ms, 1),
                "cost_usd":     round(self.cost_usd, 6),
                "error":        self.error,
                **self.tags,
            }
        )

PRICING = {
    "gpt-4o":      (2.50, 10.00),
    "gpt-4o-mini": (0.15,  0.60),
    "claude-3-5-haiku-20241022": (0.25, 1.25),
}

@contextmanager
def traced_llm_call(provider: str, model: str, **tags):
    """
    Context manager that creates an LLMTrace, times the call, and logs it.
    Usage:
        with traced_llm_call("openai", "gpt-4o-mini", feature="summarization") as trace:
            response = client.chat.completions.create(...)
            trace.input_tokens  = response.usage.prompt_tokens
            trace.output_tokens = response.usage.completion_tokens
    """
    trace = LLMTrace(
        trace_id = str(uuid.uuid4())[:8],
        provider = provider,
        model    = model,
        tags     = tags,
    )
    start = time.monotonic()
    try:
        yield trace
    except Exception as e:
        trace.error = str(e)
        raise
    finally:
        trace.latency_ms = (time.monotonic() - start) * 1000
        in_p, out_p = PRICING.get(model, (0, 0))
        trace.cost_usd = (trace.input_tokens * in_p + trace.output_tokens * out_p) / 1_000_000
        trace.log()

# Example usage
with traced_llm_call("openai", "gpt-4o-mini", feature="summarization", user_id="u_123") as trace:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Summarize the transformer paper."}]
    )
    trace.input_tokens  = response.usage.prompt_tokens
    trace.output_tokens = response.usage.completion_tokens
```

---

## Building a Resilient Multi-Provider Client

For production systems, route to a fallback provider when the primary is unavailable:

```python
class ResilientLLMClient:
    """
    Multi-provider client with automatic fallback.
    Tries providers in order; uses the first that succeeds.
    """

    def __init__(self, providers: list[LLMProvider]):
        self.providers = providers   # ordered by preference

    def complete(self, messages: list[Message], **kwargs) -> CompletionResponse:
        last_error = None

        for provider in self.providers:
            try:
                with traced_llm_call(provider.name.split("/")[0],
                                      provider.name.split("/")[1]) as trace:
                    response = provider.complete(messages, **kwargs)
                    trace.input_tokens  = response.input_tokens
                    trace.output_tokens = response.output_tokens
                return response

            except Exception as e:
                last_error = e
                logging.warning(f"Provider {provider.name} failed: {e}. Trying next provider.")
                continue

        raise RuntimeError(f"All providers failed. Last error: {last_error}")

# Usage: OpenAI primary, Anthropic fallback
client_resilient = ResilientLLMClient(providers=[
    OpenAIProvider(model="gpt-4o-mini"),
    AnthropicProvider(model="claude-3-5-haiku-20241022"),
])

messages = [Message(role="user", content="What is the capital of France?")]
response = client_resilient.complete(messages)
print(response.content)
```

---

## Edge Cases and Misconceptions

**"The OpenAI SDK handles retries automatically."** The official SDK (v1.0+) does implement basic retries with `max_retries` parameter. However, the default retry logic does not implement jitter (can cause thundering herds), does not handle token-per-minute limits, and does not distinguish between retryable and non-retryable 4xx errors. Custom retry logic gives you more control.

**"Streaming is always better than synchronous."** Streaming improves perceived latency but adds complexity. For batch processing or when the complete response is needed before any processing (e.g., parsing JSON), synchronous calls are simpler. Use streaming for chat interfaces and long responses where the user benefits from early tokens.

**"Function calling is safe — the model cannot execute code directly."** Function calling lets the model request execution of *your* functions. The model cannot directly execute arbitrary code (unless you wire up a code interpreter tool). The security boundary is in your tool implementations: validate inputs, restrict permissions, and never pass raw model output directly to `eval()`.

**"If the API returns an error, retry immediately."** Immediate retry on rate limit errors makes the situation worse — the rate limiter will reject your requests faster, and you may exhaust retries before the window clears. Always wait before retrying rate limit errors (exponential backoff with jitter is the standard).

---

## Key Takeaways

- A thin abstraction layer (base class with provider implementations) protects application logic from API changes
- Function calling works as a loop: model requests tool calls → you execute them → feed results back → model decides whether to call more tools or respond
- Error handling requires distinguishing error categories: 429 (backoff+retry), 5xx (retry once), 4xx (raise immediately)
- Rate limiting at two levels: RPM and TPM; the token-per-minute limit is often the binding constraint at scale
- Observability — logging model, tokens, latency, cost per call — is essential for debugging and cost management
- Multi-provider fallback provides resilience; route by cost, capability, or availability

---

## Further Reading

- [OpenAI API Reference](https://platform.openai.com/docs/api-reference) — complete parameter documentation
- [Anthropic API Reference](https://docs.anthropic.com/en/api/getting-started) — Claude API documentation
- [LiteLLM](https://github.com/BerriAI/litellm) — open-source unified interface for 100+ LLMs; consider it instead of building your own abstraction
- [Helicone](https://www.helicone.ai/) — production LLM observability, cost tracking, and caching as a service
- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) — 50% cost reduction for async workloads

---

**Module 01 complete.** You have the essentials to build production AI applications. Next, Module 05 gives you the mathematical foundations of neural networks that power the models you have been using.

**Next:** [Module 05: Neural Networks — Introduction to Neural Networks](../../module-05-neural-networks-deep-learning-fundamentals/lessons/01-introduction-to-neural-networks.md)
