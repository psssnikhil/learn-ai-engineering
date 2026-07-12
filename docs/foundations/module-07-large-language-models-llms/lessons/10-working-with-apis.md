---
title: Working with LLM APIs
description: >-
  Production patterns for calling OpenAI, Anthropic, and local LLM APIs:
  streaming, batching, retries, cost tracking, prompt caching, and async
  concurrency.
duration: 60 min
difficulty: intermediate
has_code: true
module: module-07
objectives:
  - Call OpenAI and Anthropic APIs with proper error handling and retries
  - Implement streaming responses for real-time UI
  - Use structured output (function calling / JSON mode) reliably
  - Track costs per request and set budget limits
  - Run local models with Ollama and vLLM
  - Implement async batching for throughput-heavy workloads
---

# Working with LLM APIs

## Prerequisites

- [Lesson 01: Introduction to LLMs](./01-introduction-to-llms.md) — tokens, temperature
- [Lesson 04: Tokenization](./04-tokenization.md) — counting tokens for cost estimation

## What You'll Learn

This lesson bridges the gap between model theory and production usage. You'll learn the patterns used in production LLM applications: how to handle errors gracefully, stream responses, extract structured data reliably, track costs, and run local models.

---

## Basic API Calls

### OpenAI

```python
import openai
from openai import OpenAI

client = OpenAI()   # reads OPENAI_API_KEY from env


def chat(
    user_message:   str,
    system_message: str = "You are a helpful assistant.",
    model:          str = "gpt-4o",
    temperature:    float = 0.7,
    max_tokens:     int = 1024,
) -> str:
    """
    Basic OpenAI chat completion.

    The messages list is the full conversation context.
    Add earlier turns to enable multi-turn dialogue.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user",   "content": user_message},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content


# Usage
answer = chat("What is the capital of France?")
print(answer)  # "The capital of France is Paris."
```

### Anthropic (Claude)

```python
import anthropic

client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env


def chat_claude(
    user_message:   str,
    system_message: str = "You are a helpful assistant.",
    model:          str = "claude-opus-4-5",
    max_tokens:     int = 1024,
) -> str:
    """
    Anthropic differs from OpenAI in two ways:
    1. System message is a separate parameter, not a list item
    2. max_tokens is required (no default)
    """
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_message,
        messages=[{"role": "user", "content": user_message}],
    )

    return message.content[0].text


# Multi-turn conversation
def multi_turn_claude(messages: list[dict], system: str = "") -> str:
    """
    Messages format: [{"role": "user"|"assistant", "content": str}, ...]
    """
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2048,
        system=system,
        messages=messages,
    )
    return response.content[0].text
```

---

## Error Handling and Retries

Production API calls fail for many reasons: rate limits, network timeouts, server errors, and context length exceeded. A robust wrapper handles all of these:

```python
import time
import logging
from typing import Any
from openai import OpenAI, RateLimitError, APITimeoutError, APIStatusError

logger = logging.getLogger(__name__)


def chat_with_retry(
    messages:    list[dict],
    model:       str   = "gpt-4o",
    max_retries: int   = 5,
    base_delay:  float = 1.0,    # seconds
    max_delay:   float = 60.0,
    **kwargs,
) -> str:
    """
    Retry OpenAI calls with exponential backoff.

    Retry on:
    - RateLimitError (429): wait and retry
    - APITimeoutError: transient, safe to retry
    - 500, 502, 503: server errors, safe to retry

    Do NOT retry on:
    - 400 (invalid request): fix the request
    - 401 (auth error): fix the API key
    - 404 (model not found): fix the model name
    - 422 (context length exceeded): reduce prompt
    """
    client = OpenAI()
    last_exception = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs,
            )
            return response.choices[0].message.content

        except RateLimitError as e:
            # Check for retry-after header
            retry_after = float(e.response.headers.get("retry-after", base_delay * (2 ** attempt)))
            delay = min(retry_after, max_delay)
            logger.warning(f"Rate limited. Waiting {delay:.1f}s (attempt {attempt+1}/{max_retries})")
            time.sleep(delay)
            last_exception = e

        except APITimeoutError as e:
            delay = min(base_delay * (2 ** attempt), max_delay)
            logger.warning(f"Timeout. Waiting {delay:.1f}s (attempt {attempt+1}/{max_retries})")
            time.sleep(delay)
            last_exception = e

        except APIStatusError as e:
            if e.status_code in (500, 502, 503):
                delay = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(f"Server error {e.status_code}. Waiting {delay:.1f}s")
                time.sleep(delay)
                last_exception = e
            else:
                # Client error — do not retry
                logger.error(f"Client error {e.status_code}: {e.message}")
                raise

    raise last_exception
```

---

## Streaming Responses

For interactive applications, streaming shows tokens as they're generated rather than waiting for the full response:

```python
import sys
from openai import OpenAI


def stream_chat(
    messages: list[dict],
    model:    str = "gpt-4o",
) -> str:
    """
    Stream tokens as they're generated.

    Streams via server-sent events. Each delta contains 0+ tokens.
    Useful for: chatbots, long generation, progress indication.

    Returns the full response text for logging/storage.
    """
    client = OpenAI()
    full_response = []

    with client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    ) as stream:
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                print(delta.content, end="", flush=True)  # real-time output
                full_response.append(delta.content)

    print()  # newline after stream ends
    return "".join(full_response)


# Async streaming for FastAPI / asyncio applications
import asyncio
from openai import AsyncOpenAI


async def astream_chat(
    messages: list[dict],
    model:    str = "gpt-4o",
):
    """
    Async streaming — use with async web frameworks like FastAPI.

    Yields each text chunk as it arrives.
    """
    client = AsyncOpenAI()

    async with client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
    ) as stream:
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content


# FastAPI example
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/chat/stream")
async def stream_endpoint(prompt: str):
    async def generate():
        async for chunk in astream_chat([{"role": "user", "content": prompt}]):
            yield f"data: {chunk}\n\n"   # Server-Sent Events format

    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## Structured Output (JSON Mode and Function Calling)

Extracting structured data reliably from LLMs:

```python
import json
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()


# Method 1: JSON mode (simpler but no schema validation)
def extract_json(text: str, schema_description: str) -> dict:
    """
    Ask the model to output JSON matching a schema.
    JSON mode guarantees valid JSON but doesn't validate against a schema.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"Extract information as JSON. Schema: {schema_description}",
            },
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},   # guarantees valid JSON output
        temperature=0,                              # deterministic for extraction
    )
    return json.loads(response.choices[0].message.content)


# Method 2: Structured output with Pydantic (type-safe)
from openai.lib._pydantic import to_strict_json_schema


class PersonInfo(BaseModel):
    name:       str
    age:        int | None
    occupation: str | None
    location:   str | None


def extract_person_info(text: str) -> PersonInfo:
    """
    Structured output: model returns JSON that Pydantic validates.
    Guarantees the response matches the PersonInfo schema.
    """
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Extract person information from the text."},
            {"role": "user", "content": text},
        ],
        response_format=PersonInfo,
        temperature=0,
    )
    return response.choices[0].message.parsed


# Method 3: Function calling for tool use
import json


def call_with_tools(user_query: str) -> str:
    """
    Function calling: model decides which tool to call and with what arguments.
    More flexible than JSON mode — handles multi-step tool use.
    """
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search the web for current information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "max_results": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Evaluate a mathematical expression",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "Math expression to evaluate"},
                    },
                    "required": ["expression"],
                },
            },
        },
    ]

    messages = [{"role": "user", "content": user_query}]

    # First call: let model decide which tool to call
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )

    # Check if model wants to call a tool
    tool_calls = response.choices[0].message.tool_calls
    if not tool_calls:
        return response.choices[0].message.content

    # Execute tool calls
    messages.append(response.choices[0].message)

    for call in tool_calls:
        tool_name = call.function.name
        tool_args = json.loads(call.function.arguments)

        # Dispatch to actual function
        if tool_name == "search_web":
            result = f"Search results for '{tool_args['query']}': [mock results]"
        elif tool_name == "calculate":
            result = str(eval(tool_args["expression"]))  # noqa: eval OK in demo
        else:
            result = "Unknown tool"

        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": result,
        })

    # Second call: get final response with tool results
    final = client.chat.completions.create(model="gpt-4o", messages=messages)
    return final.choices[0].message.content
```

---

## Cost Tracking

```python
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class CostConfig:
    """Pricing per 1M tokens (in USD) as of mid-2026."""
    input_cost_per_1m:   float
    output_cost_per_1m:  float


MODEL_COSTS: dict[str, CostConfig] = {
    "gpt-4o":             CostConfig(2.50,  10.00),
    "gpt-4o-mini":        CostConfig(0.15,   0.60),
    "claude-opus-4-5":    CostConfig(15.00,  75.00),
    "claude-sonnet-4-5":  CostConfig(3.00,   15.00),
    "gpt-3.5-turbo":      CostConfig(0.50,   1.50),
}


class CostTracker:
    """Thread-safe cost tracker for API budget management."""

    def __init__(self, budget_usd: float = float("inf")):
        self.budget_usd = budget_usd
        self._lock = Lock()
        self.total_cost: float = 0.0
        self.total_input_tokens:  int = 0
        self.total_output_tokens: int = 0
        self.request_count:       int = 0

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Record usage and return cost for this request."""
        config = MODEL_COSTS.get(model)
        if not config:
            return 0.0

        request_cost = (
            input_tokens  * config.input_cost_per_1m  / 1_000_000 +
            output_tokens * config.output_cost_per_1m / 1_000_000
        )

        with self._lock:
            self.total_cost          += request_cost
            self.total_input_tokens  += input_tokens
            self.total_output_tokens += output_tokens
            self.request_count       += 1

            if self.total_cost > self.budget_usd:
                raise BudgetExceededError(
                    f"Budget exceeded: ${self.total_cost:.4f} > ${self.budget_usd:.2f}"
                )

        return request_cost

    def summary(self) -> dict:
        return {
            "total_cost_usd":    round(self.total_cost, 4),
            "requests":          self.request_count,
            "input_tokens":      self.total_input_tokens,
            "output_tokens":     self.total_output_tokens,
            "avg_cost_per_req":  round(self.total_cost / max(1, self.request_count), 6),
        }


class BudgetExceededError(Exception):
    pass


# Instrumented client
tracker = CostTracker(budget_usd=10.0)


def tracked_chat(messages: list[dict], model: str = "gpt-4o") -> str:
    client = OpenAI()
    response = client.chat.completions.create(model=model, messages=messages)

    usage = response.usage
    cost  = tracker.record(model, usage.prompt_tokens, usage.completion_tokens)

    logger.info(f"Request cost: ${cost:.6f} | Total: ${tracker.total_cost:.4f}")
    return response.choices[0].message.content
```

---

## Async Batching for Throughput

```python
import asyncio
from openai import AsyncOpenAI


async def process_batch(
    prompts:     list[str],
    model:       str = "gpt-4o-mini",
    concurrency: int = 20,              # max parallel requests
) -> list[str]:
    """
    Process many prompts concurrently with controlled parallelism.

    Using asyncio.Semaphore to limit concurrent requests to stay
    within rate limits (default: 60 RPM for tier-1 accounts).
    """
    client  = AsyncOpenAI()
    sem     = asyncio.Semaphore(concurrency)
    results = [None] * len(prompts)

    async def process_one(i: int, prompt: str) -> None:
        async with sem:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=512,
                )
                results[i] = response.choices[0].message.content
            except Exception as e:
                logger.error(f"Request {i} failed: {e}")
                results[i] = None

    await asyncio.gather(*[process_one(i, p) for i, p in enumerate(prompts)])
    return results


# Usage
async def main():
    prompts = [f"What is {n}²?" for n in range(1, 101)]  # 100 math questions
    answers = await process_batch(prompts, concurrency=20)
    print(f"Processed {len(answers)} prompts")

asyncio.run(main())
```

---

## Running Local Models

```python
import requests


class OllamaClient:
    """
    Client for Ollama — run models locally (Llama-3, Mistral, etc.).

    Install: curl -fsSL https://ollama.ai/install.sh | sh
    Pull model: ollama pull llama3
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    def chat(
        self,
        messages: list[dict],
        model:    str   = "llama3",
        stream:   bool  = False,
    ) -> str:
        """Compatible with OpenAI chat format."""
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model":    model,
                "messages": messages,
                "stream":   stream,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["message"]["content"]


class VLLMClient:
    """
    vLLM — high-throughput local inference with OpenAI-compatible API.

    Start: python -m vllm.entrypoints.openai.api_server \
                --model meta-llama/Llama-3-8B-Instruct \
                --port 8000

    vLLM uses PagedAttention for efficient KV cache management,
    enabling 10–20× higher throughput than naive serving.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        # vLLM is OpenAI-compatible, so just use the OpenAI client
        self.client = OpenAI(api_key="EMPTY", base_url=f"{base_url}/v1")

    def chat(self, messages: list[dict], model: str = "meta-llama/Llama-3-8B-Instruct") -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return response.choices[0].message.content
```

---

## Prompt Caching (Anthropic)

Claude supports caching prefix prompts — if you send the same system prompt repeatedly, cached tokens are 90% cheaper:

```python
def chat_with_cache(
    user_message:   str,
    system_message: str,
    cached_context: str = "",   # long document or examples to cache
) -> str:
    """
    Use Anthropic prompt caching for repeated context.

    Cache rule: mark prefix content with "cache_control".
    First call: full price. Subsequent calls within 5 minutes: 10% of input price.

    Best for: RAG with long retrieved documents, few-shot examples.
    """
    messages = []

    if cached_context:
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": cached_context,
                    "cache_control": {"type": "ephemeral"},  # mark for caching
                },
                {"type": "text", "text": user_message},
            ],
        })
    else:
        messages.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=1024,
        system=system_message,
        messages=messages,
    )

    # Check cache usage in response
    usage = response.usage
    print(f"Input tokens: {usage.input_tokens}, "
          f"Cache write: {getattr(usage, 'cache_creation_input_tokens', 0)}, "
          f"Cache read: {getattr(usage, 'cache_read_input_tokens', 0)}")

    return response.content[0].text
```

---

## Edge Cases & Misconceptions

!!! warning "Misconception: Temperature=0 is deterministic"
    Temperature=0 is nearly but not perfectly deterministic due to floating-point parallelism differences across GPU runs. For truly reproducible outputs, also set `seed` (OpenAI) and use the exact same model version.

!!! note "Context window ≠ memory"
    LLMs have no persistent memory between calls. The full conversation history must be sent with each request. At 128K tokens ($0.32/call for GPT-4o), long conversations become expensive. Implement context summarization for multi-turn applications.

!!! warning "Rate limits are per-organization, not per-key"
    If multiple services share an organization's API key, they share the same rate limit. Use per-project keys (OpenAI project API keys) to isolate limits and track costs per product.

!!! note "JSON mode doesn't validate against a schema"
    `response_format: {"type": "json_object"}` guarantees valid JSON but not that it matches your expected schema. Use `client.beta.chat.completions.parse()` with a Pydantic model for schema validation.

---

## Production Checklist

```python
PRODUCTION_CHECKLIST = {
    "error_handling": [
        "Retry on 429, 500, 502, 503 with exponential backoff",
        "Don't retry on 400, 401, 404 (client errors)",
        "Set timeout (60–120s for long completions)",
        "Log all errors with request ID for debugging",
    ],
    "cost_control": [
        "Track input + output tokens per request",
        "Set per-request max_tokens limit",
        "Use gpt-4o-mini for classification/extraction tasks",
        "Cache repeated system prompts with Anthropic caching",
        "Implement hard budget limit with circuit breaker",
    ],
    "latency": [
        "Stream for interactive UI (first token < 500ms)",
        "Use async for batch processing (20× throughput)",
        "Cache deterministic outputs (temperature=0 + seed)",
        "Consider local models (Ollama/vLLM) for high-volume tasks",
    ],
    "reliability": [
        "Validate structured outputs with Pydantic",
        "Have fallback model (gpt-4o-mini if gpt-4o unavailable)",
        "Monitor: success rate, latency p99, cost per request",
        "Set up alerts for latency spikes or cost anomalies",
    ],
}
```

---

## Building a Robust LLM Client

Production systems need more than just basic API calls. Here's a complete production-ready client:

```python
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator

from openai import AsyncOpenAI, RateLimitError, APITimeoutError, APIStatusError

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Typed response from LLM API."""
    content:       str
    model:         str
    input_tokens:  int
    output_tokens: int
    latency_ms:    float
    cost_usd:      float


class ProductionLLMClient:
    """
    Battle-hardened async LLM client with:
    - Automatic retries with exponential backoff
    - Cost tracking
    - Response caching for idempotent calls
    - Structured output parsing
    - Streaming support
    """

    MODEL_COSTS = {  # (input_per_1m, output_per_1m) in USD
        "gpt-4o":          (2.50, 10.00),
        "gpt-4o-mini":     (0.15,  0.60),
        "claude-opus-4-5": (15.00, 75.00),
    }

    def __init__(
        self,
        budget_usd:    float = float("inf"),
        max_retries:   int   = 5,
        cache_enabled: bool  = True,
    ):
        self.client        = AsyncOpenAI()
        self.budget        = budget_usd
        self.spent         = 0.0
        self.max_retries   = max_retries
        self._cache: dict  = {} if cache_enabled else None
        self.request_count = 0

    def _compute_cost(self, model: str, input_tok: int, output_tok: int) -> float:
        costs = self.MODEL_COSTS.get(model, (0, 0))
        return input_tok * costs[0] / 1e6 + output_tok * costs[1] / 1e6

    def _cache_key(self, model: str, messages: list[dict], **kwargs) -> str:
        """Deterministic cache key for idempotent calls (temperature=0)."""
        return json.dumps({"model": model, "messages": messages, **kwargs}, sort_keys=True)

    async def chat(
        self,
        messages:  list[dict],
        model:     str   = "gpt-4o",
        use_cache: bool  = False,
        **kwargs,
    ) -> LLMResponse:
        """
        Async chat completion with retry and cost tracking.
        use_cache=True only for deterministic calls (temperature=0).
        """
        # Budget check
        if self.spent >= self.budget:
            raise RuntimeError(f"Budget exhausted: ${self.spent:.4f} >= ${self.budget}")

        # Cache check
        if use_cache and self._cache is not None:
            key = self._cache_key(model, messages, **kwargs)
            if key in self._cache:
                logger.debug("Cache hit")
                return self._cache[key]

        # Retry loop
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                t0 = time.monotonic()
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **kwargs,
                )
                latency_ms = (time.monotonic() - t0) * 1000

                usage = response.usage
                cost  = self._compute_cost(model, usage.prompt_tokens, usage.completion_tokens)
                self.spent += cost
                self.request_count += 1

                result = LLMResponse(
                    content       = response.choices[0].message.content,
                    model         = model,
                    input_tokens  = usage.prompt_tokens,
                    output_tokens = usage.completion_tokens,
                    latency_ms    = latency_ms,
                    cost_usd      = cost,
                )

                # Store in cache if deterministic
                if use_cache and self._cache is not None:
                    key = self._cache_key(model, messages, **kwargs)
                    self._cache[key] = result

                return result

            except RateLimitError as e:
                wait = float(e.response.headers.get("retry-after", 2 ** attempt))
                logger.warning(f"Rate limit, waiting {wait:.1f}s (attempt {attempt+1})")
                await asyncio.sleep(wait)
                last_exc = e
            except APITimeoutError as e:
                await asyncio.sleep(2 ** attempt)
                last_exc = e
            except APIStatusError as e:
                if e.status_code in (500, 502, 503):
                    await asyncio.sleep(2 ** attempt)
                    last_exc = e
                else:
                    raise  # don't retry client errors

        raise last_exc

    def stats(self) -> dict:
        return {
            "total_cost_usd": round(self.spent, 4),
            "requests":       self.request_count,
            "cache_hits":     len(self._cache) if self._cache else 0,
        }
```

---

## Key Takeaways

1. **Retry with exponential backoff** on rate limits and server errors; never retry on client errors (400, 401).
2. **Streaming** significantly improves perceived latency for interactive applications — first token appears in <500ms even for long responses.
3. **Structured output** (`response_format` + Pydantic) is more reliable than prompt-based JSON extraction for consistent schemas.
4. **Track costs per request** using `response.usage.prompt_tokens` and `response.usage.completion_tokens` — costs scale quickly at production volume.
5. **Async batching** with a semaphore achieves 10–20× throughput for offline processing compared to sequential calls.
6. **Local models** (Ollama, vLLM) eliminate API costs for high-volume or sensitive-data workloads.

---

## Further Reading

- [OpenAI API reference](https://platform.openai.com/docs/api-reference) — complete API docs
- [Anthropic API docs](https://docs.anthropic.com) — including prompt caching
- [vLLM documentation](https://docs.vllm.ai) — high-throughput local serving
- [LiteLLM](https://github.com/BerriAI/litellm) — unified interface for 100+ LLM APIs
- [Ollama](https://ollama.ai) — simplest way to run local models

---

## Module Summary

You have completed Module 07: Large Language Models. You now understand:

- How LLMs generate text through next-token prediction
- The architecture evolution from Transformer → GPT → modern LLMs
- Pre-training objectives, tokenization, and embeddings
- Fine-tuning with LoRA/QLoRA, instruction tuning, and RLHF
- How to use LLM APIs effectively in production

**Next:** [M09 · RAG — Retrieval Augmented Generation](../../../build/module-09-rag-retrieval-augmented-generation/index.md)
