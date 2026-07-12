---
title: API Design for AI Services
description: >-
  Learn how to design robust APIs for LLM-powered services including streaming,
  error handling, rate limiting, and schema validation
duration: 30 min
difficulty: intermediate
has_code: false
objectives:
  - Design a REST API endpoint for an LLM-powered service
  - Implement streaming responses with Server-Sent Events
  - Add rate limiting and usage tracking to an AI API
  - Handle LLM-specific errors gracefully
  - Version your AI API for backward compatibility
---
# API Design for AI Services

## What You'll Learn

By the end of this lesson, you'll understand:
- How AI API design differs from traditional API design
- Streaming responses with Server-Sent Events (SSE)
- Rate limiting and usage-based billing
- Error handling for non-deterministic systems
- API versioning strategies

**Time to Complete**: 30 minutes
**Difficulty**: Intermediate

---

## Why AI APIs Are Different

Traditional APIs return predictable, fast responses. AI APIs introduce new challenges:

| Challenge | Traditional API | AI API |
|-----------|----------------|--------|
| Latency | 10-200ms | 500ms-30s |
| Response size | Predictable | Variable (10-10,000 tokens) |
| Errors | Deterministic | Non-deterministic (hallucinations) |
| Cost | Negligible per request | $0.001-$0.10+ per request |
| Rate limits | Your infrastructure | Provider-imposed |

These differences require specific API design patterns.

---

## Streaming Responses

For long-running LLM calls, streaming improves perceived performance dramatically. Instead of waiting 5 seconds for a complete response, users see tokens arrive in real time.

### Server-Sent Events (SSE)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from openai import OpenAI
import json

app = FastAPI()
client = OpenAI()

@app.post("/api/chat")
async def chat(request: dict):
    """Stream LLM responses using Server-Sent Events."""
    messages = request.get("messages", [])
    model = request.get("model", "gpt-4o-mini")

    async def generate():
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                data = json.dumps({
                    "content": chunk.choices[0].delta.content,
                    "done": False
                })
                yield f"data: {data}

"

        yield f"data: {json.dumps({'content': '', 'done': True})}

"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )
```

### Client-Side Consumption

```javascript
async function streamChat(messages) {
  const response = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ messages })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let fullResponse = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const text = decoder.decode(value);
    const lines = text.split('
').filter(line => line.startsWith('data: '));

    for (const line of lines) {
      const data = JSON.parse(line.slice(6));
      if (data.done) return fullResponse;
      fullResponse += data.content;
      updateUI(fullResponse);  // Render incrementally
    }
  }
}
```

---

## Rate Limiting

AI APIs are expensive. Rate limiting protects both your budget and your provider quotas.

### Token Bucket Rate Limiter

```python
import time
from collections import defaultdict

class TokenBucketRateLimiter:
    def __init__(self, requests_per_minute: int = 60,
                 tokens_per_minute: int = 100000):
        self.rpm_limit = requests_per_minute
        self.tpm_limit = tokens_per_minute
        self.buckets = defaultdict(lambda: {
            "requests": requests_per_minute,
            "tokens": tokens_per_minute,
            "last_refill": time.time()
        })

    def check(self, api_key: str, estimated_tokens: int = 1000) -> dict:
        """Check if a request is allowed under rate limits."""
        bucket = self.buckets[api_key]
        self._refill(bucket)

        if bucket["requests"] < 1:
            return {"allowed": False, "reason": "Request rate limit exceeded",
                    "retry_after": 60}
        if bucket["tokens"] < estimated_tokens:
            return {"allowed": False, "reason": "Token rate limit exceeded",
                    "retry_after": 60}

        bucket["requests"] -= 1
        bucket["tokens"] -= estimated_tokens
        return {"allowed": True}

    def _refill(self, bucket):
        now = time.time()
        elapsed = now - bucket["last_refill"]
        if elapsed >= 60:
            bucket["requests"] = self.rpm_limit
            bucket["tokens"] = self.tpm_limit
            bucket["last_refill"] = now
```

### Usage Tracking

```python
class UsageTracker:
    def __init__(self):
        self.usage = defaultdict(lambda: {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0
        })

    def record(self, api_key: str, input_tokens: int,
               output_tokens: int, model: str):
        entry = self.usage[api_key]
        entry["requests"] += 1
        entry["input_tokens"] += input_tokens
        entry["output_tokens"] += output_tokens
        entry["cost_usd"] += self._calculate_cost(
            model, input_tokens, output_tokens
        )

    def _calculate_cost(self, model, input_tokens, output_tokens) -> float:
        pricing = {
            "gpt-4o": {"input": 2.50, "output": 10.00},
            "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        }  # per million tokens
        rates = pricing.get(model, {"input": 1.0, "output": 3.0})
        return (
            input_tokens / 1_000_000 * rates["input"] +
            output_tokens / 1_000_000 * rates["output"]
        )

    def get_usage(self, api_key: str) -> dict:
        return dict(self.usage[api_key])
```

---

## Error Handling

AI APIs need error handling that accounts for non-deterministic failures.

### Structured Error Responses

```python
from enum import Enum

class AIErrorCode(str, Enum):
    RATE_LIMITED = "rate_limited"
    CONTEXT_TOO_LONG = "context_too_long"
    CONTENT_FILTERED = "content_filtered"
    MODEL_UNAVAILABLE = "model_unavailable"
    TIMEOUT = "timeout"
    INVALID_REQUEST = "invalid_request"

def handle_llm_error(error: Exception) -> dict:
    """Map provider errors to consistent API error responses."""
    error_msg = str(error).lower()

    if "rate limit" in error_msg:
        return {
            "error": AIErrorCode.RATE_LIMITED,
            "message": "Too many requests. Please retry after a short wait.",
            "retry_after": 30,
            "status_code": 429
        }
    elif "context length" in error_msg or "too many tokens" in error_msg:
        return {
            "error": AIErrorCode.CONTEXT_TOO_LONG,
            "message": "Input is too long for the selected model.",
            "status_code": 400
        }
    elif "content filter" in error_msg or "safety" in error_msg:
        return {
            "error": AIErrorCode.CONTENT_FILTERED,
            "message": "Request was filtered by content safety policies.",
            "status_code": 400
        }
    elif "timeout" in error_msg:
        return {
            "error": AIErrorCode.TIMEOUT,
            "message": "The model took too long to respond. Try a shorter input.",
            "retry_after": 5,
            "status_code": 504
        }
    else:
        return {
            "error": AIErrorCode.MODEL_UNAVAILABLE,
            "message": "The AI service is temporarily unavailable.",
            "retry_after": 10,
            "status_code": 503
        }
```

---

## API Versioning

AI APIs change frequently as models improve. Version your API to avoid breaking clients.

### URL-Based Versioning

```
POST /v1/chat/completions   # Original
POST /v2/chat/completions   # Added streaming, new response format
```

### Header-Based Versioning

```python
@app.post("/api/chat")
async def chat(request: dict, x_api_version: str = "2024-01"):
    if x_api_version == "2024-01":
        return await chat_v1(request)
    elif x_api_version == "2024-06":
        return await chat_v2(request)
    else:
        return {"error": "Unsupported API version"}
```

### Best Practices

- **Always default to the latest stable version** for new clients
- **Support at least 2 versions** to give clients migration time
- **Deprecation notices** in response headers before removing old versions
- **Changelog** documenting what changed between versions

---

## Resources

- **FastAPI** -- Modern Python web framework ideal for AI APIs
- **OpenAI API Reference** -- Industry-standard AI API design to learn from
- **Anthropic API Reference** -- Clean streaming and error handling patterns

---

## Key Takeaways

1. **Stream responses** to improve perceived latency for long-running LLM calls
2. **Rate limit by both requests and tokens** to control costs
3. **Track usage** per API key for billing and monitoring
4. **Map provider errors** to consistent, actionable error codes
5. **Version your API** to evolve without breaking existing clients
