---
title: API Design for AI Services
description: >-
  Learn how to design robust APIs for LLM-powered services including streaming,
  error handling, rate limiting, and schema validation
duration: 40 min
difficulty: intermediate
has_code: true
module: module-10
---
# API Design for AI Services

## Prerequisites

- Completed Lessons 1–7 (LLMOps Introduction through Model Deployment)
- Familiarity with REST APIs, HTTP status codes, and JSON
- Basic FastAPI or Flask experience (Python async is helpful but not required)

---

## What You'll Learn

| Objective | Outcome |
|-----------|---------|
| Explain why AI API design differs from traditional REST API design | Can articulate latency, variability, and cost challenges |
| Implement streaming responses with Server-Sent Events | Can reduce perceived latency from 5+ seconds to near-instant |
| Build token-bucket rate limiting for dual request/token constraints | Can protect both your API budget and your provider quota |
| Design structured, provider-agnostic error codes | Can give clients actionable error information for any failure mode |
| Version an AI API for backward compatibility | Can evolve your API without breaking existing integrations |

---

## Intuition First: AI APIs Aren't Just Slow REST APIs

Most REST API design advice assumes your endpoint takes 50–200ms to respond. You design for synchronous, predictable, cheap interactions.

AI APIs break every assumption:

| Property | Traditional API | AI API |
|----------|----------------|--------|
| Latency | 10–200ms | 500ms–30s |
| Response size | Predictable | 10 to 10,000+ tokens |
| Per-request cost | ~$0.000001 | $0.001–$0.10+ |
| Errors | Deterministic | Stochastic (hallucination, safety filter) |
| Output validity | 200 OK = correct output | 200 OK = output exists, not necessarily correct |

These differences require design patterns that traditional API documentation rarely covers. A client that uses polling, keeps connections alive for 30 seconds, and handles "I cannot answer that" as a successful response needs a different API contract than a typical CRUD service.

---

## Streaming Responses with Server-Sent Events

When an LLM takes 5 seconds to generate 300 tokens, users don't want to wait 5 seconds for anything to appear. Streaming sends tokens as they're generated, reducing perceived latency from 5 seconds to near-instant.

**Server-Sent Events (SSE)** is the standard protocol for streaming text from server to browser. It's simpler than WebSockets and works over standard HTTP.

### FastAPI Streaming Endpoint

```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from openai import OpenAI
import json
import time

app = FastAPI(title="AI Service API", version="2.0.0")
client = OpenAI()

class ChatRequest(BaseModel):
    messages: list[dict[str, str]] = Field(..., min_length=1)
    model: str = Field(default="gpt-4o-mini")
    max_tokens: int = Field(default=500, ge=1, le=4096)
    stream: bool = Field(default=True)

class ChatMetadata(BaseModel):
    request_id: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float

@app.post("/v2/chat")
async def chat(request: ChatRequest, http_request: Request):
    """
    Stream an LLM response using Server-Sent Events.

    SSE Format:
      data: {"type": "content", "content": "Hello"}\\n\\n
      data: {"type": "content", "content": " world"}\\n\\n
      data: {"type": "done", "metadata": {...}}\\n\\n
    """
    request_id = http_request.headers.get("X-Request-ID", str(time.time()))

    async def generate():
        start = time.time()
        input_tokens = 0
        output_tokens = 0
        cost_usd = 0.0

        try:
            stream = client.chat.completions.create(
                model=request.model,
                messages=request.messages,
                max_tokens=request.max_tokens,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    data = json.dumps({
                        "type": "content",
                        "content": delta.content,
                        "request_id": request_id,
                    })
                    yield f"data: {data}\n\n"

                # Capture usage from the final chunk
                if chunk.usage:
                    input_tokens = chunk.usage.prompt_tokens
                    output_tokens = chunk.usage.completion_tokens

            latency_ms = (time.time() - start) * 1000
            cost_usd = (input_tokens / 1e6 * 0.15 +
                        output_tokens / 1e6 * 0.60)

            # Final event with metadata
            done_data = json.dumps({
                "type": "done",
                "request_id": request_id,
                "metadata": {
                    "model": request.model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "latency_ms": round(latency_ms, 1),
                    "cost_usd": round(cost_usd, 6),
                },
            })
            yield f"data: {done_data}\n\n"

        except Exception as e:
            error_data = json.dumps({
                "type": "error",
                "error": classify_error(e),
                "request_id": request_id,
            })
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",    # Disable nginx buffering for SSE
        },
    )
```

### JavaScript Client for SSE

```javascript
async function* streamChat(messages, options = {}) {
  const response = await fetch('/v2/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Request-ID': crypto.randomUUID(),
    },
    body: JSON.stringify({ messages, ...options }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Chat request failed');
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n\n');
    buffer = lines.pop();  // Keep incomplete chunk in buffer

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const event = JSON.parse(line.slice(6));
      yield event;  // Caller handles content, done, and error events
    }
  }
}

// Usage
async function renderChat(messages, outputElement) {
  let fullResponse = '';
  for await (const event of streamChat(messages)) {
    if (event.type === 'content') {
      fullResponse += event.content;
      outputElement.textContent = fullResponse;  // Update UI incrementally
    } else if (event.type === 'done') {
      console.log('Cost:', event.metadata.cost_usd);
    } else if (event.type === 'error') {
      console.error('AI error:', event.error);
    }
  }
}
```

---

## Rate Limiting: Dual Constraints

AI APIs need rate limiting on two axes simultaneously:
- **Requests per minute (RPM)**: Prevents burst abuse and matches provider RPM limits
- **Tokens per minute (TPM)**: Prevents high-cost requests from consuming all budget

A simple request-count rate limiter misses the token dimension entirely—one user sending 50 massive context windows counts the same as 50 users sending short queries.

```python
import time
from collections import defaultdict
from dataclasses import dataclass, field
import threading

@dataclass
class RateBucket:
    requests: int
    tokens: int
    last_refill: float = field(default_factory=time.time)

class DualRateLimiter:
    """
    Token-bucket rate limiter enforcing both RPM and TPM limits.
    Buckets refill every 60 seconds per API key.
    """

    def __init__(self, rpm_limit: int = 60, tpm_limit: int = 100_000):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self._buckets: dict[str, RateBucket] = defaultdict(
            lambda: RateBucket(requests=rpm_limit, tokens=tpm_limit)
        )
        self._lock = threading.Lock()

    def _refill(self, bucket: RateBucket):
        now = time.time()
        elapsed = now - bucket.last_refill
        if elapsed >= 60:
            bucket.requests = self.rpm_limit
            bucket.tokens = self.tpm_limit
            bucket.last_refill = now

    def check(self, api_key: str, estimated_tokens: int = 1_000) -> dict:
        """
        Returns {"allowed": True} if within limits.
        Returns {"allowed": False, "reason": ..., "retry_after": N} otherwise.
        """
        with self._lock:
            bucket = self._buckets[api_key]
            self._refill(bucket)

            if bucket.requests < 1:
                return {
                    "allowed": False,
                    "reason": "request_rate_limit",
                    "retry_after": int(60 - (time.time() - bucket.last_refill)),
                    "limit": self.rpm_limit,
                    "remaining": 0,
                }

            if bucket.tokens < estimated_tokens:
                return {
                    "allowed": False,
                    "reason": "token_rate_limit",
                    "retry_after": int(60 - (time.time() - bucket.last_refill)),
                    "token_limit": self.tpm_limit,
                    "tokens_remaining": bucket.tokens,
                }

            bucket.requests -= 1
            bucket.tokens -= estimated_tokens
            return {
                "allowed": True,
                "requests_remaining": bucket.requests,
                "tokens_remaining": bucket.tokens,
            }


# FastAPI dependency
rate_limiter = DualRateLimiter(rpm_limit=60, tpm_limit=100_000)

def get_api_key(request: Request) -> str:
    key = request.headers.get("Authorization", "").removeprefix("Bearer ")
    if not key:
        raise HTTPException(status_code=401, detail="Missing API key")
    return key

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path.startswith("/v"):
        api_key = request.headers.get("Authorization", "anon")
        # Estimate tokens from request body
        body = await request.body()
        estimated_tokens = len(body) // 4  # Rough token estimate

        result = rate_limiter.check(api_key, estimated_tokens)
        if not result["allowed"]:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=429,
                content={"error": result["reason"], "retry_after": result["retry_after"]},
                headers={"Retry-After": str(result["retry_after"])},
            )
    return await call_next(request)
```

---

## Structured Error Handling

LLM APIs can fail in ways that traditional APIs don't. Your error taxonomy should map to actions clients can take.

```python
from enum import Enum
from fastapi import HTTPException

class AIErrorCode(str, Enum):
    # Client errors (4xx) — client should change the request
    RATE_LIMITED = "rate_limited"              # 429: slow down
    CONTEXT_TOO_LONG = "context_too_long"      # 400: shorten input
    CONTENT_FILTERED = "content_filtered"      # 400: request blocked by safety
    INVALID_MODEL = "invalid_model"            # 400: model name not recognized
    INVALID_REQUEST = "invalid_request"        # 400: malformed request body

    # Server errors (5xx) — client should retry with backoff
    MODEL_UNAVAILABLE = "model_unavailable"    # 503: try again
    TIMEOUT = "timeout"                        # 504: request took too long
    PROVIDER_ERROR = "provider_error"          # 502: upstream LLM failed

    # Quality errors (200 with error flag) — client should handle gracefully
    QUALITY_FILTERED = "quality_filtered"      # Response failed output validation


def classify_error(exc: Exception) -> dict:
    """
    Map provider-specific exceptions to a consistent error schema.
    This insulates clients from provider changes.
    """
    from openai import RateLimitError, APITimeoutError, APIStatusError, BadRequestError

    if isinstance(exc, RateLimitError):
        return {
            "code": AIErrorCode.RATE_LIMITED,
            "message": "Too many requests. Retry after the specified delay.",
            "http_status": 429,
            "retry_after": 30,
            "actionable": True,
        }
    elif isinstance(exc, APITimeoutError):
        return {
            "code": AIErrorCode.TIMEOUT,
            "message": "The model took too long to respond. Try a shorter input.",
            "http_status": 504,
            "retry_after": 5,
            "actionable": True,
        }
    elif isinstance(exc, BadRequestError):
        msg = str(exc).lower()
        if "context_length" in msg or "too many tokens" in msg:
            return {
                "code": AIErrorCode.CONTEXT_TOO_LONG,
                "message": "Input exceeds the model's context window.",
                "http_status": 400,
                "retry_after": None,
                "actionable": True,
            }
        elif "content_filter" in msg or "safety" in msg:
            return {
                "code": AIErrorCode.CONTENT_FILTERED,
                "message": "Request was blocked by content safety policies.",
                "http_status": 400,
                "retry_after": None,
                "actionable": False,
            }
    elif isinstance(exc, APIStatusError) and exc.status_code >= 500:
        return {
            "code": AIErrorCode.PROVIDER_ERROR,
            "message": "The AI service is temporarily unavailable.",
            "http_status": 502,
            "retry_after": 10,
            "actionable": True,
        }

    return {
        "code": AIErrorCode.MODEL_UNAVAILABLE,
        "message": "An unexpected error occurred. Please retry.",
        "http_status": 503,
        "retry_after": 10,
        "actionable": True,
    }
```

Expose these error codes in your OpenAPI schema so clients can write exhaustive error handling without guessing.

---

## API Versioning

AI APIs change frequently: models are deprecated, response formats evolve, new capabilities are added. Versioning lets you evolve the API without breaking existing clients.

```python
# URL-based versioning (most explicit)
# /v1/chat  — original sync format
# /v2/chat  — added streaming, metadata in events
# /v3/chat  — added tool use, multi-modal input

# Header-based versioning (cleaner URLs)
@app.post("/api/chat")
async def chat(request: ChatRequest,
               api_version: str = Header(default="2024-01", alias="X-API-Version")):
    if api_version == "2024-01":
        return await chat_v1(request)
    elif api_version == "2024-06":
        return await chat_v2(request)
    else:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "unsupported_api_version",
                "message": f"Version {api_version} is not supported.",
                "supported_versions": ["2024-01", "2024-06"],
                "latest_version": "2024-06",
            }
        )
```

### Versioning Best Practices

| Practice | Why |
|----------|-----|
| Include version in URL path (`/v2/chat`) for the primary API | Clients can pin to a version without header manipulation |
| Support at least 2 versions simultaneously | Gives clients time to migrate without breaking them |
| Announce deprecations with `Deprecation` and `Sunset` response headers | RFC 8594; clients can detect upcoming breakage programmatically |
| Never change the response schema within a version | Even adding optional fields can break strict parsers |
| Document all versions in your OpenAPI spec | Generates client SDKs with version-specific type safety |

---

## Usage Tracking and Billing

If you're building a multi-tenant AI service, you need per-tenant usage tracking for billing, quota enforcement, and abuse detection.

```python
from collections import defaultdict
from datetime import datetime, date

class UsageTracker:
    """Track per-API-key usage for billing and quota enforcement."""

    PRICING = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},    # per 1M tokens
        "gpt-4o": {"input": 2.50, "output": 10.00},
    }

    def __init__(self):
        self._daily: dict[str, dict[date, dict]] = defaultdict(
            lambda: defaultdict(lambda: {"requests": 0, "input_tokens": 0,
                                         "output_tokens": 0, "cost_usd": 0.0})
        )

    def record(self, api_key: str, model: str,
               input_tokens: int, output_tokens: int):
        today = date.today()
        rates = self.PRICING.get(model, {"input": 1.0, "output": 3.0})
        cost = (input_tokens / 1e6 * rates["input"] +
                output_tokens / 1e6 * rates["output"])

        entry = self._daily[api_key][today]
        entry["requests"] += 1
        entry["input_tokens"] += input_tokens
        entry["output_tokens"] += output_tokens
        entry["cost_usd"] += cost

    def get_today_usage(self, api_key: str) -> dict:
        return dict(self._daily[api_key][date.today()])

    def get_monthly_cost(self, api_key: str) -> float:
        today = date.today()
        return sum(
            day_data["cost_usd"]
            for day, day_data in self._daily[api_key].items()
            if day.year == today.year and day.month == today.month
        )
```

---

## Edge Cases and Misconceptions

**"SSE works like WebSockets."**
SSE is unidirectional (server → client) over plain HTTP. WebSockets are bidirectional. SSE is simpler, works through proxies and CDNs more reliably, and is sufficient for LLM streaming. Use WebSockets only if you need bidirectional streaming (e.g., user can interrupt generation mid-stream).

**"Rate limiting by request count is enough."**
A single request with a 100K-token context costs as much as 100 typical requests. RPM limits alone let a user send one very expensive request per minute indefinitely. Always rate limit on both request count and token count.

**"I can just use HTTP 500 for all AI errors."**
Clients need to distinguish retryable errors (rate limits, timeouts) from permanent failures (content filtered, context too long). A generic 500 prevents clients from implementing smart retry logic. Use structured error codes with `retry_after` hints.

**"Versioning is only for public APIs."**
Internal APIs also need versioning. If your web app and mobile app hit the same AI backend, and you change the response format, both break simultaneously. Version internal APIs with the same discipline as public ones.

---

## Production Scenario: Building a Multi-Tenant AI API

You are building an AI writing assistant API used by three different client applications: a web editor, a mobile app, and a browser extension. Each has different needs, and you need the API to serve all three without breaking when your AI models or prompts change.

### API Versioning Strategy

```python
# Client apps declare which version they support at build time
# Web editor: /v1/completions (current)
# Mobile app: /v1/completions (current, pinned at build time)
# Browser extension: /v2/completions (beta, testing new streaming format)

API_VERSIONS = {
    "v1": {
        "status": "stable",
        "sunset_date": None,      # Current stable; no sunset
        "response_format": "json",
        "streaming_protocol": "sse_text",
    },
    "v2": {
        "status": "beta",
        "sunset_date": None,      # Not yet sunset
        "response_format": "json",
        "streaming_protocol": "sse_json",     # Richer streaming metadata
    },
}

def add_version_headers(response, version: str):
    """Add deprecation headers if the version is being sunset."""
    info = API_VERSIONS.get(version, {})
    if info.get("sunset_date"):
        response.headers["Sunset"] = info["sunset_date"]
        response.headers["Deprecation"] = "true"
        response.headers["Link"] = '</docs/migration/v1-to-v2>; rel="deprecation"'
    response.headers["API-Version"] = version
    return response
```

### Per-Tenant Rate Limiting

Each client application gets its own quota:

```python
TENANT_RATE_LIMITS = {
    "web_editor":        {"rpm": 300, "tpm": 600_000},
    "mobile_app":        {"rpm": 100, "tpm": 200_000},
    "browser_extension": {"rpm": 60,  "tpm": 120_000},
    "api_tier_free":     {"rpm": 10,  "tpm": 20_000},
    "api_tier_pro":      {"rpm": 100, "tpm": 300_000},
}
```

When the web editor hits its TPM limit at 11 AM on a high-traffic day, it receives:

```json
{
  "error": {
    "code": "TOKEN_RATE_LIMIT_EXCEEDED",
    "message": "You have exceeded 600,000 tokens per minute for this API key.",
    "type": "rate_limit_error",
    "param": null,
    "retry_after": 38,
    "quota": {
      "rpm_limit": 300,
      "tpm_limit": 600000,
      "tpm_used_this_minute": 612441
    }
  }
}
```

The web editor's client code reads `retry_after: 38` and backs off 38 seconds before retrying. No user-visible error, no wasted requests.

### Handling a V1 → V2 Migration for Mobile

When you ship v2 with structured streaming tokens, mobile clients on v1 cannot parse the new format. The versioning strategy prevents breakage:

1. Mobile app remains on `/v1/completions` — receives the v1 response format
2. v2 is tested by the browser extension for 4 weeks
3. Once v2 is stable, send deprecation notice to mobile team with 90-day migration window
4. `Sunset` and `Deprecation` headers appear in v1 responses: mobile app logs a warning
5. Mobile app migrates at their own pace; v1 stays available until sunset date

Without versioning, the same change would break the mobile app at deploy time, requiring an emergency app store update.

---

## Key Takeaways

- Stream AI responses using Server-Sent Events to reduce perceived latency from seconds to near-instant; SSE works over standard HTTP and is simpler than WebSockets
- Rate limit on both requests per minute and tokens per minute; RPM-only limits fail for users sending large context windows
- Define a structured error taxonomy that maps to client actions: retryable errors include `retry_after`; permanent errors explain what the client must change
- Version your API from day one; support at least two versions simultaneously and announce deprecations with response headers
- Track per-key usage for billing and abuse detection; cost is the metric that connects usage to business impact

---

## Further Reading

- [The Streaming Problem in AI APIs](https://arxiv.org/abs/2312.01543) — Analysis of streaming patterns and latency perception in LLM interfaces
- [HTTP Semantics for Rate Limiting](https://www.rfc-editor.org/rfc/rfc6585) — RFC covering the 429 Too Many Requests status code and Retry-After header
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference) — Industry-standard AI API design; study the streaming, error, and versioning patterns
- [FastAPI documentation](https://fastapi.tiangolo.com/) — Python web framework with excellent streaming response support

---

## Next Lesson

**Lesson 9: Security & Privacy for LLM Applications** — Learn to defend against prompt injection, detect and redact PII before it reaches third-party APIs, and implement output filtering to prevent data leakage.
