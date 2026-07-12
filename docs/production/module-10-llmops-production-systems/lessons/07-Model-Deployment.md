---
title: Model Deployment Patterns
description: >-
  Learn deployment patterns for LLM applications including API gateways, model
  routers, fallback chains, and blue-green deployments
duration: 30 min
difficulty: intermediate
has_code: false
module: module-10
objectives:
  - Explain the difference between direct API calls and gateway patterns
  - Implement a model router with fallback logic
  - Design a blue-green deployment for prompt updates
  - Understand load balancing across multiple model providers
  - Describe canary deployment strategies for AI systems
---
# Model Deployment Patterns

## What You'll Learn

By the end of this lesson, you'll understand:
- Why LLM deployments need different patterns than traditional software
- API gateway patterns for model access
- Model routing and fallback chains
- Blue-green and canary deployments for AI
- Load balancing across providers

**Time to Complete**: 30 minutes
**Difficulty**: Intermediate

---

## Why LLM Deployment Is Different

Traditional software deployments are predictable: the same input always produces the same output. LLM deployments face unique challenges:

- **Provider outages**: OpenAI, Anthropic, and Google all have downtime
- **Rate limits**: Hitting limits means dropped requests, not just slower responses
- **Cost variability**: Different models have 10-100x cost differences
- **Quality variability**: Model updates can change output quality without warning
- **Latency variability**: Response times range from 200ms to 30+ seconds

These challenges require deployment patterns that go beyond traditional load balancers.

---

## Pattern 1: API Gateway

An API gateway sits between your application and LLM providers, adding reliability and observability.

```python
import time
import logging
from typing import Optional

class LLMGateway:
    def __init__(self, providers: dict):
        self.providers = providers  # {"openai": client, "anthropic": client}
        self.request_log = []

    def call(self, prompt: str, model: str = "gpt-4o-mini",
             timeout: float = 30.0) -> dict:
        """Route a request through the gateway with logging and timeout."""
        start = time.time()
        provider = self._get_provider(model)

        try:
            response = provider.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                timeout=timeout
            )
            latency = time.time() - start

            self._log_request(model, latency, success=True)
            return {
                "content": response.choices[0].message.content,
                "model": model,
                "latency": latency
            }
        except Exception as e:
            latency = time.time() - start
            self._log_request(model, latency, success=False, error=str(e))
            raise

    def _get_provider(self, model: str):
        if model.startswith("claude"):
            return self.providers["anthropic"]
        return self.providers["openai"]

    def _log_request(self, model, latency, success, error=None):
        self.request_log.append({
            "model": model,
            "latency": latency,
            "success": success,
            "error": error,
            "timestamp": time.time()
        })
```

### What the Gateway Adds

- **Unified interface**: One API for all providers
- **Request logging**: Track latency, errors, and costs
- **Timeout enforcement**: Prevent hung requests
- **Authentication**: Centralized API key management

---

## Pattern 2: Model Router with Fallbacks

Route requests to the best available model and fall back gracefully on failures.

```python
class ModelRouter:
    def __init__(self, routes: list[dict]):
        """
        routes = [
            {"model": "gpt-4o", "provider": openai_client, "priority": 1},
            {"model": "claude-sonnet-4-20250514", "provider": anthropic_client, "priority": 2},
            {"model": "gpt-4o-mini", "provider": openai_client, "priority": 3},
        ]
        """
        self.routes = sorted(routes, key=lambda r: r["priority"])

    def call(self, prompt: str, max_retries: int = 3) -> dict:
        """Try each model in priority order until one succeeds."""
        errors = []

        for route in self.routes:
            try:
                response = route["provider"].chat.completions.create(
                    model=route["model"],
                    messages=[{"role": "user", "content": prompt}]
                )
                return {
                    "content": response.choices[0].message.content,
                    "model": route["model"],
                    "fallback": len(errors) > 0
                }
            except Exception as e:
                errors.append({"model": route["model"], "error": str(e)})
                logging.warning(f"Model {route['model']} failed: {e}")

        raise RuntimeError(f"All models failed: {errors}")
```

### Smart Routing

Route based on request characteristics, not just availability:

```python
class SmartRouter:
    def __init__(self, routes: dict):
        self.routes = routes

    def route(self, prompt: str, priority: str = "balanced") -> str:
        """Choose a model based on the request priority."""
        if priority == "quality":
            return "gpt-4o"          # Best quality, highest cost
        elif priority == "speed":
            return "gpt-4o-mini"     # Fastest, cheapest
        elif priority == "balanced":
            # Use expensive model for complex queries
            if len(prompt) > 2000 or "analyze" in prompt.lower():
                return "gpt-4o"
            return "gpt-4o-mini"
```

---

## Pattern 3: Blue-Green Deployment

Run two identical environments and switch traffic between them for zero-downtime updates.

```
                    ┌─────────────────┐
                    │   Load Balancer  │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
     ┌─────────────────┐          ┌─────────────────┐
     │  Blue (Current)  │          │ Green (New)      │
     │  Prompt v3       │          │ Prompt v4        │
     │  Model: GPT-4o   │          │ Model: GPT-4o    │
     │  100% traffic    │          │ 0% traffic       │
     └─────────────────┘          └─────────────────┘
```

```python
class BlueGreenDeployer:
    def __init__(self):
        self.blue_config = None
        self.green_config = None
        self.active = "blue"  # Which environment serves traffic

    def deploy_to_green(self, new_config: dict):
        """Deploy new configuration to the inactive environment."""
        self.green_config = new_config

    def switch_traffic(self):
        """Swap active environment."""
        self.active = "green" if self.active == "blue" else "blue"

    def rollback(self):
        """Instantly revert to the previous environment."""
        self.switch_traffic()

    def get_active_config(self) -> dict:
        return self.blue_config if self.active == "blue" else self.green_config
```

---

## Pattern 4: Canary Deployment

Gradually shift traffic to the new version while monitoring for issues.

```python
class CanaryDeployer:
    def __init__(self, stable_config: dict, canary_config: dict):
        self.stable = stable_config
        self.canary = canary_config
        self.canary_percentage = 0
        self.metrics = {"stable": [], "canary": []}

    def set_canary_percentage(self, pct: int):
        """Set what percentage of traffic goes to canary (0-100)."""
        self.canary_percentage = min(100, max(0, pct))

    def get_config(self, user_id: str) -> dict:
        """Route user to stable or canary based on traffic split."""
        import hashlib
        bucket = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 100
        if bucket < self.canary_percentage:
            return {"env": "canary", **self.canary}
        return {"env": "stable", **self.stable}

    def should_promote(self, min_requests: int = 100) -> bool:
        """Check if canary is healthy enough to promote."""
        if len(self.metrics["canary"]) < min_requests:
            return False  # Not enough data

        canary_success = sum(1 for m in self.metrics["canary"] if m["success"])
        canary_rate = canary_success / len(self.metrics["canary"])

        stable_success = sum(1 for m in self.metrics["stable"] if m["success"])
        stable_rate = stable_success / len(self.metrics["stable"])

        return canary_rate >= stable_rate * 0.95  # Within 5% of stable
```

### Recommended Canary Rollout Schedule

| Stage | Canary Traffic | Duration | Action if Healthy |
|-------|---------------|----------|-------------------|
| 1 | 1% | 1 hour | Monitor error rates |
| 2 | 5% | 4 hours | Check quality metrics |
| 3 | 25% | 24 hours | Compare latency and cost |
| 4 | 50% | 24 hours | Full statistical comparison |
| 5 | 100% | - | Promote canary to stable |

---

## Resources

- **LiteLLM** -- Unified API for 100+ LLM providers with load balancing
- **Portkey** -- AI gateway with routing, fallbacks, and caching
- **Kong AI Gateway** -- Enterprise API gateway with LLM support
- **Martin Fowler's Blue-Green Deployment** -- Original pattern description

---

## Key Takeaways

1. **Use an API gateway** to add reliability and observability to LLM calls
2. **Implement fallback chains** across providers for high availability
3. **Blue-green deployments** enable instant rollback for prompt/model changes
4. **Canary deployments** let you validate changes with real traffic safely
5. **Smart routing** matches request characteristics to the right model
