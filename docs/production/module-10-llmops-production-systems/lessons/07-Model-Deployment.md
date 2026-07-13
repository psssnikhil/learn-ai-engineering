---
title: Model Deployment Patterns
description: >-
  Learn deployment patterns for LLM applications including API gateways, model
  routers, fallback chains, and blue-green deployments
duration: 40 min
difficulty: intermediate
has_code: true
module: module-10
---
# Model Deployment Patterns

## Prerequisites

- Completed Lessons 1–6 (LLMOps Introduction through Cost Optimization)
- Familiarity with Python exception handling and basic async patterns
- Understanding of API gateway concepts (routing, load balancing)

---

## What You'll Learn

| Objective | Outcome |
|-----------|---------|
| Explain why LLM deployments need patterns beyond traditional load balancing | Can articulate provider dependencies, cost variability, and quality variability as unique challenges |
| Implement an API gateway for LLM provider abstraction | Can build a single interface routing to any provider |
| Design a fallback chain across providers | Can maintain uptime during provider outages |
| Deploy prompt and model changes safely using blue-green patterns | Can swap configurations with zero-downtime rollback |
| Execute a canary deployment with automated promotion/rollback | Can validate changes on 1% of traffic before full exposure |

---

## Intuition First: The Third-Party Dependency Problem

Traditional web app deployments are deterministic and self-contained. You control your servers, your databases, your code. If something breaks, you fix it.

LLM applications depend on third-party model providers—OpenAI, Anthropic, Google, Mistral. You cannot scale them, cannot fix their outages, cannot predict when they update model behavior. In the year following GPT-4's release, OpenAI experienced over 20 significant service disruptions. Anthropic and Google have similar track records.

Your deployment architecture must assume providers will fail, degrade, and change. Everything else follows from this assumption.

```
Traditional web app reliability:
  YOU control: servers, databases, code, networking
  Failure mode: you broke something → you fix it

LLM app reliability:
  YOU control: prompts, caching, routing logic, fallback chains
  Third party controls: model quality, latency, availability, pricing
  Failure mode: external factors → you need resilience built in from day one
```

---

## Pattern 1: The API Gateway

An API gateway is a single entry point that abstracts your application from the specifics of LLM providers. All LLM calls go through the gateway, which handles routing, logging, timeout enforcement, and authentication.

```python
import time
import logging
from typing import Optional
from dataclasses import dataclass, field
from openai import OpenAI, RateLimitError, APITimeoutError, APIStatusError
import anthropic

logger = logging.getLogger("llm_gateway")

@dataclass
class GatewayRequest:
    prompt: str
    model: str = "gpt-4o-mini"
    system_prompt: Optional[str] = None
    max_tokens: int = 500
    temperature: float = 0.0
    timeout_seconds: float = 30.0
    trace_id: Optional[str] = None


@dataclass
class GatewayResponse:
    content: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    trace_id: Optional[str] = None
    from_fallback: bool = False


class LLMGateway:
    """
    Provider-agnostic gateway for LLM calls.
    Handles routing, logging, timeout, and cost tracking.
    """

    PROVIDER_MAP = {
        "gpt-": "openai",
        "claude-": "anthropic",
        "gemini-": "google",
    }

    COST_PER_1M = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
        "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    }

    def __init__(self, openai_client: OpenAI,
                 anthropic_client: Optional[anthropic.Anthropic] = None):
        self.clients = {
            "openai": openai_client,
            "anthropic": anthropic_client,
        }
        self._request_count = 0
        self._error_count = 0

    def _detect_provider(self, model: str) -> str:
        for prefix, provider in self.PROVIDER_MAP.items():
            if model.lower().startswith(prefix):
                return provider
        return "openai"  # Default

    def _calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        rates = self.COST_PER_1M.get(model, {"input": 1.0, "output": 3.0})
        return (input_tokens / 1e6 * rates["input"] +
                output_tokens / 1e6 * rates["output"])

    def call(self, request: GatewayRequest) -> GatewayResponse:
        start = time.time()
        self._request_count += 1
        provider = self._detect_provider(request.model)

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        logger.info({
            "event": "gateway_request",
            "trace_id": request.trace_id,
            "model": request.model,
            "provider": provider,
            "estimated_input_tokens": len(request.prompt.split()) * 1.3,
        })

        try:
            if provider == "openai":
                resp = self.clients["openai"].chat.completions.create(
                    model=request.model,
                    messages=messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    timeout=request.timeout_seconds,
                )
                content = resp.choices[0].message.content
                input_t = resp.usage.prompt_tokens
                output_t = resp.usage.completion_tokens

            elif provider == "anthropic" and self.clients.get("anthropic"):
                resp = self.clients["anthropic"].messages.create(
                    model=request.model,
                    max_tokens=request.max_tokens,
                    messages=[{"role": "user", "content": request.prompt}],
                    system=request.system_prompt or "",
                )
                content = resp.content[0].text
                input_t = resp.usage.input_tokens
                output_t = resp.usage.output_tokens
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            latency = (time.time() - start) * 1000
            cost = self._calculate_cost(request.model, input_t, output_t)

            logger.info({
                "event": "gateway_response",
                "trace_id": request.trace_id,
                "model": request.model,
                "input_tokens": input_t,
                "output_tokens": output_t,
                "latency_ms": round(latency, 1),
                "cost_usd": round(cost, 6),
            })

            return GatewayResponse(
                content=content, model=request.model, provider=provider,
                input_tokens=input_t, output_tokens=output_t,
                latency_ms=latency, cost_usd=cost, trace_id=request.trace_id,
            )

        except Exception as e:
            self._error_count += 1
            logger.error({"event": "gateway_error", "trace_id": request.trace_id,
                          "error": str(e), "model": request.model})
            raise
```

---

## Pattern 2: Fallback Chains

A fallback chain tries each model in priority order, moving to the next on failure. This provides resilience against provider outages, rate limits, and model-specific errors.

```python
from typing import NamedTuple

class FallbackRoute(NamedTuple):
    model: str
    priority: int
    max_retries: int = 1

class FallbackChain:
    """
    Tries models in priority order. Catches transient errors and falls back.
    Raises only if all routes fail.
    """

    # Error types that should trigger fallback (vs. errors that should be re-raised)
    FALLBACK_ERRORS = (RateLimitError, APITimeoutError)

    def __init__(self, gateway: LLMGateway, routes: list[FallbackRoute]):
        self.gateway = gateway
        self.routes = sorted(routes, key=lambda r: r.priority)

    def call(self, prompt: str, system_prompt: str = "",
             **kwargs) -> GatewayResponse:
        errors = []

        for route in self.routes:
            for attempt in range(route.max_retries):
                try:
                    request = GatewayRequest(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        model=route.model,
                        **kwargs,
                    )
                    response = self.gateway.call(request)
                    # Mark as fallback if we used anything but the first-priority route
                    response.from_fallback = route.priority > 1
                    if response.from_fallback:
                        logger.warning({
                            "event": "fallback_activated",
                            "model": route.model,
                            "prior_errors": len(errors),
                        })
                    return response
                except self.FALLBACK_ERRORS as e:
                    errors.append({"model": route.model, "attempt": attempt, "error": str(e)})
                    logger.warning({"event": "fallback_trigger", "model": route.model,
                                   "error": type(e).__name__})
                    if attempt < route.max_retries - 1:
                        time.sleep(0.5 * (attempt + 1))  # Brief backoff before retry
                except Exception as e:
                    # Non-retriable error: skip retries, try next route
                    errors.append({"model": route.model, "error": str(e)})
                    break

        raise RuntimeError(f"All fallback routes failed: {errors}")


# Production fallback configuration
fallback = FallbackChain(
    gateway=gateway,
    routes=[
        FallbackRoute(model="gpt-4o-mini", priority=1, max_retries=2),
        FallbackRoute(model="claude-3-5-haiku-20241022", priority=2, max_retries=1),
        FallbackRoute(model="gpt-4o", priority=3, max_retries=1),  # Premium fallback
    ],
)
```

!!! note "Why Not Just Retry the Same Model?"
    For rate limit errors, retrying the same model usually fails again immediately. Provider rate limits are enforced per API key per minute—hitting the limit means the limit is exhausted for that window. Falling back to a different provider immediately restores service without waiting.

---

## Pattern 3: Blue-Green Deployment

Blue-green deployment maintains two environments—the current ("blue") and the new ("green")—and switches traffic between them instantaneously. For LLM apps, this pattern handles prompt updates, model migrations, and RAG config changes.

```
                      ┌──────────────────┐
                      │  Traffic Router   │
                      └────────┬─────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
   ┌──────────────────┐             ┌──────────────────┐
   │   BLUE (active)  │             │  GREEN (standby) │
   │  prompt v3       │             │  prompt v4       │
   │  gpt-4o-mini     │             │  gpt-4o-mini     │
   │  100% traffic    │             │  0% traffic      │
   └──────────────────┘             └──────────────────┘

After validation:
   │   BLUE (standby) │             │  GREEN (active)  │
   │  prompt v3       │             │  prompt v4       │
   │  gpt-4o-mini     │             │  gpt-4o-mini     │
   │  0% traffic      │             │  100% traffic    │
```

```python
import os
from dataclasses import dataclass

@dataclass
class DeploymentConfig:
    prompt_name: str
    prompt_version: int
    model: str
    temperature: float = 0.0
    max_tokens: int = 500

class BlueGreenDeployer:
    """
    Manages two deployment environments with instant traffic switching.
    Blue is always the "last known good" configuration.
    """

    def __init__(self):
        self.blue: Optional[DeploymentConfig] = None
        self.green: Optional[DeploymentConfig] = None
        self._active = "blue"

    @property
    def active_config(self) -> Optional[DeploymentConfig]:
        return self.blue if self._active == "blue" else self.green

    @property
    def standby_config(self) -> Optional[DeploymentConfig]:
        return self.green if self._active == "blue" else self.blue

    def deploy_to_standby(self, config: DeploymentConfig):
        """Load a new configuration into the standby slot."""
        if self._active == "blue":
            self.green = config
            logger.info({"event": "deploy_to_green", "config": vars(config)})
        else:
            self.blue = config
            logger.info({"event": "deploy_to_blue", "config": vars(config)})

    def promote_standby(self):
        """Switch traffic from the current active to the standby."""
        old_active = self._active
        self._active = "green" if self._active == "blue" else "blue"
        logger.info({"event": "traffic_switched",
                     "from": old_active, "to": self._active})

    def rollback(self):
        """Instantly revert to the previous active environment."""
        self.promote_standby()
        logger.warning({"event": "rollback_executed", "now_active": self._active})

    def status(self) -> dict:
        return {
            "active": self._active,
            "active_config": vars(self.active_config) if self.active_config else None,
            "standby_config": vars(self.standby_config) if self.standby_config else None,
        }


# Deployment workflow:
deployer = BlueGreenDeployer()
deployer.blue = DeploymentConfig("support_agent", 3, "gpt-4o-mini")  # Current

# 1. Load new config into standby (no traffic impact)
deployer.deploy_to_standby(DeploymentConfig("support_agent", 4, "gpt-4o-mini"))

# 2. Run eval suite against standby
# result = tester.run("support_agent", 4, test_suite)
# assert result["pass_rate"] >= 0.95, "Quality gate failed"

# 3. Switch traffic (instantaneous)
deployer.promote_standby()

# 4. Monitor for 30 minutes; if quality drops:
# deployer.rollback()
```

**Time to rollback**: Under 1 second. The old environment is always warm and ready.

---

## Pattern 4: Canary Deployment

Canary deployment exposes the new configuration to a small percentage of real users while the majority stays on the stable version. It catches issues that eval suites miss—production data distributions, edge cases, and user-specific behavior.

```python
import hashlib

class CanaryDeployer:
    """
    Gradual traffic shift with automated promotion and rollback.
    Canary percentage increases as confidence in the new config grows.
    """

    def __init__(self, stable: DeploymentConfig, canary: DeploymentConfig):
        self.stable = stable
        self.canary = canary
        self._canary_pct = 0.0       # Start at 0%
        self._metrics: dict[str, list[float]] = {"stable": [], "canary": []}

    @property
    def canary_pct(self) -> float:
        return self._canary_pct

    def set_canary_pct(self, pct: float):
        assert 0 <= pct <= 100
        self._canary_pct = pct
        logger.info({"event": "canary_pct_updated", "pct": pct})

    def get_config(self, request_id: str) -> tuple[DeploymentConfig, str]:
        """
        Deterministically assign a request to stable or canary.
        Returns (config, environment_name).
        """
        bucket = int(hashlib.sha256(request_id.encode()).hexdigest(), 16) % 100
        if bucket < self._canary_pct:
            return self.canary, "canary"
        return self.stable, "stable"

    def record_metric(self, environment: str, metric: str, value: float):
        key = f"{environment}_{metric}"
        if key not in self._metrics:
            self._metrics[key] = []
        self._metrics[key].append(value)

    def should_promote(self, min_samples: int = 200) -> dict:
        """
        Evaluate whether the canary is ready to promote to 100%.
        Returns a decision dict with reasons.
        """
        canary_satisfaction = self._metrics.get("canary_satisfaction", [])
        stable_satisfaction = self._metrics.get("stable_satisfaction", [])

        if len(canary_satisfaction) < min_samples:
            return {
                "action": "hold",
                "reason": f"Insufficient samples: {len(canary_satisfaction)}/{min_samples}",
            }

        canary_avg = sum(canary_satisfaction) / len(canary_satisfaction)
        stable_avg = sum(stable_satisfaction) / len(stable_satisfaction)
        regression_pct = (stable_avg - canary_avg) / stable_avg * 100

        if regression_pct > 3:    # More than 3% regression from stable
            return {
                "action": "rollback",
                "reason": f"Canary {regression_pct:.1f}% worse than stable",
                "canary_avg": canary_avg,
                "stable_avg": stable_avg,
            }

        return {
            "action": "promote",
            "reason": f"Canary within {regression_pct:.1f}% of stable quality",
            "canary_avg": canary_avg,
            "stable_avg": stable_avg,
        }
```

### Recommended Canary Rollout Schedule

| Stage | Canary Traffic | Minimum Duration | Decision Criteria |
|-------|---------------|------------------|-------------------|
| 1 | 1% | 2 hours | Error rate and timeout rate |
| 2 | 5% | 8 hours | Quality score vs. stable |
| 3 | 20% | 24 hours | Full quality + cost + latency |
| 4 | 50% | 24 hours | Statistical significance confirmed |
| 5 | 100% | — | Promote, retire old version |

Never rush through these stages. The 1% stage exists to catch catastrophic failures (model errors, format breakage) before they affect real users.

---

## Edge Cases and Misconceptions

**"Blue-green deployments require maintaining duplicate infrastructure."**
For LLM apps, blue-green is primarily a configuration pattern, not an infrastructure pattern. Both "environments" use the same API endpoint and infrastructure—they just use different prompt versions, model configs, or RAG settings. The cost of maintaining two configs is negligible.

**"Canary deployments work like traditional software canaries."**
Traditional canaries catch code crashes and errors. LLM canaries must additionally measure quality degradation, which requires an evaluation layer (LLM-as-judge or user feedback collection) running on canary traffic in parallel. Without quality measurement, your canary only catches hard failures, not the more common soft degradation.

**"Once a fallback activates, it means the primary is down."**
Fallbacks activate on rate limits, timeouts, and transient errors—not just full outages. Track your fallback activation rate as a metric. A rising fallback rate might indicate you're approaching your rate limit tier and need to upgrade, not that there's a provider outage.

**"I should test blue-green by running both environments in parallel."**
Running both environments simultaneously on the same traffic introduces confounding: the same user gets different responses at different times. For comparison testing, use canary deployment (deliberate traffic split) or shadow traffic (run both but only return stable). Blue-green is for zero-downtime deployment with instant rollback, not for A/B comparison.

---

## Production Scenario: Migrating from GPT-4o to Claude 3.5 Sonnet

Your team wants to switch your customer support bot's primary model from GPT-4o to Claude 3.5 Sonnet after internal tests show better instruction-following. Here is how you deploy this safely using the full deployment stack.

### Phase 1: Validate Before Any Production Traffic

```python
# Run your golden set against both models; compare pass rates
from eval.runner import run_eval_suite

gpt4o_results = run_eval_suite(model="gpt-4o", golden_set="eval/golden-set/")
claude_results = run_eval_suite(model="claude-3-5-sonnet-20241022", golden_set="eval/golden-set/")

print(f"GPT-4o pass rate:   {gpt4o_results['pass_rate']:.1%}")
print(f"Claude pass rate:   {claude_results['pass_rate']:.1%}")

# If Claude pass rate >= GPT-4o pass rate: proceed
# If Claude pass rate < GPT-4o pass rate - 5%: investigate before proceeding
```

Suppose results show: GPT-4o 91.2%, Claude 93.4%. Claude is better offline. Proceed to shadow traffic.

### Phase 2: Shadow Traffic (Zero-Risk Validation on Real Queries)

Configure the shadow traffic handler to run Claude on production queries for 72 hours while continuing to return GPT-4o responses to users:

```python
shadow_handler = ShadowTrafficHandler(
    stable_handler=GPT4oHandler(),
    candidate_handler=ClaudeHandler(model="claude-3-5-sonnet-20241022"),
    eval_recorder=EvalRecorder(output_dir="shadow_results/"),
)
```

After 72 hours, analyze:
- Output changed in 38% of requests (expected; models have different output style)
- Average quality delta: +0.031 (Claude slightly better across production queries)
- Average latency delta: +210ms (Claude slightly slower)
- No format violations in candidate output

Shadow traffic confirms Claude is safe. Proceed to canary.

### Phase 3: Canary Rollout

```python
CANARY_STAGES = [
    {"pct": 1,  "duration_h": 2,  "gate": ["error_rate", "timeout_rate"]},
    {"pct": 5,  "duration_h": 8,  "gate": ["quality_score", "error_rate"]},
    {"pct": 20, "duration_h": 24, "gate": ["quality_score", "latency_p99", "cost_per_request"]},
    {"pct": 50, "duration_h": 24, "gate": ["all"]},
    {"pct": 100,"duration_h": 0,  "gate": []},
]
```

At 5% canary after 8 hours:
- Error rate: 0.3% (within threshold of 2%)
- Quality score: 0.891 vs stable 0.874 (+2%)
- Cost per request: $0.0042 vs $0.0038 (+11%)
- Decision: **Proceed to 20%** (cost increase acceptable given quality gain)

At 20% canary after 24 hours:
- All metrics stable; cost slightly lower than 5% stage (Claude caches better)
- Decision: **Proceed to 50%**

At 50% after 24 hours:
- Quality score 0.887 vs baseline 0.874 (+1.5%); cost neutral
- Decision: **Promote to 100%**

Total safe migration time: 60 hours. Zero user-facing quality incidents. Full rollback option available at every stage.

---

## Key Takeaways

- LLM applications depend on third-party providers—build resilience in from day one, not as an afterthought
- An API gateway abstracts provider specifics, centralizes logging and cost tracking, and enables flexible routing
- Fallback chains restore service during rate limit and outage events by trying alternative providers or models in priority order
- Blue-green deployments enable instant rollback of prompt and model changes without infrastructure changes
- Canary deployments expose new configurations to a small traffic percentage with quality monitoring before full rollout
- Track fallback activation rate as a leading indicator of infrastructure stress; a rising rate often precedes outages

---

## Further Reading

- [Site Reliability Engineering](https://sre.google/sre-book/table-of-contents/) — Google's SRE book; chapters on release engineering and change management apply directly to LLM deployments
- [Blue-Green Deployments](https://martinfowler.com/bliki/BlueGreenDeployment.html) — Martin Fowler's original description of the pattern
- [Canary Releases](https://martinfowler.com/bliki/CanaryRelease.html) — Martin Fowler's canary deployment pattern
- [LiteLLM documentation](https://docs.litellm.ai/) — Open-source unified LLM API with built-in fallback and load balancing

---

## Next Lesson

**Lesson 8: API Design for AI Services** — Learn to design AI APIs with streaming responses, token-based rate limiting, structured error codes, and versioning strategies that keep clients from breaking when your models change.
