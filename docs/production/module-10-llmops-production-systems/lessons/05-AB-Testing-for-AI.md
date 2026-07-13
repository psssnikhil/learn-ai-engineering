---
title: A/B Testing for AI Applications
description: >-
  Learn how to design and run A/B tests for LLM applications, comparing prompts,
  models, and configurations with statistical rigor
duration: 40 min
difficulty: intermediate
has_code: true
module: module-10
---
# A/B Testing for AI Applications

## Prerequisites

- Completed Lessons 1–4 (LLMOps Introduction through Caching)
- Basic familiarity with probability and averages
- Understanding that LLM outputs are non-deterministic

---

## What You'll Learn

| Objective | Outcome |
|-----------|---------|
| Explain why LLM experiments require different design than traditional A/B tests | Can identify the unique failure modes in AI experimentation |
| Design a hypothesis-driven experiment for an LLM change | Can write a complete experiment spec before writing any code |
| Implement deterministic user-to-variant assignment | Can ensure the same user always gets the same variant |
| Track primary and guard metrics simultaneously | Can detect quality improvements without sacrificing latency or cost |
| Interpret statistical significance for LLM metrics | Can avoid calling winners prematurely |

---

## Intuition First: Why "It Felt Better" Isn't Good Enough

Your team updates the system prompt for a customer support bot. In a quick internal test session, responses seem noticeably more helpful. Everyone agrees the new version is better. You ship it.

Three weeks later you notice user satisfaction scores are down 4%. The prompt that felt better to your team—who know the product deeply and ask polished test questions—performs worse for real users who ask ambiguous, truncated, or context-free queries.

This is the **internal tester bias**. Developers and product managers are not representative users. A/B testing on real traffic with proper measurement is the only way to know whether a change actually helps.

A/B testing for LLM applications also has a unique challenge: you're testing systems that are **non-deterministic and multi-dimensional**. A prompt change might improve answer quality but increase latency and cost. A model upgrade might improve accuracy but hurt the specific tone your brand requires. You need to measure all of these simultaneously.

---

## What You Can A/B Test

| Variable | Control | Treatment | Goal |
|----------|---------|-----------|------|
| **System prompt** | Minimal instructions | Detailed persona + format rules | Does structure improve quality? |
| **Model** | gpt-4o-mini | gpt-4o | Is quality worth the cost increase? |
| **Temperature** | 0.2 | 0.7 | Does higher creativity hurt accuracy? |
| **RAG top-k** | Retrieve 3 chunks | Retrieve 5 chunks | Do more chunks improve faithfulness? |
| **RAG chunking** | 512-token chunks | 256-token chunks | Do smaller chunks reduce noise? |
| **Output format** | Prose response | Bullet-point response | Which do users prefer? |
| **Max tokens** | 500 token limit | 200 token limit | Does brevity improve satisfaction? |

Test **one variable at a time**. Running multi-variable experiments (called factorial designs) requires 4× the sample size and expertise to interpret interaction effects. Start simple.

---

## Designing a Rigorous LLM Experiment

Good experiment design happens before writing any code.

### Step 1: Write a Hypothesis

A hypothesis is a falsifiable prediction with an expected magnitude:

```
WEAK:   "The new prompt will be better."

STRONG: "Replacing the open-ended system prompt with a structured prompt
         that includes explicit output format instructions will increase
         user satisfaction rate (thumbs-up ratio) by at least 8 percentage
         points (from 82% to 90%), without increasing P99 latency by more
         than 10% or per-request cost by more than 15%."
```

The strong version forces you to define: what metric you're optimizing, what improvement you need to declare success, and what guard rails protect against regressions.

### Step 2: Choose Primary and Guard Metrics

```python
experiment_spec = {
    "name": "structured_prompt_v2_test",
    "hypothesis": (
        "A structured prompt with explicit output format instructions "
        "increases thumbs-up rate from 82% to ≥90%"
    ),
    "variants": {
        "control": {
            "prompt_name": "support_agent",
            "prompt_version": 2,
            "model": "gpt-4o-mini",
        },
        "treatment": {
            "prompt_name": "support_agent",
            "prompt_version": 3,     # New structured version
            "model": "gpt-4o-mini",
        },
    },
    "traffic_split": {"control": 50, "treatment": 50},
    "primary_metric": "user_satisfaction_rate",
    "success_threshold": 0.90,            # Treatment must reach ≥90%
    "min_lift_pct": 8,                    # Must be ≥8 percentage points better
    "guard_metrics": {
        "latency_p99_ms": {"max_regression_pct": 10},   # Cannot increase p99 by >10%
        "cost_per_request_usd": {"max_regression_pct": 15},  # Cannot cost >15% more
        "error_rate_pct": {"max_regression_pct": 0},         # Zero tolerance for errors
    },
    "min_samples_per_variant": 500,
    "min_duration_days": 7,
    "statistical_alpha": 0.05,
}
```

Guard metrics prevent local optimization. A prompt that gets higher satisfaction by being more verbose might fail the cost guard rail—and you want to catch that before declaring it a winner.

### Step 3: Implement Deterministic Variant Assignment

Users must be **consistently assigned** to the same variant throughout the experiment. If a user gets control on Monday and treatment on Tuesday, their behavior is confounded and your results are meaningless.

```python
import hashlib
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    name: str
    variants: dict[str, dict]
    traffic_split: dict[str, int]   # Must sum to 100

class ABExperiment:
    """
    Deterministic A/B experiment using hash-based user bucketing.
    The same (experiment_name, user_id) pair always maps to the same variant.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        assert sum(config.traffic_split.values()) == 100, \
            "Traffic split must sum to 100"

    def assign_variant(self, user_id: str) -> str:
        """
        Hash the experiment name + user ID to a 0-99 bucket.
        Deterministic: same user always gets the same variant.
        Isolated: changing the experiment name generates a new assignment.
        """
        hash_input = f"{self.config.name}:{user_id}"
        hash_int = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)
        bucket = hash_int % 100

        cumulative = 0
        for variant, percentage in self.config.traffic_split.items():
            cumulative += percentage
            if bucket < cumulative:
                return variant

        return list(self.config.variants.keys())[-1]

    def get_config(self, user_id: str) -> dict:
        """Return the full config for a user's assigned variant."""
        variant = self.assign_variant(user_id)
        return {
            "variant": variant,
            "experiment": self.config.name,
            **self.config.variants[variant],
        }


# Usage in your request handler
experiment = ABExperiment(ExperimentConfig(
    name="structured_prompt_v2_test",
    variants={
        "control": {"prompt_version": 2, "model": "gpt-4o-mini"},
        "treatment": {"prompt_version": 3, "model": "gpt-4o-mini"},
    },
    traffic_split={"control": 50, "treatment": 50},
))

def handle_request(user_id: str, message: str) -> str:
    config = experiment.get_config(user_id)
    prompt = registry.get("support_agent", version=config["prompt_version"])
    response = llm_client.chat_completion(
        model=config["model"],
        messages=[{"role": "user", "content": prompt.render(user_query=message)}],
    )
    # Log the variant so you can compute per-variant metrics
    log_event(experiment=config["experiment"], variant=config["variant"],
              user_id=user_id, latency_ms=..., cost_usd=..., response=response)
    return response
```

!!! warning "The New User vs Returning User Problem"
    Users who interact with your application for the first time behave differently from returning users. If your experiment launches mid-week and most new users arrive on weekends, you might see quality differences that are actually day-of-week effects. Stratify by new vs. returning user status and run for at least one full week-cycle.

---

## Tracking and Aggregating Metrics

```python
from collections import defaultdict
from datetime import datetime

class ExperimentTracker:
    """In-memory event store for experiment metrics. In production, use a database."""

    def __init__(self):
        self._events: list[dict] = []

    def log(self, experiment: str, variant: str, user_id: str,
            metric: str, value: float, timestamp: datetime | None = None):
        self._events.append({
            "experiment": experiment,
            "variant": variant,
            "user_id": user_id,
            "metric": metric,
            "value": value,
            "timestamp": (timestamp or datetime.now()).isoformat(),
        })

    def summary(self, experiment: str) -> dict:
        """Compute per-variant statistics for each metric."""
        variant_data: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for event in self._events:
            if event["experiment"] == experiment:
                variant_data[event["variant"]][event["metric"]].append(event["value"])

        result = {}
        for variant, metrics in variant_data.items():
            result[variant] = {}
            for metric, values in metrics.items():
                n = len(values)
                mean = sum(values) / n
                result[variant][metric] = {
                    "mean": round(mean, 4),
                    "n": n,
                    "min": round(min(values), 4),
                    "max": round(max(values), 4),
                }
        return result
```

---

## Statistical Significance: When Can You Call a Winner?

Calling a winner too early is called **peeking** or **p-hacking**. Even if your treatment looks 10% better after 50 samples, you need more data before that difference is reliable.

```python
from scipy import stats
import numpy as np

def analyze_experiment(
    control_values: list[float],
    treatment_values: list[float],
    alpha: float = 0.05,
    min_samples: int = 200,
) -> dict:
    """
    Two-sample t-test for experiment analysis.
    Returns significance result and effect size.
    """
    n_control = len(control_values)
    n_treatment = len(treatment_values)

    if n_control < min_samples or n_treatment < min_samples:
        return {
            "status": "insufficient_data",
            "message": f"Need ≥{min_samples} samples per variant. "
                       f"Have {n_control} (control) and {n_treatment} (treatment).",
            "significant": False,
        }

    control_arr = np.array(control_values)
    treatment_arr = np.array(treatment_values)

    t_stat, p_value = stats.ttest_ind(control_arr, treatment_arr, equal_var=False)
    control_mean = float(np.mean(control_arr))
    treatment_mean = float(np.mean(treatment_arr))
    lift_pct = (treatment_mean - control_mean) / control_mean * 100

    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(control_arr) + np.var(treatment_arr)) / 2)
    cohens_d = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0.0

    return {
        "status": "ready",
        "significant": bool(p_value < alpha),
        "p_value": round(float(p_value), 4),
        "control_mean": round(control_mean, 4),
        "treatment_mean": round(treatment_mean, 4),
        "lift_pct": round(lift_pct, 2),
        "cohens_d": round(float(cohens_d), 3),
        "confidence_level": f"{(1 - alpha) * 100:.0f}%",
        "n_control": n_control,
        "n_treatment": n_treatment,
        "conclusion": (
            f"Treatment {'significantly' if p_value < alpha else 'does NOT significantly'} "
            f"outperforms control (p={p_value:.4f}, lift={lift_pct:+.1f}%)"
        ),
    }

# Example output:
# {
#   "status": "ready",
#   "significant": True,
#   "p_value": 0.0031,
#   "control_mean": 0.820,
#   "treatment_mean": 0.891,
#   "lift_pct": 8.66,
#   "cohens_d": 0.41,
#   "conclusion": "Treatment significantly outperforms control (p=0.0031, lift=+8.66%)"
# }
```

### Practical Sample Size Guidelines

Use a power calculator before starting an experiment. As a rule of thumb:

| Expected Lift | Min Samples Per Variant | Min Duration |
|--------------|------------------------|--------------|
| 2% (small effect) | ~4,000 | 2–3 weeks |
| 5% (medium effect) | ~650 | 1–2 weeks |
| 10% (large effect) | ~175 | 1 week minimum |
| 20%+ (very large) | ~50 | Still run 1 week for time effects |

Never stop an experiment early because one variant looks better. Statistical flukes in small samples commonly produce large-looking effects that disappear with more data.

---

## A Worked Experiment: Prompt vs Model

Let's trace through a complete experiment comparing model cost vs quality:

**Question**: Is GPT-4o worth approximately 10× the cost of GPT-4o-mini for our support bot?

```python
cost_vs_quality_experiment = ABExperiment(ExperimentConfig(
    name="gpt4o_vs_mini_support_q1",
    variants={
        "mini": {
            "model": "gpt-4o-mini",
            "prompt_version": 3,
            # Estimated cost per request: ~$0.003
        },
        "gpt4o": {
            "model": "gpt-4o",
            "prompt_version": 3,
            # Estimated cost per request: ~$0.030
        },
    },
    traffic_split={"mini": 90, "gpt4o": 10},  # Small canary for the expensive model
))
```

After 1 week with 1,000 requests per variant:

| Metric | gpt-4o-mini (control) | gpt-4o (treatment) | Δ |
|--------|----------------------|-------------------|---|
| User satisfaction | 84.2% | 87.1% | +2.9 pp |
| P-value | — | — | 0.031 (significant) |
| P99 latency | 4.2s | 6.8s | +62% |
| Cost/request | $0.003 | $0.031 | +933% |

**Decision**: The quality improvement is statistically significant but small (+2.9 percentage points). The cost is 10× higher and latency increases 62%. For this support bot workload, the trade-off does not favor GPT-4o. Continue with gpt-4o-mini.

This is precisely the kind of evidence-based decision that A/B testing enables.

---

## Common Pitfalls

**Testing too many things at once**: Each additional variable doubles the required sample size. Start with the change most likely to matter and test others sequentially.

**Ignoring interaction effects**: A prompt change may work brilliantly with gpt-4o but fail with gpt-4o-mini. If you change both simultaneously, you cannot diagnose the problem.

**Novelty effect**: Users sometimes engage more with any change simply because it's new. Run experiments for at least one week, ideally two, to let novelty wear off.

**Selection bias in the golden set**: If you let your internal team "feel out" which variant seems better before committing to a metrics-based decision, you've introduced evaluator bias. Commit to the metrics definition before seeing results.

**Not testing for regressions in edge cases**: The treatment might improve average-case behavior while worsening tail behavior. Track not just average satisfaction but also the rate of very low scores (e.g., the proportion of thumbs-down ratings).

---

## Production Scenario: Testing a Tone Change in Customer Support

Your customer support bot uses a professional, formal tone. The product team hypothesizes that a friendlier, conversational tone would increase user satisfaction. Here is how you run this correctly.

### Step 1: Write the Hypothesis Before Touching Any Code

```
Hypothesis: Switching from formal to conversational tone will increase
user satisfaction (thumbs-up rate) by 5–10 percentage points while
maintaining resolution rate and not increasing average response length.

Primary metric:   thumbs_up_rate (minimum detectable effect: 4pp)
Guard metrics:    resolution_rate (max allowed drop: 2pp),
                  avg_response_length_tokens (max allowed increase: 20%),
                  p99_latency_ms (max allowed increase: 5%)

Sample size:      ~500 per variant (based on 80% power, 5% significance)
Run duration:     14 days minimum (two full week cycles)
Rollback trigger: thumbs_up_rate drops > 5pp in first 48h
```

Committing this to a document before starting ensures you cannot later change the success criteria to match the outcome.

### Step 2: Build the Two Variants

```python
# Variant A: Current formal tone (control)
PROMPT_A = """
You are a professional customer service representative.
Provide accurate, concise assistance. Use formal language.
Address the customer's inquiry directly without unnecessary preamble.
"""

# Variant B: Conversational tone (treatment)
PROMPT_B = """
You are a friendly customer service assistant who genuinely wants to help.
Be warm and conversational. Use contractions (I'll, you'll, let's).
Acknowledge the customer's situation before diving into the solution.
End with an offer to help further if needed.
"""
```

### Step 3: Monitor Leading Indicators in the First 48 Hours

Before reaching statistical significance on the primary metric, watch for signals that the treatment is failing:

```python
EARLY_STOPPING_RULES = {
    "error_rate": {
        "description": "Treatment producing more errors",
        "threshold": "error_rate_B > error_rate_A * 1.5",
        "action": "Rollback immediately",
    },
    "guardrail_violations": {
        "description": "Conversational tone slipping into inappropriate content",
        "threshold": "violation_rate_B > 0.02",
        "action": "Rollback immediately",
    },
    "dramatic_satisfaction_drop": {
        "description": "Users strongly disliking the new tone",
        "threshold": "thumbs_up_rate_B < thumbs_up_rate_A - 0.10",
        "action": "Investigate; rollback if sustained for 12h",
    },
}
```

### Step 4: Read the Results With Full Context

After 14 days with 600 users per variant:

| Metric | Control (Formal) | Treatment (Conversational) | Delta |
|--------|-----------------|---------------------------|-------|
| Thumbs-up rate | 78.4% | 83.1% | +4.7 pp |
| P-value | — | — | 0.018 ✓ |
| Resolution rate | 91.2% | 90.8% | -0.4 pp |
| Avg tokens/response | 142 | 168 | +18% |
| P99 latency | 4.1s | 4.4s | +7% |

**Analysis**: Satisfaction improved significantly (+4.7pp, p=0.018). Guard metrics show resolution rate dropped only 0.4pp (within the 2pp limit). Latency increased 7%, just above the 5% guard threshold.

**Decision**: Ship the conversational tone. The latency increase is within acceptable UX range (under 5.5 seconds p99) and the satisfaction gain clearly outweighs it. Document the guard threshold miss as a learning: tighten the max_tokens parameter to reduce verbosity without losing the friendly tone.

This is exactly how data-driven decisions are made: acknowledge trade-offs explicitly, document the reasoning, and ship with confidence.

---

## Key Takeaways

- A/B test every LLM change that could affect user experience: prompts, models, RAG configurations, and output formats
- Write a hypothesis with specific expected magnitude before starting; this prevents post-hoc rationalization of ambiguous results
- Use deterministic hash-based user assignment: the same user always gets the same variant, preventing contamination
- Always track guard metrics (latency, cost, error rate) alongside the primary metric; a quality win that doubles cost may not be worth it
- Require statistical significance (p < 0.05) with at least 200 samples per variant before calling a winner
- Run for a minimum of 7 days to account for day-of-week behavioral patterns

---

## Further Reading

- [Trustworthy Online Controlled Experiments](https://arxiv.org/abs/1710.03028) — Kohavi et al.; the canonical reference on A/B testing at scale
- [Evaluating LLMs is Harder Than You Think](https://arxiv.org/abs/2306.04751) — Academic analysis of why LLM evaluation requires different statistical approaches
- [Statistical Methods for A/B Testing](https://arxiv.org/abs/1710.03028) — Foundational stats for experimentation practitioners
- [Braintrust documentation](https://www.braintrust.dev/docs) — Production AI experimentation platform with built-in statistical analysis

---

## Next Lesson

**Lesson 6: Cost Optimization** — Learn caching, model routing, prompt compression, and batch processing strategies that can reduce LLM API costs by 50–90% without sacrificing quality.
