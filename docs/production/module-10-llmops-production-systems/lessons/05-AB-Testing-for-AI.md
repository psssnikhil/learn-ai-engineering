---
title: A/B Testing for AI Applications
description: >-
  Learn how to design and run A/B tests for LLM applications, comparing prompts,
  models, and configurations with statistical rigor
duration: 30 min
difficulty: intermediate
has_code: false
module: module-10
objectives:
  - Design an A/B test for comparing two prompt variants
  - Implement a traffic splitting mechanism
  - Calculate statistical significance for LLM experiments
  - Track and compare key metrics across variants
  - Avoid common pitfalls in AI experimentation
---
# A/B Testing for AI Applications

## What You'll Learn

By the end of this lesson, you'll understand:
- How to design rigorous A/B tests for LLM systems
- Traffic splitting and experiment assignment
- Metrics to track: quality, latency, cost, user satisfaction
- Statistical significance for AI experiments
- Common pitfalls and how to avoid them

**Time to Complete**: 30 minutes
**Difficulty**: Intermediate

---

## Why A/B Test AI Systems?

Unlike traditional software where code is deterministic, LLM applications have inherent variability. A/B testing helps you:

- **Compare prompts**: Which system prompt produces better answers?
- **Compare models**: Is GPT-4o worth 10x the cost over GPT-4o-mini?
- **Compare configurations**: Does temperature 0.3 or 0.7 work better?
- **Measure impact**: Did that RAG improvement actually help users?

### What You Can A/B Test

| Variable | Example |
|----------|---------|
| Prompt wording | "Summarize concisely" vs. "Give a brief overview" |
| System prompt | Detailed persona vs. minimal instructions |
| Model | GPT-4o vs. Claude Sonnet |
| Temperature | 0.0 vs. 0.3 vs. 0.7 |
| RAG strategy | Top-3 chunks vs. Top-5 chunks |
| Output format | Bullet points vs. paragraphs |

---

## Designing an LLM A/B Test

### Step 1: Define Your Hypothesis

```
Hypothesis: Using a structured system prompt with explicit output format
instructions will increase user satisfaction by 15% compared to the
current minimal prompt.
```

### Step 2: Choose Your Metrics

**Primary metrics** (what you're optimizing for):
- User satisfaction rating (thumbs up/down)
- Task completion rate
- Response accuracy (human eval or automated)

**Secondary metrics** (guard rails):
- Latency (don't make things slower)
- Cost per request (stay within budget)
- Error rate (don't break things)

### Step 3: Implement the Experiment

```python
import hashlib
import random

class ABExperiment:
    def __init__(self, name: str, variants: dict, traffic_split: dict):
        self.name = name
        self.variants = variants      # {"control": config_a, "treatment": config_b}
        self.traffic_split = traffic_split  # {"control": 50, "treatment": 50}

    def assign_variant(self, user_id: str) -> str:
        """Deterministically assign a user to a variant."""
        # Hash ensures same user always gets same variant
        hash_input = f"{self.name}:{user_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        bucket = hash_value % 100

        cumulative = 0
        for variant, percentage in self.traffic_split.items():
            cumulative += percentage
            if bucket < cumulative:
                return variant

        return list(self.variants.keys())[-1]

    def get_config(self, user_id: str) -> dict:
        """Get the configuration for a user's assigned variant."""
        variant = self.assign_variant(user_id)
        return {"variant": variant, **self.variants[variant]}


# Set up an experiment
experiment = ABExperiment(
    name="prompt_v2_test",
    variants={
        "control": {
            "system_prompt": "You are a helpful assistant.",
            "model": "gpt-4o-mini"
        },
        "treatment": {
            "system_prompt": "You are a senior AI tutor. Always structure your answers with: 1) A direct answer, 2) An explanation, 3) An example.",
            "model": "gpt-4o-mini"
        }
    },
    traffic_split={"control": 50, "treatment": 50}
)
```

---

## Tracking Metrics

```python
from datetime import datetime

class ExperimentTracker:
    def __init__(self):
        self.events = []

    def log_event(self, experiment: str, variant: str, user_id: str,
                  metric: str, value: float):
        self.events.append({
            "experiment": experiment,
            "variant": variant,
            "user_id": user_id,
            "metric": metric,
            "value": value,
            "timestamp": datetime.now().isoformat()
        })

    def get_summary(self, experiment: str) -> dict:
        """Compute per-variant metric averages."""
        from collections import defaultdict
        variant_metrics = defaultdict(lambda: defaultdict(list))

        for event in self.events:
            if event["experiment"] == experiment:
                variant_metrics[event["variant"]][event["metric"]].append(event["value"])

        summary = {}
        for variant, metrics in variant_metrics.items():
            summary[variant] = {
                metric: {
                    "mean": sum(values) / len(values),
                    "count": len(values),
                    "min": min(values),
                    "max": max(values),
                }
                for metric, values in metrics.items()
            }
        return summary
```

---

## Statistical Significance

Don't call a winner too early. Use proper statistics:

```python
from scipy import stats
import numpy as np

def is_significant(control_values, treatment_values, alpha=0.05):
    """Check if the difference between variants is statistically significant."""
    t_stat, p_value = stats.ttest_ind(control_values, treatment_values)

    return {
        "p_value": p_value,
        "significant": p_value < alpha,
        "control_mean": np.mean(control_values),
        "treatment_mean": np.mean(treatment_values),
        "lift": (np.mean(treatment_values) - np.mean(control_values)) / np.mean(control_values) * 100
    }
```

### Sample Size Guidelines

- **Minimum 100 observations per variant** for reliable results
- **Run for at least 1 week** to account for daily patterns
- **Don't peek** and stop early when results look good (p-hacking)

---

## Common Pitfalls

1. **Testing too many things at once** — Change one variable at a time
2. **Not accounting for LLM randomness** — Run multiple evaluations per input
3. **Ignoring cost** — A 5% quality gain might not justify 3x the cost
4. **User segment bias** — Ensure variant assignment is truly random
5. **Short experiment duration** — Run long enough for statistical power

---

## Resources

- **Braintrust** — AI-native experimentation platform
- **Statsig** — Feature flags and A/B testing (supports AI use cases)
- **Weights & Biases** — Experiment tracking for ML/AI

---

## Key Takeaways

1. **A/B test everything** — prompts, models, configs, RAG strategies
2. **Deterministic assignment** — hash user IDs for consistent variant assignment
3. **Track multiple metrics** — quality, latency, cost, user satisfaction
4. **Wait for significance** — don't call winners prematurely
5. **One variable at a time** — isolate the effect of each change
