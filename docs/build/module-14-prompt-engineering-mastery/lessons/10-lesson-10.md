---
title: Production Prompt Engineering
description: >-
  Learn end-to-end practices for managing prompts in production systems
  including monitoring, versioning, and continuous improvement
duration: 40 min
difficulty: advanced
has_code: false
module: module-14
---
# Production Prompt Engineering

## Learning Objectives

By the end of this lesson, you will be able to:
- Set up a production prompt management workflow
- Monitor prompt performance with logging and metrics
- Implement prompt A/B testing in production
- Build a continuous improvement pipeline for prompts

---

## The Production Prompt Lifecycle

```
Development → Evaluation → Staging → Production → Monitoring → Iteration
     ↑                                                              │
     └──────────────────────────────────────────────────────────────┘
```

---

## Prompt Versioning in Production

```python
import json
from pathlib import Path
from datetime import datetime

class PromptManager:
    def __init__(self, prompts_dir: str = "./prompts"):
        self.prompts_dir = Path(prompts_dir)
        self.prompts_dir.mkdir(exist_ok=True)

    def save(self, name: str, template: str, metadata: dict = None):
        """Save a new prompt version."""
        versions_dir = self.prompts_dir / name
        versions_dir.mkdir(exist_ok=True)

        # Auto-increment version
        existing = sorted(versions_dir.glob("v*.json"))
        version = len(existing) + 1

        prompt_data = {
            "template": template,
            "version": version,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }

        filepath = versions_dir / f"v{version}.json"
        filepath.write_text(json.dumps(prompt_data, indent=2))
        return version

    def load(self, name: str, version: int = None) -> str:
        """Load a prompt template by name and version."""
        versions_dir = self.prompts_dir / name
        if version:
            filepath = versions_dir / f"v{version}.json"
        else:
            # Load latest
            files = sorted(versions_dir.glob("v*.json"))
            filepath = files[-1]

        data = json.loads(filepath.read_text())
        return data["template"]

# Usage
pm = PromptManager()
pm.save("sentiment", "Classify sentiment: {text}", {"accuracy": 0.72})
pm.save("sentiment", "Classify as positive/negative/neutral:

{text}", {"accuracy": 0.89})

current_prompt = pm.load("sentiment")  # loads v2
stable_prompt = pm.load("sentiment", version=1)  # loads v1
```

---

## Logging and Monitoring

```python
import time
import logging

logger = logging.getLogger("prompt_monitor")

class MonitoredLLMCall:
    def __init__(self, client):
        self.client = client

    def call(self, prompt_name: str, prompt_version: int,
             messages: list, **kwargs) -> dict:
        """Make an LLM call with full logging."""
        start = time.time()

        try:
            response = self.client.chat.completions.create(
                messages=messages, **kwargs
            )
            latency = time.time() - start
            output = response.choices[0].message.content

            # Log metrics
            logger.info(json.dumps({
                "prompt_name": prompt_name,
                "prompt_version": prompt_version,
                "model": kwargs.get("model", "unknown"),
                "latency_ms": int(latency * 1000),
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "cost_usd": self._estimate_cost(response.usage, kwargs.get("model")),
                "status": "success",
            }))

            return {"output": output, "latency": latency, "usage": response.usage}

        except Exception as e:
            logger.error(json.dumps({
                "prompt_name": prompt_name,
                "prompt_version": prompt_version,
                "status": "error",
                "error": str(e),
            }))
            raise

    def _estimate_cost(self, usage, model: str) -> float:
        rates = {
            "gpt-4o": {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
            "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
        }
        r = rates.get(model, {"input": 0, "output": 0})
        return usage.prompt_tokens * r["input"] + usage.completion_tokens * r["output"]
```

---

## Production A/B Testing

```python
import random

class PromptABTest:
    def __init__(self, prompt_a: str, prompt_b: str, traffic_split: float = 0.5):
        self.prompt_a = prompt_a
        self.prompt_b = prompt_b
        self.traffic_split = traffic_split
        self.results_a = []
        self.results_b = []

    def get_prompt(self) -> tuple[str, str]:
        """Return (prompt, variant_name) based on traffic split."""
        if random.random() < self.traffic_split:
            return self.prompt_a, "A"
        return self.prompt_b, "B"

    def record_result(self, variant: str, score: float):
        """Record a quality score for a variant."""
        if variant == "A":
            self.results_a.append(score)
        else:
            self.results_b.append(score)

    def get_stats(self) -> dict:
        """Get current test statistics."""
        return {
            "A": {
                "count": len(self.results_a),
                "avg_score": sum(self.results_a) / len(self.results_a) if self.results_a else 0,
            },
            "B": {
                "count": len(self.results_b),
                "avg_score": sum(self.results_b) / len(self.results_b) if self.results_b else 0,
            },
        }
```

---

## Continuous Improvement Checklist

| Step | Frequency | Action |
|------|-----------|--------|
| **Monitor accuracy** | Daily | Check automated eval scores |
| **Review failures** | Weekly | Analyze low-scoring responses |
| **Update eval set** | Bi-weekly | Add new edge cases from production |
| **Test new models** | Monthly | Benchmark against newer models |
| **Optimize cost** | Monthly | Check if cheaper model achieves same quality |
| **Update prompts** | As needed | Iterate based on failure analysis |

---

## Key Takeaways

- Treat prompts as versioned artifacts, not hardcoded strings
- Log every LLM call with prompt name, version, latency, cost, and token usage
- A/B test prompt changes in production before full rollout
- Build a feedback loop: monitor, analyze failures, improve, redeploy
- Cost monitoring is essential -- small prompt changes can have large cost implications

## Resources

- [YouTube: LLM Ops in Production](https://www.youtube.com/watch?v=gHKlagKpN0k) -- Managing prompts at scale
- [PromptFoo](https://www.promptfoo.dev/) -- Open-source prompt evaluation and testing
- [LangSmith](https://smith.langchain.com/) -- LLM observability and prompt management platform
- [Humanloop](https://humanloop.com/) -- Prompt management and evaluation platform

---

## Module Complete!

You have completed the Prompt Engineering Mastery module. You can now design, test, optimize, and manage prompts for production AI applications.
