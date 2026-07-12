---
title: Prompt Versioning & Management
description: >-
  Learn how to version, test, and manage prompts as first-class software
  artifacts in production LLM systems
duration: 30 min
difficulty: intermediate
has_code: false
youtube: 'https://www.youtube.com/watch?v=dOxUroR57xs'
objectives:
  - Explain why prompts need version control
  - Implement a prompt registry with versioning
  - Set up A/B testing between prompt versions
  - Use parameterized prompt templates
  - Describe rollback strategies for prompt changes
---
# Prompt Versioning & Management

## What You'll Learn

By the end of this lesson, you'll understand:
- Why prompts are code and deserve version control
- How to build a prompt registry system
- Parameterized templates for reusable prompts
- Testing and validating prompt changes before deployment
- Rollback strategies when a prompt update goes wrong

**Time to Complete**: 30 minutes
**Difficulty**: Intermediate

---

## Why Version Your Prompts?

In production LLM applications, prompts are your most important code. A single word change can:

- **Improve accuracy by 20%** or break your entire application
- **Change output format**, causing downstream parsing failures
- **Introduce bias** or safety issues you didn't anticipate
- **Increase costs** dramatically with longer prompts

Yet many teams treat prompts as strings in code, with no versioning, testing, or rollback capability.

### The Prompt Management Problem

```
Week 1: "Summarize this document in 3 bullet points"
Week 2: "Summarize this document concisely in 3-5 bullet points, focusing on key findings"
Week 3: "As an expert analyst, summarize this document..." (broke the JSON parser)
Week 4: "Wait, what was the prompt that worked last month?"
```

Sound familiar? This is why we need systematic prompt management.

---

## Prompt as Code

### Principle 1: Prompts Are First-Class Artifacts

Treat prompts like code:
- **Version controlled** in git
- **Tested** before deployment
- **Reviewed** by team members
- **Documented** with expected behavior
- **Monitored** in production

### Principle 2: Separate Prompts from Application Code

```python
# Bad: Prompt embedded in application logic
def summarize(text):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a helpful assistant. Summarize the following text in 3 bullet points."}]
    )

# Good: Prompt loaded from a registry
def summarize(text):
    prompt = prompt_registry.get("summarize", version="latest")
    response = client.chat.completions.create(
        model=prompt.model,
        messages=[{"role": "system", "content": prompt.render(text=text)}]
    )
```

---

## Building a Prompt Registry

A prompt registry stores, versions, and serves prompts to your application.

### Simple File-Based Registry

```python
# prompts/summarize/v1.yaml
name: summarize
version: 1
model: gpt-4o-mini
template: |
  Summarize the following text in {{num_points}} bullet points.
  Focus on key findings and actionable insights.

  Text: {{text}}
parameters:
  num_points:
    type: integer
    default: 3
metadata:
  author: "team"
  created: "2024-11-01"
  description: "General-purpose text summarization"
  tags: ["summarization", "general"]
```

### Python Prompt Registry Implementation

```python
import yaml
import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

@dataclass
class PromptVersion:
    name: str
    version: int
    model: str
    template: str
    parameters: dict
    metadata: dict

    def render(self, **kwargs) -> str:
        """Render the template with provided parameters."""
        result = self.template
        # Apply defaults for missing parameters
        for param, config in self.parameters.items():
            if param not in kwargs and "default" in config:
                kwargs[param] = config["default"]
        # Replace template variables
        for key, value in kwargs.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
        return result

class PromptRegistry:
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = Path(prompts_dir)
        self._cache = {}

    def get(self, name: str, version: Optional[int] = None) -> PromptVersion:
        """Get a prompt by name and optional version."""
        prompt_dir = self.prompts_dir / name

        if version is None:
            # Get latest version
            versions = sorted(prompt_dir.glob("v*.yaml"))
            if not versions:
                raise ValueError(f"No versions found for prompt '{name}'")
            version_file = versions[-1]
        else:
            version_file = prompt_dir / f"v{version}.yaml"

        cache_key = str(version_file)
        if cache_key not in self._cache:
            with open(version_file) as f:
                data = yaml.safe_load(f)
            self._cache[cache_key] = PromptVersion(**data)

        return self._cache[cache_key]

    def list_versions(self, name: str) -> list[int]:
        """List all available versions for a prompt."""
        prompt_dir = self.prompts_dir / name
        versions = []
        for f in sorted(prompt_dir.glob("v*.yaml")):
            v = int(f.stem[1:])  # Extract version number from "v1", "v2", etc.
            versions.append(v)
        return versions
```

---

## Parameterized Prompt Templates

Use templates to make prompts reusable across different contexts:

```python
# Template with parameters
CLASSIFICATION_TEMPLATE = """
You are a {{role}} specializing in {{domain}}.

Classify the following {{item_type}} into one of these categories:
{{categories}}

Rules:
- Return ONLY the category name
- If uncertain, choose the closest match
- Consider {{context_hint}} when making your decision

{{item_type}}: {{input}}
"""

# Usage
prompt = CLASSIFICATION_TEMPLATE.replace("{{role}}", "content moderator")
prompt = prompt.replace("{{domain}}", "social media")
prompt = prompt.replace("{{item_type}}", "post")
prompt = prompt.replace("{{categories}}", "- Safe\
- Needs Review\
- Violation")
prompt = prompt.replace("{{context_hint}}", "cultural context and sarcasm")
prompt = prompt.replace("{{input}}", user_post)
```

---

## Testing Prompt Changes

Before deploying a prompt change, validate it against a test suite:

```python
class PromptTester:
    def __init__(self, client, test_cases: list[dict]):
        self.client = client
        self.test_cases = test_cases

    def run_tests(self, prompt: PromptVersion) -> dict:
        """Run all test cases against a prompt version."""
        results = {"passed": 0, "failed": 0, "errors": []}

        for case in self.test_cases:
            rendered = prompt.render(**case["input"])
            response = self.client.chat.completions.create(
                model=prompt.model,
                messages=[{"role": "user", "content": rendered}],
                temperature=0  # Deterministic for testing
            )
            output = response.choices[0].message.content

            if case["validator"](output):
                results["passed"] += 1
            else:
                results["failed"] += 1
                results["errors"].append({
                    "input": case["input"],
                    "expected": case.get("expected"),
                    "actual": output
                })

        return results

# Example test suite
test_cases = [
    {
        "input": {"text": "AI is transforming healthcare with better diagnostics."},
        "validator": lambda output: len(output.split("\
")) >= 1,
        "expected": "At least one bullet point"
    },
    {
        "input": {"text": "", "num_points": 3},
        "validator": lambda output: "cannot" in output.lower() or "no text" in output.lower(),
        "expected": "Graceful handling of empty input"
    },
]
```

---

## Rollback Strategies

When a prompt update causes issues in production:

### 1. Instant Rollback
```python
# Feature flag approach
PROMPT_VERSION = os.environ.get("SUMMARIZE_PROMPT_VERSION", "latest")

# To rollback: set SUMMARIZE_PROMPT_VERSION=2
prompt = registry.get("summarize", version=PROMPT_VERSION)
```

### 2. Gradual Rollout
```python
import random

def get_prompt_with_rollout(name, new_version, rollout_pct=10):
    """Gradually roll out a new prompt version."""
    if random.randint(1, 100) <= rollout_pct:
        return registry.get(name, version=new_version)
    else:
        return registry.get(name)  # Current stable version
```

### 3. Automated Rollback
```python
def monitor_and_rollback(name, new_version, quality_threshold=0.8):
    """Automatically rollback if quality drops."""
    metrics = get_prompt_metrics(name, new_version, window="1h")

    if metrics["success_rate"] < quality_threshold:
        set_active_version(name, new_version - 1)  # Rollback
        alert_team(f"Auto-rolled back {name} from v{new_version}")
```

---

## Resources

- **LangSmith** — Prompt versioning and testing platform from LangChain
- **Humanloop** — Prompt management with evaluation and monitoring
- **PromptLayer** — Prompt version control and observability
- **Braintrust** — Prompt playground with automated evals

---

## Key Takeaways

1. **Prompts are code** — version, test, and review them like software
2. **Separate prompts from logic** — use a registry pattern
3. **Parameterize templates** — make prompts reusable across contexts
4. **Test before deploy** — run validation suites on prompt changes
5. **Plan for rollback** — always have a way to revert quickly
