---
title: Prompt Templates and Variables
description: >-
  Build reusable, testable prompt templates with dynamic variables, conditional
  logic, and version control
duration: 35 min
difficulty: intermediate
has_code: true
module: module-14
---
# Prompt Templates and Variables

## Prerequisites

Before this lesson you should be comfortable with:

- **Prompt anatomy** — structuring prompts with clear sections (Lesson 1)
- **System prompts** — separating rules from variable data (Lesson 3)
- **Python string formatting** — f-strings, `.format()`, and basic templating

You do not need Jinja2 experience. This lesson introduces it alongside simpler alternatives.

---

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Explain why hardcoded prompts fail at scale | 8 min | Intermediate |
| Build templates with f-strings, string.Template, and Jinja2 | 12 min | Intermediate |
| Implement a prompt registry with versioning | 8 min | Intermediate |
| Write unit tests for prompt templates | 7 min | Intermediate |

---

## Intuition First: The Mail Merge Analogy

Imagine sending 10,000 personalized emails by manually editing each one. You would never do that — you'd use mail merge: one template, variables for name, order number, and shipping date.

Hardcoded prompts in application code are the same anti-pattern. When your summarization prompt appears in 14 files and you need to change the output format, you hunt through the codebase and hope you didn't miss one.

Templates separate **structure** (the prompt design) from **data** (the user's input, context, and configuration). That separation makes prompts reusable, testable, and versionable — the foundation of production prompt management.

---

## Why Templates?

```python
# Hardcoded (fragile, untestable, duplicated)
prompt = f"Summarize this {language} code:\n{code}\nFocus on {aspect}."

# Template (reusable, testable, versionable)
SUMMARIZE_TEMPLATE = """Summarize this {language} code:

```{language}
{code}
```

Focus on: {aspect}
Output format: {format}
Max length: {max_words} words
"""
```

When the product team asks for bullet-point summaries instead of paragraphs, you change one template — not fourteen string literals scattered across services.

---

## Simple Templates with Python

### f-strings — Quick and Direct

```python
def build_summary_prompt(
    language: str,
    code: str,
    aspect: str,
    fmt: str = "paragraph",
    max_words: int = 150,
) -> str:
    return f"""Summarize this {language} code:

```{language}
{code}
```

Focus on: {aspect}
Output format: {fmt}
Max length: {max_words} words
"""
```

### string.Template — Safe Against Injection

`string.Template` uses `$variable` syntax and ignores `{` `}` in user content — important when templating code or JSON:

```python
from string import Template

template = Template("""You are a $role specializing in $domain.

Analyze the following and provide:
1. A summary in $language
2. Key findings (max $max_findings)
3. Risk assessment

Input:
$input_text
""")

prompt = template.substitute(
    role="financial analyst",
    domain="quarterly earnings",
    language="English",
    max_findings=5,
    input_text=quarterly_report,
)
```

Use `Template` when user-provided content might contain `{` or `$` characters that would break f-strings.

---

## Jinja2 for Complex Templates

When you need conditionals, loops, or defaults, Jinja2 is the standard choice:

```python
from jinja2 import Template

ANALYSIS_TEMPLATE = Template("""You are a {{ role }}.

{% if context %}
## Context
{{ context }}
{% endif %}

## Task
Analyze the following {{ doc_type }}:

{{ content }}

## Requirements
{% for req in requirements %}
- {{ req }}
{% endfor %}

{% if examples %}
## Examples
{% for example in examples %}
### Example {{ loop.index }}
Input: {{ example.input }}
Output: {{ example.output }}
{% endfor %}
{% endif %}

Respond in {{ format | default("plain text") }}.
""")

prompt = ANALYSIS_TEMPLATE.render(
    role="technical writer",
    context="This is for a developer audience",
    doc_type="API documentation",
    content=api_docs,
    requirements=["Check accuracy", "Suggest improvements", "Rate clarity 1-10"],
    examples=[
        {"input": "GET /users", "output": "Clear, well-documented endpoint"},
    ],
    format="JSON",
)
```

Jinja2 conditionals let you include few-shot examples only when available, or add context sections only when provided — without maintaining separate template files.

---

## Prompt Registry Pattern

```python
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import json

@dataclass
class PromptVersion:
    template: str
    version: str
    created_at: str
    description: str
    variables: list[str]

class PromptRegistry:
    def __init__(self, storage_dir: str = "./prompts"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self._cache: dict[str, list[PromptVersion]] = {}

    def register(
        self,
        name: str,
        template: str,
        version: str,
        description: str = "",
        variables: list[str] | None = None,
    ):
        entry = PromptVersion(
            template=template,
            version=version,
            created_at=datetime.now(timezone.utc).isoformat(),
            description=description,
            variables=variables or [],
        )
        self._cache.setdefault(name, []).append(entry)
        path = self.storage_dir / f"{name}_v{version}.json"
        path.write_text(json.dumps(asdict(entry), indent=2))

    def get(self, name: str, version: str = "latest") -> PromptVersion:
        versions = self._cache.get(name, [])
        if not versions:
            raise KeyError(f"Prompt '{name}' not found")
        if version == "latest":
            return versions[-1]
        for v in versions:
            if v.version == version:
                return v
        raise KeyError(f"Version '{version}' not found for '{name}'")

    def render(self, name: str, version: str = "latest", **kwargs) -> str:
        pv = self.get(name, version)
        missing = set(pv.variables) - set(kwargs.keys())
        if missing:
            raise ValueError(f"Missing template variables: {missing}")
        return pv.template.format(**kwargs)

# Usage
registry = PromptRegistry()
registry.register(
    "summarize",
    "Summarize the following {doc_type} in {num_sentences} sentences.\n\n{text}",
    version="1.0",
    description="Basic summarization",
    variables=["doc_type", "num_sentences", "text"],
)
registry.register(
    "summarize",
    "Summarize the following {doc_type} in {num_sentences} sentences. "
    "Focus on key insights and actionable items.\n\n{text}",
    version="2.0",
    description="Added focus instructions",
    variables=["doc_type", "num_sentences", "text"],
)

prompt = registry.render(
    "summarize", version="2.0",
    doc_type="research paper",
    num_sentences=3,
    text="Sample text here.",
)
```

---

## Testing Prompt Templates

Treat templates like code — test rendering, required variables, and edge cases:

```python
def test_summarize_v2_renders_all_variables():
    pv = registry.get("summarize", "2.0")
    result = registry.render(
        "summarize", "2.0",
        doc_type="research paper",
        num_sentences=3,
        text="Sample text.",
    )
    assert "research paper" in result
    assert "3 sentences" in result
    assert "Sample text." in result

def test_summarize_v2_requires_all_variables():
    try:
        registry.render("summarize", "2.0", doc_type="paper")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "num_sentences" in str(e)

def test_summarize_v2_no_unreplaced_placeholders():
    result = registry.render(
        "summarize", "2.0",
        doc_type="report", num_sentences=5, text="Content.",
    )
    assert "{" not in result, f"Unreplaced placeholder in: {result}"
```

Run these in CI alongside your application tests. A broken template variable name should fail the build, not production.

---

## Worked Example: Multi-Tier Template

Production apps often need different prompt behavior based on user tier or locale:

```python
from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader("./templates"))

TIER_TEMPLATE = env.from_string("""You are a {{ role }} for {{ product_name }}.

{% if user_tier == "enterprise" %}
This is an enterprise customer. Provide detailed technical responses.
Include architecture recommendations and link to enterprise documentation.
{% elif user_tier == "pro" %}
This is a pro customer. Provide thorough responses with code examples.
{% else %}
This is a free-tier customer. Keep responses concise and helpful.
Suggest upgrading for advanced features when relevant.
{% endif %}

## User Question
{{ question }}

{% if locale != "en" %}
Respond in {{ locale }} language.
{% endif %}
""")

prompt = TIER_TEMPLATE.render(
    role="customer support agent",
    product_name="CloudAPI",
    user_tier="enterprise",
    question="How do I set up webhook retries?",
    locale="en",
)
```

One template file handles three tiers and multiple locales. When enterprise support policy changes, you edit one file — not twelve service copies.

---

## Production Connection

Templates are the backbone of production prompt management:

- **Version every template change** — `summarize_v2.0` → `summarize_v2.1`. Never overwrite in place.
- **A/B test template variants** — route 10% of traffic to v2.1, compare accuracy and token usage against v2.0.
- **Eval loops** — render templates with eval set inputs, send to LLM, measure output quality. Automate this on every template PR.
- **Failure recovery** — if rendering fails (missing variable), log the error and fall back to the previous template version rather than sending a malformed prompt.
- **Separate storage from code** — store templates in a registry (file, database, or prompt management platform), not inline in application logic. Engineers update prompts without redeploying code.

---

## Edge Cases & Common Misconceptions

**Misconception 1: f-strings are always fine.**
If user input contains `{` or `}`, f-strings break or inject unintended variables. Use `string.Template` or Jinja2 for user-controlled content.

**Misconception 2: Templates don't need tests.**
A renamed variable (`{num_sentences}` → `{sentence_count}`) silently produces prompts with literal `{sentence_count}` in the text. Unit tests catch this.

**Misconception 3: One template per task is enough.**
Production systems often need locale variants, tier-specific instructions (free vs. enterprise), and model-specific phrasing. Template parameters handle this without duplicating files.

**Misconception 4: Jinja2 is overkill for simple prompts.**
It is — for simple cases. But the moment you need "include examples if provided" or "switch format based on user tier," Jinja2 saves significant complexity.

---

## Key Takeaways

- Templates separate prompt structure from variable data, enabling reuse and centralized updates.
- Use f-strings for simple internal templates; use `string.Template` or Jinja2 when user content may contain special characters.
- Jinja2 handles conditionals, loops, and defaults — essential for dynamic few-shot example injection.
- A prompt registry with versioning lets you pin, roll back, and A/B test template changes.
- Test templates in CI: verify rendering, required variables, and no unreplaced placeholders.
- Store templates as configuration artifacts, not hardcoded strings in application code.
- Log template name + version with every LLM call for debugging and cost attribution.
- Fail gracefully: fall back to the previous template version if rendering fails.

---

## Next Lesson

**[Lesson 6: Prompt Chaining and Pipelines](06-lesson-06.md)** — Learn to decompose complex tasks into multi-step prompt chains where each step builds on the previous output.
