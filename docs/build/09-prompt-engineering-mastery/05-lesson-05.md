---
title: Prompt Templates and Variables
description: >-
  Build reusable, testable prompt templates with dynamic variables, conditional
  logic, and version control
duration: 35 min
difficulty: intermediate
has_code: false
---
# Prompt Templates and Variables

## Learning Objectives

By the end of this lesson, you will be able to:
- Build reusable prompt templates with dynamic variables
- Use Jinja2 and f-strings for prompt templating
- Implement prompt versioning for production systems
- Test prompt templates systematically

---

## Why Templates?

Hardcoded prompts become unmanageable as your application grows. Templates separate the structure from the data:

```python
# Hardcoded (fragile, untestable)
prompt = f"Summarize this {language} code:
{code}
Focus on {aspect}."

# Template (reusable, testable, versionable)
SUMMARIZE_TEMPLATE = """Summarize this {language} code:

```{language}
{code}
```

Focus on: {aspect}
Output format: {format}
"""
```

---

## Simple Templates with Python

```python
from string import Template

# Using string.Template (safe against injection)
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
    input_text=quarterly_report
)
```

---

## Jinja2 for Complex Templates

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
    format="JSON"
)
```

---

## Prompt Registry Pattern

```python
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PromptVersion:
    template: str
    version: str
    created_at: str
    description: str

class PromptRegistry:
    def __init__(self):
        self._prompts: dict[str, list[PromptVersion]] = {}

    def register(self, name: str, template: str, version: str, description: str = ""):
        if name not in self._prompts:
            self._prompts[name] = []
        self._prompts[name].append(PromptVersion(
            template=template,
            version=version,
            created_at=datetime.utcnow().isoformat(),
            description=description,
        ))

    def get(self, name: str, version: str = "latest") -> str:
        versions = self._prompts.get(name, [])
        if not versions:
            raise KeyError(f"Prompt '{name}' not found")
        if version == "latest":
            return versions[-1].template
        for v in versions:
            if v.version == version:
                return v.template
        raise KeyError(f"Version '{version}' not found for prompt '{name}'")

# Usage
registry = PromptRegistry()
registry.register(
    "summarize",
    "Summarize the following text in {num_sentences} sentences:

{text}",
    version="1.0",
    description="Basic summarization prompt"
)
registry.register(
    "summarize",
    "Summarize the following {doc_type} in {num_sentences} sentences. "
    "Focus on key insights and actionable items.

{text}",
    version="2.0",
    description="Added doc_type and focus instructions"
)

template = registry.get("summarize", version="2.0")
```

---

## Testing Prompt Templates

```python
import pytest

def test_summarize_template_renders():
    template = registry.get("summarize", "2.0")
    result = template.format(
        doc_type="research paper",
        num_sentences=3,
        text="Sample text here."
    )
    assert "research paper" in result
    assert "3 sentences" in result
    assert "Sample text here." in result

def test_summarize_template_no_missing_vars():
    template = registry.get("summarize", "2.0")
    # This should not raise KeyError
    required_vars = {"doc_type", "num_sentences", "text"}
    for var in required_vars:
        assert f"{{{var}}}" in template
```

---

## Key Takeaways

- Prompt templates separate structure from data, making prompts reusable and testable
- Use Python f-strings for simple cases, Jinja2 for conditional logic and loops
- A prompt registry enables versioning and rollback in production
- Test templates like code: verify rendering, required variables, and edge cases
- Store prompts as configuration, not hardcoded in application logic

## Resources

- [LangChain: Prompt Templates](https://python.langchain.com/docs/concepts/prompt_templates/) -- Framework-level prompt management
- [YouTube: Production Prompt Management](https://www.youtube.com/watch?v=dOxUroR57xs) -- Patterns for managing prompts at scale
- [Jinja2 Documentation](https://jinja.palletsprojects.com/) -- Template engine reference

---

Next: Prompt Chaining and Pipelines
