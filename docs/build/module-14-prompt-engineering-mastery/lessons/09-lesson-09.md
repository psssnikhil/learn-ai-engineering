---
title: Prompt Engineering for Different Models
description: >-
  Understand how to adapt prompts for different LLM providers including OpenAI,
  Anthropic, Google, and open-source models
duration: 35 min
difficulty: intermediate
has_code: true
module: module-14
---
# Prompt Engineering for Different Models

## Prerequisites

Before this lesson you should be comfortable with:

- **Prompt anatomy and system prompts** — Lessons 1 and 3
- **Structured output** — JSON mode and Pydantic schemas (Lesson 4)
- **Prompt templates** — building portable prompt structures (Lesson 5)

You do not need accounts with every provider. Code examples show API patterns; adapt to whichever providers you use.

---

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Compare prompting conventions across major LLM providers | 10 min | Intermediate |
| Adapt the same task prompt for OpenAI, Anthropic, and Google APIs | 10 min | Intermediate |
| Write portable prompts that work well across providers | 8 min | Intermediate |
| Choose the right model for a given task based on strengths and cost | 7 min | Intermediate |

---

## Intuition First: Same Recipe, Different Ovens

A recipe written for a convection oven may need temperature and timing adjustments for a conventional oven. The dish is the same; the equipment behaves differently.

Prompts work the same way. The core anatomy — role, task, format, constraints — transfers across models. But each model family has different strengths, context handling, and formatting preferences. GPT-4o responds well to numbered instructions. Claude excels with XML-tagged structure. Gemini handles multimodal and very long contexts.

Portable prompts use universal patterns (clear headers, numbered rules, explicit output format). Model-specific adapters handle the rest.

---

## Model Comparison

| Feature | GPT-4o | Claude 3.5 Sonnet | Gemini 1.5 Pro |
|---------|--------|-------------------|----------------|
| **Strengths** | General reasoning, code, instruction following | Long context, analysis, safety | Multimodal, long context, speed |
| **Context window** | 128K tokens | 200K tokens | 1M tokens |
| **System prompt** | `role: "system"` | `role: "system"` or `system` param | `system_instruction` param |
| **Structured output** | Native JSON mode + Pydantic | JSON mode, tool use | JSON mode |
| **Best for** | General-purpose, production APIs | Analysis, writing, long documents | Multimodal, very long contexts |

No single model wins every task. Production systems often use multiple models — a capable model for complex reasoning, a cheap model for classification, a long-context model for document analysis.

---

## OpenAI (GPT-4o)

```python
from openai import OpenAI

client = OpenAI()

def gpt_classify(text: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a concise sentiment classifier."},
            {"role": "user", "content": f"Classify sentiment (one word):\n\n{text}"},
        ],
        temperature=0,
        max_tokens=10,
    )
    return response.choices[0].message.content.strip()
```

**GPT-4o tips:**

- Responds well to numbered instructions and explicit output formats
- Use `temperature=0` for deterministic outputs (classification, extraction)
- Structured outputs with Pydantic are highly reliable
- Strong coding and general reasoning at competitive cost

---

## Anthropic (Claude)

```python
import anthropic

client = anthropic.Anthropic()

def claude_classify(text: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=10,
        system="You are a concise sentiment classifier. Respond with one word.",
        messages=[
            {"role": "user", "content": f"Classify sentiment:\n\n{text}"},
        ],
    )
    return response.content[0].text.strip()
```

**Claude tips:**

- Excels at following complex, multi-part instructions
- Responds well to XML tags for structuring input
- Use `<thinking>` tags to encourage step-by-step reasoning
- Very strong at long document analysis (200K context)

```python
def claude_code_review(code: str) -> str:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""Analyze this code for security issues:

<code>
{code}
</code>

Respond in this format:
<analysis>
<issues>List each issue</issues>
<severity>critical/warning/info</severity>
<fix>Corrected code</fix>
</analysis>""",
        }],
    )
    return response.content[0].text
```

---

## Google (Gemini)

```python
import google.generativeai as genai

genai.configure(api_key="your-api-key")

model = genai.GenerativeModel(
    "gemini-1.5-pro",
    system_instruction="You are a concise sentiment classifier.",
)

def gemini_classify(text: str) -> str:
    response = model.generate_content(f"Classify sentiment (one word):\n\n{text}")
    return response.text.strip()
```

**Gemini tips:**

- Best for multimodal tasks (images + text in one prompt)
- Handles very long contexts (up to 1M tokens)
- Use for analyzing large codebases or long documents in a single call

---

## Writing Portable Prompts

Prompts that work across models follow universal patterns:

```python
PORTABLE_TEMPLATE = """## Task
{task_description}

## Rules
1. {rule_1}
2. {rule_2}
3. {rule_3}

## Input
{input}

## Output Format
{format_description}
"""

def build_portable_prompt(task: str, rules: list[str], input_text: str, fmt: str) -> str:
    return PORTABLE_TEMPLATE.format(
        task_description=task,
        rule_1=rules[0],
        rule_2=rules[1],
        rule_3=rules[2],
        input=input_text,
        format_description=fmt,
    )
```

These patterns work across all major models:

- Clear section headers with `##`
- Numbered rules
- Explicit output format specification
- Concrete examples (few-shot) when format matters

### Model Adapter Pattern

Separate portable prompt content from provider-specific API calls:

```python
from dataclasses import dataclass
from typing import Protocol

@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int

class LLMProvider(Protocol):
    def complete(self, system: str, user: str, max_tokens: int = 256) -> LLMResponse: ...

class OpenAIProvider:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 256) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=0,
        )
        return LLMResponse(
            text=response.choices[0].message.content,
            model=self.model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

def classify_sentiment(provider: LLMProvider, text: str) -> str:
    """Same portable prompt, any provider."""
    system = "Classify sentiment. Respond with one word: positive, negative, or neutral."
    user = build_portable_prompt(
        task="Classify the sentiment of the input text.",
        rules=[
            "Respond with exactly one word",
            "positive = satisfaction or enthusiasm",
            "negative = dissatisfaction or frustration",
        ],
        input_text=text,
        fmt="One word: positive, negative, or neutral",
    )
    return provider.complete(system, user, max_tokens=10).text.strip()
```

---

## Choosing the Right Model

| Task | Best Model | Why |
|------|-----------|-----|
| Code generation | GPT-4o or Claude | Strong coding abilities |
| Long document analysis | Claude or Gemini | Large context windows |
| Classification/extraction | GPT-4o-mini | Fast, cheap, structured output |
| Image understanding | GPT-4o or Gemini | Native multimodal |
| Creative writing | Claude | Natural, nuanced writing |
| Cost-sensitive batch | GPT-4o-mini | Cheapest per token |

In production, benchmark your specific prompt against 2–3 models on your eval set. Published benchmarks reflect general tasks; your domain may differ.

---

## Production Connection

Multi-model production systems need disciplined management:

- **Version prompts per model** — the same logical prompt may have OpenAI and Claude variants with different formatting. Track `{prompt_name}_{provider}_v{version}`.
- **A/B test across models** — route 10% of traffic to a challenger model. Compare accuracy, latency, and cost on the same eval set.
- **Eval loops per provider** — a prompt scoring 94% on GPT-4o-mini may score 78% on a cheaper open-source model. Maintain separate eval baselines.
- **Failure recovery** — if the primary model times out or returns an error, fall back to a secondary provider with the same portable prompt.
- **Cost monitoring** — log cost per provider per prompt. Monthly reviews often reveal that 80% of spend goes to one task that could use a cheaper model.

---

## Edge Cases & Common Misconceptions

**Misconception 1: One prompt works identically everywhere.**
Portable structure transfers, but optimal phrasing varies. Budget time to adapt prompts when switching providers.

**Misconception 2: The most expensive model is always best.**
GPT-4o-mini often matches GPT-4o on classification and extraction at 10–20× lower cost. Benchmark on your eval set.

**Misconception 3: Model choice is permanent.**
New models release every few months. Re-benchmark quarterly; your current "best" model may no longer be.

**Misconception 4: Open-source models don't need prompt engineering.**
Open-source models often need more explicit instructions and stronger format constraints than frontier models.

---

## Key Takeaways

- Each model family has different strengths; adapt prompting style while keeping core anatomy portable.
- GPT-4o excels at structured outputs and instruction following; Claude at long-context analysis and XML structure; Gemini at multimodal and very long contexts.
- Portable prompts use clear headers, numbered rules, explicit output formats, and few-shot examples.
- Use a model adapter pattern to separate prompt content from provider-specific API calls.
- Benchmark your prompt on 2–3 models against your eval set before choosing a production model.
- Version prompts per provider; A/B test model switches with accuracy, latency, and cost metrics.
- Implement fallback to a secondary provider when the primary model fails or times out.
- Re-benchmark quarterly as new models release.

---

## Next Lesson

**[Lesson 10: Production Prompt Engineering](10-lesson-10.md)** — Learn end-to-end practices for managing prompts in production systems including monitoring, versioning, and continuous improvement.
