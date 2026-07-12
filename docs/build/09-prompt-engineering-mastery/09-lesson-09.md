---
title: Prompt Engineering for Different Models
description: >-
  Understand how to adapt prompts for different LLM providers including OpenAI,
  Anthropic, Google, and open-source models
duration: 35 min
difficulty: intermediate
has_code: false
---
# Prompt Engineering for Different Models

## Learning Objectives

By the end of this lesson, you will be able to:
- Adapt prompts for OpenAI (GPT-4o), Anthropic (Claude), and Google (Gemini) models
- Understand model-specific strengths and prompting conventions
- Write portable prompts that work well across providers
- Choose the right model for your specific task

---

## Model Comparison

| Feature | GPT-4o | Claude 3.5 Sonnet | Gemini 1.5 Pro |
|---------|--------|-------------------|----------------|
| **Strengths** | General reasoning, code, instruction following | Long context, analysis, safety | Multimodal, long context, speed |
| **Context window** | 128K tokens | 200K tokens | 1M tokens |
| **System prompt** | `role: "system"` | `role: "system"` or `system` param | `system_instruction` param |
| **Structured output** | Native JSON mode + Pydantic | JSON mode, tool use | JSON mode |
| **Best for** | General-purpose, production APIs | Analysis, writing, long documents | Multimodal, very long contexts |

---

## OpenAI (GPT-4o)

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a concise technical writer."},
        {"role": "user", "content": "Explain what a vector database is."}
    ],
    temperature=0.3,       # Lower = more deterministic
    max_tokens=500,
)
print(response.choices[0].message.content)
```

**GPT-4o tips:**
- Responds well to numbered instructions and explicit output formats
- Use `temperature=0` for deterministic outputs (classification, extraction)
- Structured outputs with Pydantic are very reliable

---

## Anthropic (Claude)

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    system="You are a concise technical writer.",
    messages=[
        {"role": "user", "content": "Explain what a vector database is."}
    ]
)
print(response.content[0].text)
```

**Claude tips:**
- Excels at following complex, multi-part instructions
- Responds well to XML tags for structuring input: `<document>...</document>`
- Use `<thinking>` tags to encourage step-by-step reasoning
- Very strong at long document analysis (200K context)

```python
# Claude-optimized: XML tags for structure
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{
        "role": "user",
        "content": """Analyze this code for security issues:

<code>
def login(username, password):
    query = f"SELECT * FROM users WHERE name='{username}' AND pass='{password}'"
    return db.execute(query)
</code>

Respond in this format:
<analysis>
<issues>List each issue</issues>
<severity>critical/warning/info</severity>
<fix>Corrected code</fix>
</analysis>"""
    }]
)
```

---

## Google (Gemini)

```python
import google.generativeai as genai

genai.configure(api_key="your-api-key")
model = genai.GenerativeModel(
    "gemini-1.5-pro",
    system_instruction="You are a concise technical writer."
)

response = model.generate_content("Explain what a vector database is.")
print(response.text)
```

**Gemini tips:**
- Best for multimodal tasks (images + text)
- Handles very long contexts (up to 1M tokens)
- Use for analyzing large codebases or long documents in a single call

---

## Writing Portable Prompts

Prompts that work across models follow these principles:

```python
# Portable prompt template
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

# These patterns work well across all major models:
# - Clear section headers with ## 
# - Numbered rules
# - Explicit output format
# - Concrete examples (few-shot)
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

---

## Key Takeaways

- Each model family has different strengths; adapt your prompting style accordingly
- GPT-4o excels at structured outputs and instruction following
- Claude excels at long-context analysis and responds well to XML-structured prompts
- Gemini handles multimodal inputs and very long contexts
- Portable prompts use clear headers, numbered rules, and explicit output formats
- For production, test your prompts across models to find the best quality-cost balance

## Resources

- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) -- GPT-specific techniques
- [Anthropic Prompt Engineering](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering) -- Claude-specific patterns
- [YouTube: Comparing LLM Providers](https://www.youtube.com/watch?v=xbgKMQGcRmk) -- Side-by-side model comparison

---

Next: Production Prompt Engineering
