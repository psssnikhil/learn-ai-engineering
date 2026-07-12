---
title: Prompt Engineering Fundamentals
description: >-
  Understand how prompts interact with the Transformer's probability distribution,
  master the core prompting patterns, learn to iterate with systematic evaluation,
  and understand the limits of what prompt engineering can and cannot do
duration: 75 min
difficulty: beginner
has_code: true
module: module-01
---
# Prompt Engineering Fundamentals

## Prerequisites

- [Lesson 02: Your First AI Application](02-first-ai-application.md) — Chat Completions API
- [Module 00 Lessons 05-08](../../module-00-genai-foundations-from-nlp-to-transformers/lessons/05-attention-mechanism.md) — attention mechanism, next-token prediction (recommended)

## What You'll Learn

| Objective | Why It Matters |
|-----------|---------------|
| Understand *why* prompts work mechanistically | Prevents cargo-cult prompting; helps you debug failures |
| Apply the core prompt anatomy systematically | Reduces prompt iteration time |
| Use few-shot learning correctly | The most reliable way to shift model behavior |
| Implement chain-of-thought prompting | Required for multi-step reasoning tasks |
| Use structured output prompts reliably | Critical for any AI system that parses model output |
| Recognize when prompt engineering is the wrong tool | Saves you weeks of frustration |

---

## Why Prompts Work: The Mechanistic View

To write effective prompts, it helps to understand what is happening mechanistically. A language model is a function that computes:

\[
P(\text{next token} \mid \text{all previous tokens})
\]

The prompt is just the "all previous tokens" part. Every word you include in the prompt shifts the probability distribution over the next token — and transitively, over the entire completion.

```
Prompt: "The capital of France is"
P(next token = "Paris"): 0.94    ← very high
P(next token = "the"):   0.02
P(next token = "Lyon"):  0.01

Prompt: "My friend said the capital of France is"
P(next token = "Paris"): 0.87
P(next token = "Brussels"): 0.05   ← slightly higher — reporting uncertainty
P(next token = "London"): 0.03

Prompt: "In the fictional country of Ruronia, the capital is"
P(next token = "Paris"): 0.08   ← much lower — fiction context shifts distribution
P(next token = "Ruronia-City"): 0.15
```

This means:
- **More context → more constrained distribution** (usually better)
- **Ambiguous prompts → high-entropy output** (unpredictable, often wrong)
- **Examples shift the distribution toward the pattern demonstrated** (few-shot learning)

!!! note "The Prompt Is Not Instructions to a Person"
    A helpful mental model: the model does not "read" your prompt and "decide" what to do. It continues the text. Everything after your prompt is what the model thinks is the statistically most likely continuation of that text. If your prompt looks like a document that ends with a helpful answer, the model produces helpful answers. If it looks like a forum argument, it may continue arguing.

---

## The Prompt Anatomy

A well-structured prompt has five components. Not all are needed for every task:

```
[ROLE]        — Who is producing this text?
[TASK]        — What should they do?
[CONTEXT]     — What do they need to know to do it?
[FORMAT]      — How should the output be structured?
[CONSTRAINTS] — What should they avoid or limit?
```

### Before and After: A Real Example

**Before (bad):**
```
Tell me about machine learning.
```
This is maximally ambiguous. The model has to guess: audience level, desired length, format, depth, perspective. The output will be generic.

**After (structured):**
```
[ROLE] You are a technical educator writing for software engineers with 3+ years of Python experience but no ML background.

[TASK] Explain gradient descent.

[CONTEXT] The reader understands calculus at a high school level and knows what a function is.

[FORMAT] Structure your response as:
1. One-sentence intuition (no jargon)
2. Mathematical definition with variable names defined
3. A runnable Python example (< 20 lines)
4. One common pitfall

[CONSTRAINTS] Do not use analogies involving hills, mountains, or terrain.
Limit the total response to 400 words.
```

The structured version constrains the distribution so tightly that almost any completion will be useful.

---

## Core Prompting Patterns

### Pattern 1: Role Assignment

Specifying a role shifts the model toward the writing style, vocabulary, and perspectives associated with that role in the training data:

```python
from openai import OpenAI

client = OpenAI()

def role_comparison(question: str):
    roles = [
        "You are a pirate. Answer in pirate dialect.",
        "You are a cautious senior software engineer with 15 years of experience. Answer conservatively, note risks.",
        "You are a startup founder trying to ship quickly. Answer pragmatically, minimize complexity.",
    ]

    for role in roles:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": question}
            ],
            max_tokens=100
        )
        print(f"\n[{role[:50]}...]")
        print(response.choices[0].message.content)

role_comparison("Should I use Redis or PostgreSQL for session storage?")
```

### Pattern 2: Few-Shot Learning

Provide examples of the input-output mapping you want. The model generalizes from the pattern:

```python
def few_shot_classifier(text_to_classify: str) -> str:
    """
    Classify customer support ticket priority.
    Few-shot examples define the classification scheme precisely.
    """
    few_shot_messages = [
        {
            "role": "system",
            "content": (
                "Classify customer support tickets as CRITICAL, HIGH, MEDIUM, or LOW priority. "
                "Return only the priority label, nothing else."
            )
        },
        # Example 1
        {"role": "user", "content": "Our entire production database is down. No customers can access the app."},
        {"role": "assistant", "content": "CRITICAL"},
        # Example 2
        {"role": "user", "content": "The export to CSV button is not working in Safari."},
        {"role": "assistant", "content": "MEDIUM"},
        # Example 3
        {"role": "user", "content": "Can you add dark mode to the dashboard?"},
        {"role": "assistant", "content": "LOW"},
        # Example 4
        {"role": "user", "content": "Login is failing for about 30% of users in the EU region."},
        {"role": "assistant", "content": "HIGH"},
        # Now the actual query
        {"role": "user", "content": text_to_classify},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=few_shot_messages,
        temperature=0,    # deterministic — classification should be consistent
        max_tokens=10,    # label only, no explanation
    )

    return response.choices[0].message.content.strip()

# Test cases
tickets = [
    "The payment gateway is throwing 500 errors.",
    "Can you update the font in the header to be larger?",
    "All API requests are timing out after 2 minutes.",
]

for ticket in tickets:
    priority = few_shot_classifier(ticket)
    print(f"{priority}: {ticket}")
```

!!! note "Why Few-Shot Works"
    From the Transformer perspective: the examples are tokens in the context. The model's attention mechanism "reads" these examples and adjusts its internal representations to match the pattern. This is called **in-context learning** — learning from the prompt without updating weights.

### Pattern 3: Chain of Thought (CoT)

For multi-step reasoning, instruct the model to think step by step before answering. This increases accuracy on math, logic, and planning tasks significantly.

```python
def solve_with_cot(problem: str) -> dict:
    """
    Use chain-of-thought prompting for a reasoning problem.
    Returns both the reasoning steps and the final answer.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Solve problems step by step. "
                "Show your work clearly before stating the final answer. "
                "Format as:\n"
                "Reasoning: <step-by-step thinking>\n"
                "Answer: <final answer only>"
            )
        },
        {"role": "user", "content": problem}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
    )

    text = response.choices[0].message.content

    # Parse the structured output
    reasoning = text.split("Answer:")[0].replace("Reasoning:", "").strip()
    answer    = text.split("Answer:")[-1].strip() if "Answer:" in text else text

    return {"reasoning": reasoning, "answer": answer}

result = solve_with_cot(
    "A store sells apples for $0.45 each and oranges for $0.60 each. "
    "If Maya buys 8 apples and 5 oranges and pays with a $10 bill, "
    "how much change does she receive?"
)

print("Reasoning:\n", result["reasoning"])
print("\nAnswer:", result["answer"])
```

**Why CoT works:** Generating the reasoning steps *before* the answer gives the model "scratchpad" space. Each reasoning token becomes part of the context for subsequent tokens, allowing the model to condition on intermediate conclusions it has already stated. Without CoT, the model must arrive at the final answer directly from the problem statement — a harder prediction task.

### Pattern 4: Structured Output

When your code parses the model's output, request a structured format explicitly. JSON is the most reliable choice:

```python
import json
from pydantic import BaseModel, ValidationError
from openai import OpenAI

class ProductReview(BaseModel):
    sentiment:    str   # "positive", "negative", "neutral"
    score:        float  # 1.0 to 5.0
    key_phrases:  list[str]  # up to 3 key phrases
    summary:      str   # one sentence
    actionable:   bool  # should the team respond?

def analyze_review(review_text: str) -> ProductReview:
    """
    Extract structured information from a product review.
    Uses Pydantic for validation — rejects malformed outputs.
    """
    schema_example = json.dumps(ProductReview.model_json_schema(), indent=2)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a review analysis API. Respond with valid JSON only, "
                f"matching this schema:\n{schema_example}\n\n"
                "Sentiment must be exactly 'positive', 'negative', or 'neutral'. "
                "Score must be a float between 1.0 and 5.0. "
                "Key phrases: list of up to 3 strings."
            )
        },
        {"role": "user", "content": f"Analyze this review:\n\n{review_text}"}
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},   # enforce JSON mode
    )

    raw = response.choices[0].message.content

    try:
        data = json.loads(raw)
        return ProductReview(**data)
    except (json.JSONDecodeError, ValidationError) as e:
        raise ValueError(f"Model returned invalid JSON: {e}\nRaw output: {raw}")

# Test
review = "The build quality is excellent and setup was painless, but the companion app crashes constantly. Would not buy again."
result = analyze_review(review)
print(f"Sentiment: {result.sentiment} ({result.score}/5.0)")
print(f"Key phrases: {result.key_phrases}")
print(f"Needs response: {result.actionable}")
```

---

## Systematic Prompt Iteration

Good prompts come from measurement, not intuition. Build a small evaluation set and test changes:

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class EvalCase:
    input: str
    expected: str
    check: Callable[[str, str], bool] = lambda output, expected: expected.lower() in output.lower()

def evaluate_prompt(system_prompt: str, eval_cases: list[EvalCase],
                    model: str = "gpt-4o-mini") -> dict:
    """
    Run a prompt against all eval cases and return metrics.
    """
    results = []

    for case in eval_cases:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": case.input}
            ],
            temperature=0,
            max_tokens=200,
        )
        output = response.choices[0].message.content
        passed = case.check(output, case.expected)
        results.append({"passed": passed, "input": case.input,
                        "expected": case.expected, "output": output})

    accuracy = sum(r["passed"] for r in results) / len(results)
    failures = [r for r in results if not r["passed"]]

    return {"accuracy": accuracy, "failures": failures, "n": len(results)}

# Define eval cases for a classification task
eval_cases = [
    EvalCase("Production DB down", "CRITICAL"),
    EvalCase("Can't export to PDF", "MEDIUM"),
    EvalCase("Add emoji reactions", "LOW"),
    EvalCase("Login 500 errors for 20% of users", "HIGH"),
]

# Baseline prompt
baseline = "Classify tickets as CRITICAL, HIGH, MEDIUM, or LOW."
metrics_v1 = evaluate_prompt(baseline, eval_cases)
print(f"Baseline accuracy: {metrics_v1['accuracy']:.0%}")

# Improved prompt
improved = """
Classify customer support tickets:
CRITICAL: System down, data loss risk, affecting all users
HIGH:     Major feature broken, affecting significant portion of users
MEDIUM:   Feature degraded, workaround exists
LOW:      Nice-to-have, non-urgent improvement

Return only the priority label.
"""
metrics_v2 = evaluate_prompt(improved, eval_cases)
print(f"Improved accuracy: {metrics_v2['accuracy']:.0%}")
```

The key discipline: **change one thing at a time** between iterations. If you change both the role description and the examples, you cannot know which change helped.

---

## When Prompt Engineering Is the Wrong Tool

Prompt engineering has real limits. Knowing these saves significant time:

| Problem | Why Prompting Won't Fix It | Better Approach |
|---------|---------------------------|-----------------|
| Model lacks knowledge of recent events | Training cutoff is fixed; prompting cannot add new knowledge | RAG: retrieve and inject current information |
| Task requires > 3-4 reasoning hops reliably | Models still struggle with long chains of inference | Fine-tune or break into sub-tasks |
| Consistent formatting across 1000s of outputs | Probabilistic outputs have variance | Use `response_format={"type": "json_object"}` + validation |
| Specific domain terminology the model has not seen | No amount of prompting adds vocabulary | Fine-tune on domain text |
| Task requires precise computation | LLMs are probabilistic; math can fail | Use tool calling (code interpreter, calculator) |

---

## Edge Cases and Misconceptions

**"Longer prompts are always better."** Not true. Verbose prompts consume more tokens and can distract the model from the key instruction. Instructions buried in paragraph 5 of a long system prompt often get less "attention" than clear instructions at the top.

**"Adding 'think step by step' always helps."** Chain-of-thought helps for reasoning tasks but adds tokens and latency for simple tasks (classification, extraction). Apply it selectively.

**"Temperature=0 makes outputs deterministic."** Almost always, but see previous lesson. For truly critical classification tasks, consider running at temperature=0 and validating the output matches the expected schema, retrying if it does not.

**"The system prompt is never seen by users."** Technically true in most interfaces, but prompt injection attacks attempt to override the system prompt through user input. Never trust user input that claims to modify system behavior.

---

## Production Connection

| Pattern | Where It Appears in Production |
|---------|-------------------------------|
| **Few-shot examples** | Classification systems, extraction pipelines, formatting consistency |
| **Chain-of-thought** | Complex analysis, code review, multi-step reasoning |
| **Structured output** | Any AI component that feeds into downstream code |
| **Prompt versioning** | AI engineering workflows version prompts alongside code; changes tracked in git |
| **A/B testing prompts** | Route X% of traffic to prompt variant B; measure quality metrics |

---

## Key Takeaways

- Prompts work by constraining the token probability distribution, not by giving instructions to a reasoning agent
- The core anatomy — role, task, context, format, constraints — reduces iteration time by making each component independently adjustable
- Few-shot learning is the most reliable way to specify output format and classification behavior
- Chain-of-thought works because intermediate reasoning tokens become context for subsequent tokens
- Structured output (JSON) + schema validation is required for any AI component that feeds into downstream code
- Build an evaluation set before iterating on prompts — you need measurement to know if a change helped

---

## Further Reading

- [Anthropic Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering) — production patterns with Claude; most techniques transfer across models
- [Wei et al. (2022): Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) — the original CoT paper with empirical results
- [Brown et al. (2020): Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3 paper showing in-context learning scales with model size
- [Prompt Engineering Guide](https://www.promptingguide.ai/) — community-maintained reference covering advanced techniques

---

**Next:** [Working with LLM APIs — Production Guide](05-working-with-apis.md)
