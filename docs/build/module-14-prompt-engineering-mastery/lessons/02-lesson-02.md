---
title: Few-Shot and Chain-of-Thought Prompting
description: >-
  Master few-shot examples and chain-of-thought reasoning to dramatically
  improve LLM output quality
duration: 45 min
difficulty: intermediate
has_code: true
module: module-14
---
# Few-Shot and Chain-of-Thought Prompting

## Prerequisites

Before this lesson you should be comfortable with:

- **Prompt anatomy** — role, context, task, format, constraints (Lesson 1)
- **Basic LLM API calls** — sending messages and reading responses
- **Python string formatting** — f-strings and multiline strings

You do not need machine learning background. Few-shot and chain-of-thought are prompting techniques, not training methods.

---

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Contrast zero-shot, few-shot, and many-shot prompting | 10 min | Intermediate |
| Select and curate examples that teach the model your pattern | 10 min | Intermediate |
| Apply chain-of-thought (CoT) to reasoning and math tasks | 10 min | Intermediate |
| Combine few-shot + CoT and run a classification pipeline in Python | 15 min | Intermediate |

---

## Intuition First: Show, Don't Just Tell

Imagine onboarding a new analyst to classify customer support tickets. You could hand them a rulebook (zero-shot). Or you could sit beside them and walk through five real tickets: "Here's how I'd classify this one, and here's why."

The second approach — **learning from examples** — is what few-shot prompting does. You embed input-output pairs directly in the prompt. The model pattern-matches against them at inference time.

Chain-of-thought adds one more layer: you don't just show the answer, you show the **reasoning path**. "This ticket mentions a double charge and billing — that's Finance, High priority, escalate." The model learns both the label and the decision logic.

Together, these two techniques are among the highest-ROI changes you can make to a production prompt.

---

## Zero-Shot vs Few-Shot Prompting

### Zero-Shot: No Examples

You give the model a task with no examples. Works well for simple, well-defined tasks where the model already understands the domain.

```python
from openai import OpenAI

client = OpenAI()

def classify_zero_shot(review: str) -> str:
    prompt = f"""Classify this customer review as Positive, Negative, or Neutral.

Review: "{review}"
Classification:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

# classify_zero_shot("The product arrived on time but the packaging was damaged.")
# → "Neutral"
```

### Few-Shot: Learning from Examples

You provide input-output pairs. The model learns the pattern and applies it to new inputs.

```python
FEW_SHOT_TEMPLATE = """Classify each customer review as Positive, Negative, or Neutral.

Review: "Absolutely love this product! Best purchase ever."
Classification: Positive

Review: "Terrible quality. Broke after one day. Want a refund."
Classification: Negative

Review: "It's okay. Does what it says, nothing special."
Classification: Neutral

Review: "{review}"
Classification:"""

def classify_few_shot(review: str) -> str:
    prompt = FEW_SHOT_TEMPLATE.format(review=review)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()
```

Few-shot is more reliable because the model sees your exact label definitions in action — including how you handle ambiguous cases.

### How Many Examples?

| Examples | Name | Best For |
|----------|------|----------|
| 0 | Zero-shot | Simple tasks, clear instructions |
| 1–3 | Few-shot | Most tasks, establishing format |
| 5–10 | Many-shot | Complex classification, nuanced output |
| 10+ | Many-shot | Specialized domains, precise formatting |

**Rule of thumb**: Start with 3 diverse examples that cover edge cases. Add more only when eval shows specific failure patterns.

---

## Few-Shot Best Practices

### 1. Choose Diverse Examples

```python
# BAD: All examples are similar — model only sees positive
bad_examples = [
    ("Great product!", "Positive"),
    ("Love it!", "Positive"),
    ("Amazing!", "Positive"),
]

# GOOD: Cover the full range
good_examples = [
    ("Great product! Exceeded expectations.", "Positive"),
    ("Broken on arrival. Worst purchase.", "Negative"),
    ("It works. Nothing remarkable.", "Neutral"),
]
```

### 2. Match Your Target Distribution

If 60% of real inputs are negative, your examples should roughly reflect that. Skewed examples teach skewed behavior.

### 3. Include Edge Cases

```python
edge_case_examples = [
    ("Best thing I ever bought!", "Positive"),
    ("Total waste of money.", "Negative"),
    ("Good product but overpriced.", "Neutral"),       # Mixed → Neutral
    ("Not bad at all, surprisingly.", "Positive"),     # Double negative → Positive
    ("I guess it's fine...", "Neutral"),               # Lukewarm → Neutral
]
```

Edge-case examples are how you encode **your** business definitions, not the model's default assumptions.

---

## Chain-of-Thought (CoT) Prompting

Chain-of-thought prompting asks the model to show its reasoning step by step. This dramatically improves performance on logic, math, and multi-step reasoning.

### Without CoT

```python
no_cot = """
Q: A store has 15 red shirts and 8 blue shirts. They sell
3/5 of the red shirts and 1/4 of the blue shirts. How many
shirts remain in total?

A:"""
# Model might jump to a wrong answer without intermediate steps
```

### With CoT

```python
with_cot = """
Q: A store has 15 red shirts and 8 blue shirts. They sell
3/5 of the red shirts and 1/4 of the blue shirts. How many
shirts remain in total?

A: Let me work through this step by step.

Step 1: Calculate red shirts sold
- Started with 15 red shirts
- Sold 3/5 of them: 15 × 3/5 = 9 red shirts sold
- Red shirts remaining: 15 - 9 = 6

Step 2: Calculate blue shirts sold
- Started with 8 blue shirts
- Sold 1/4 of them: 8 × 1/4 = 2 blue shirts sold
- Blue shirts remaining: 8 - 2 = 6

Step 3: Total remaining
- 6 red + 6 blue = 12 shirts remaining

The answer is 12 shirts."""
```

### The Magic Phrase

Simply adding **"Let's think step by step"** can significantly improve reasoning:

```python
zero_shot_cot = """
Q: If it takes 5 machines 5 minutes to make 5 widgets,
how long would it take 100 machines to make 100 widgets?

Let's think step by step.
"""
# Without CoT, many models answer "100 minutes" (wrong!)
# With CoT: 5 minutes — each machine makes 1 widget in 5 minutes
```

---

## Combining Few-Shot + CoT

The most powerful technique: show examples **with** reasoning chains.

```python
TICKET_CLASSIFIER = """You are a customer support classifier. Classify tickets
and explain your reasoning.

Ticket: "I can't log in. I've tried resetting my password 3 times."
Reasoning: The customer is experiencing a login issue and has already
attempted self-service (password reset) multiple times. This indicates
a possible account lockout or system bug, not just a forgotten password.
Category: Technical - Account Access
Priority: High
Escalate: Yes (multiple failed self-service attempts)

Ticket: "When is your Black Friday sale?"
Reasoning: The customer is asking about an upcoming promotion. This is
a straightforward information request with no urgency.
Category: Sales - Inquiry
Priority: Low
Escalate: No

Ticket: "I was charged twice for my subscription this month."
Reasoning:"""
```

The model will follow the same structured reasoning pattern for the new ticket.

---

## Runnable Example: Building a Few-Shot Classifier

```python
from openai import OpenAI

client = OpenAI()

EXAMPLES = [
    ("Great product! Exceeded expectations.", "Positive"),
    ("Broken on arrival. Worst purchase.", "Negative"),
    ("It works. Nothing remarkable.", "Neutral"),
    ("Good quality but overpriced.", "Neutral"),
]

def build_few_shot_prompt(text: str, examples: list[tuple[str, str]]) -> str:
    lines = ["Classify sentiment as Positive, Negative, or Neutral.\n"]
    for input_text, label in examples:
        lines.append(f'Text: "{input_text}"')
        lines.append(f"Sentiment: {label}\n")
    lines.append(f'Text: "{text}"')
    lines.append("Sentiment:")
    return "\n".join(lines)

def classify(text: str) -> str:
    prompt = build_few_shot_prompt(text, EXAMPLES)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip()

# Quick eval loop
test_cases = [
    ("Absolutely love it!", "Positive"),
    ("Total waste of money", "Negative"),
    ("It's fine I guess", "Neutral"),
]

for text, expected in test_cases:
    result = classify(text)
    status = "✓" if expected.lower() in result.lower() else "✗"
    print(f"{status} Expected: {expected}, Got: {result}")
```

---

## When to Use Each Technique

```
Is the task simple and well-defined?
  ├── Yes → Zero-shot (just clear instructions)
  └── No
        ├── Does it need specific output format?
        │     └── Yes → Few-shot (show format examples)
        └── Does it require reasoning/logic?
              ├── Yes → Chain-of-thought
              └── Both format + reasoning?
                    └── Few-shot + CoT (most powerful)
```

---

## Production Connection

Few-shot and CoT patterns appear everywhere in production AI:

- **Version your example sets** — examples are part of the prompt artifact. When you change examples, bump the version and re-run eval.
- **A/B test example selection** — swapping 3 examples for 5 different ones can move accuracy ±10%. Test before deploying.
- **Eval loops** — maintain a golden set of 20–50 inputs with expected outputs. Run it after every prompt change.
- **Failure recovery** — if CoT output is correct but format is wrong, add a final instruction: "After reasoning, output ONLY the classification label on the last line."
- **Token budget** — each example costs tokens. In high-volume APIs, 5 examples × 10K requests/day adds up. Measure accuracy vs. cost.

!!! warning "Example contamination"
    Never put production user data directly into few-shot examples without review. Use curated, anonymized examples stored in your prompt registry.

---

## Edge Cases & Common Misconceptions

**Misconception 1: More examples always help.**
Beyond 5–10 examples, returns diminish and token costs rise. Quality and diversity matter more than quantity.

**Misconception 2: CoT is only for math.**
CoT improves any task requiring multi-step reasoning: legal analysis, code debugging, medical triage, financial calculations.

**Misconception 3: You must show reasoning in production output.**
You can ask the model to reason internally and output only the final answer: "Think step by step, then respond with only the classification label."

**Misconception 4: Few-shot examples must match input length.**
Short examples work fine for long inputs. What matters is that the **pattern** (format, labels, edge-case handling) is clear.

---

## Key Takeaways

- Zero-shot works for simple tasks; few-shot teaches the model your exact pattern and label definitions.
- Choose 3 diverse examples covering all categories and at least one edge case.
- Chain-of-thought dramatically improves reasoning — "Let's think step by step" is surprisingly effective.
- Few-shot + CoT combined is the strongest pattern for complex classification and routing tasks.
- Version example sets, A/B test changes, and run eval loops before deploying to production.
- Control token cost by limiting example count and stripping reasoning from final output when not needed.
- Match example distribution to real-world class distribution to avoid biased predictions.
- Diagnose failures by checking whether the model misunderstood the task (zero-shot issue) or your definitions (few-shot issue).

---

## Next Lesson

**[Lesson 3: System Prompts and Role Design](03-lesson-03.md)** — Learn how to design system prompts that control LLM behavior, personality, and output format across an entire conversation.
