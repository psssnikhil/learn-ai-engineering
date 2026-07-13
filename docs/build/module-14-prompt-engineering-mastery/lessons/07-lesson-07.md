---
title: Prompt Optimization and Iteration
description: >-
  Learn systematic methods to improve prompt performance through evaluation, A/B
  testing, and iterative refinement
duration: 35 min
difficulty: intermediate
has_code: true
module: module-14
---
# Prompt Optimization and Iteration

## Prerequisites

Before this lesson you should be comfortable with:

- **Prompt anatomy and few-shot prompting** — Lessons 1 and 2
- **Prompt templates and versioning** — Lesson 5
- **Basic Python** — loops, dictionaries, and simple metrics

You do not need statistics background. This lesson focuses on practical eval workflows, not rigorous experimental design.

---

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Build an evaluation dataset to measure prompt quality | 10 min | Intermediate |
| Run A/B tests comparing prompt variants systematically | 10 min | Intermediate |
| Apply progressive refinement techniques that improve most prompts | 8 min | Intermediate |
| Analyze failure cases to target the next iteration | 7 min | Intermediate |

---

## Intuition First: Tuning Without a Speedometer

Optimizing a prompt by "trying things until it looks better" is like tuning a car engine by ear with no speedometer. You might feel improvement, but you cannot prove it, reproduce it, or know when you have regressed.

Prompt optimization needs measurement. An **eval set** is your speedometer: a fixed list of inputs with known expected outputs. Run every prompt variant against the same eval set, compare scores, and keep the winner.

Most production teams iterate 5–15 times before a prompt reaches acceptable quality. The teams that ship fastest are not the ones with the best intuition — they are the ones with the best eval loops.

---

## The Prompt Optimization Loop

```
Write prompt → Test on eval set → Measure metrics → Identify failures → Refine → Repeat
```

Each iteration should change **one thing** so you know what caused the improvement (or regression). Change the examples, re-run eval. Change the constraints, re-run eval. Change both at once and you learn nothing.

---

## Building an Evaluation Set

```python
from openai import OpenAI

client = OpenAI()

eval_set = [
    {"input": "I absolutely love this product! Best purchase ever.", "expected": "positive"},
    {"input": "Terrible experience. Broke after one day.", "expected": "negative"},
    {"input": "It works fine. Nothing special but does the job.", "expected": "neutral"},
    {"input": "Good quality but way too expensive.", "expected": "neutral"},
    {"input": "Not bad at all, actually pretty good.", "expected": "positive"},
    {"input": "Great, another broken product. Thanks a lot.", "expected": "negative"},
    # Target 20-50 examples for reliable results
]

def llm_call(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content.strip().lower()

def evaluate_prompt(prompt_template: str, eval_set: list[dict]) -> dict:
    """Run a prompt against an evaluation set and measure accuracy."""
    correct = 0
    results = []

    for item in eval_set:
        prompt = prompt_template.format(text=item["input"])
        predicted = llm_call(prompt)
        is_correct = item["expected"] in predicted
        correct += int(is_correct)
        results.append({
            "input": item["input"][:50],
            "expected": item["expected"],
            "predicted": predicted,
            "correct": is_correct,
        })

    return {"accuracy": correct / len(eval_set), "results": results}
```

Start with 20 examples covering all categories and known edge cases. Add new failures from production weekly. The eval set is a living document, not a one-time artifact.

---

## A/B Testing Prompts

```python
prompt_v1 = "Classify the sentiment as positive, negative, or neutral:\n\n{text}"

prompt_v2 = """Classify the sentiment of the following text.

Rules:
- "positive": expresses satisfaction, happiness, or enthusiasm
- "negative": expresses dissatisfaction, frustration, or disappointment
- "neutral": factual, balanced, or neither positive nor negative

Text: {text}

Sentiment (respond with exactly one word):"""

results_v1 = evaluate_prompt(prompt_v1, eval_set)
results_v2 = evaluate_prompt(prompt_v2, eval_set)

print(f"v1 accuracy: {results_v1['accuracy']:.1%}")
print(f"v2 accuracy: {results_v2['accuracy']:.1%}")
# Typical: v1 ~72%, v2 ~85% after adding explicit rules
```

In production, A/B testing routes live traffic: 90% gets the stable prompt, 10% gets the challenger. Compare accuracy, latency, and cost over 24–48 hours before full rollout.

---

## Progressive Refinement

Most prompts improve through a predictable sequence:

```python
# Iteration 1: Basic (~72% accuracy)
v1 = "Classify sentiment: {text}"

# Iteration 2: Add definitions (~85% accuracy)
v2 = """Classify sentiment as positive, negative, or neutral.
positive = expresses satisfaction or enthusiasm
negative = expresses dissatisfaction or frustration
neutral = factual or balanced

Text: {text}
Sentiment:"""

# Iteration 3: Add examples (~91% accuracy)
v3 = """Classify sentiment as positive, negative, or neutral.

Examples:
"Best purchase ever!" → positive
"Broke after one day" → negative
"It does the job" → neutral

Text: {text}
Sentiment:"""

# Iteration 4: Handle edge cases (~94% accuracy)
v4 = """Classify sentiment as positive, negative, or neutral.

Rules:
- Sarcasm: classify by actual intent ("Great, another broken product" = negative)
- Mixed sentiment: default to the dominant emotion
- Questions without sentiment are neutral

Examples:
"Best purchase ever!" → positive
"Broke after one day" → negative
"It does the job" → neutral

Text: {text}
Sentiment (one word):"""

for version, prompt in [("v1", v1), ("v2", v2), ("v3", v3), ("v4", v4)]:
    result = evaluate_prompt(prompt, eval_set)
    print(f"{version}: {result['accuracy']:.1%}")
```

| Technique | When to Use | Typical Improvement |
|-----------|-------------|-------------------|
| Add explicit rules | Output is inconsistent | +10–20% accuracy |
| Add examples (few-shot) | Model misunderstands the task | +15–25% accuracy |
| Define output format | Responses vary in structure | Consistency, easier parsing |
| Add chain-of-thought | Reasoning tasks, math | +10–30% on complex tasks |
| Reduce ambiguity | Edge cases fail | +5–15% accuracy |
| Add constraints | Output too long/short | Controls output quality |

---

## Analyzing Failures

```python
def analyze_failures(eval_results: dict) -> None:
    """Print failed cases to identify patterns."""
    failures = [r for r in eval_results["results"] if not r["correct"]]
    print(f"\n{len(failures)} failures out of {len(eval_results['results'])} cases:\n")
    for f in failures:
        print(f"  Input:    {f['input']}")
        print(f"  Expected: {f['expected']}")
        print(f"  Got:      {f['predicted']}")
        print()

# Run after each iteration
analyze_failures(evaluate_prompt(v3, eval_set))

# Common failure patterns to look for:
# - Sarcasm misclassified
# - Mixed sentiment inconsistent
# - Edge cases (empty text, single words)
# - Format issues (extra words in output)
```

Group failures by pattern, not by individual case. "3 sarcasm failures" → add a sarcasm rule. "2 format failures" → add "respond with exactly one word." Targeted fixes beat random prompt edits.

---

## Production Connection

Prompt optimization is not a one-time task — it is a continuous process:

- **Version and track scores** — store `{prompt_version: accuracy}` in metadata. Never deploy a prompt without recording its eval score.
- **A/B test in staging first** — run the challenger against the full eval set. Only promote to production A/B if it wins on accuracy without regressing on latency or cost.
- **Eval loops on every PR** — CI runs the eval set against changed prompts. Block merge if accuracy drops below threshold.
- **Failure recovery** — when production accuracy drops, roll back to the previous prompt version immediately. Investigate with the eval set afterward.
- **Refresh eval sets bi-weekly** — add production failures as new test cases. Prompts that score 95% on a stale eval set may score 80% on current traffic.

---

## Edge Cases & Common Misconceptions

**Misconception 1: 100% eval accuracy means production-ready.**
Eval sets are samples. Overfitting to 20 examples is easy. Hold out a separate test set you never optimize against.

**Misconception 2: Bigger models fix bad prompts.**
A better model on a vague prompt might go from 60% to 75%. A refined prompt on a cheap model might go from 60% to 94%. Optimize the prompt first.

**Misconception 3: One metric is enough.**
Track accuracy, latency, token cost, and parse failure rate. A prompt that wins on accuracy but doubles cost may not be worth deploying.

**Misconception 4: Optimization ends at deployment.**
Production traffic surfaces new edge cases continuously. The eval loop never stops.

---

## Key Takeaways

- You cannot improve what you do not measure — build an eval set before optimizing.
- Change one variable per iteration so you know what caused improvement or regression.
- Progressive refinement: rules → examples → edge-case handling is the most common winning sequence.
- Analyze failures by pattern (sarcasm, format, ambiguity), not individual cases.
- A/B test in staging against the full eval set; canary in production before full rollout.
- Track accuracy, latency, and cost together — a winning prompt must pass all three.
- Refresh eval sets with production failures bi-weekly to prevent eval-production drift.
- Roll back immediately on production accuracy drops; investigate with eval afterward.

---

## Next Lesson

**[Lesson 8: Handling Edge Cases and Guardrails](08-lesson-08.md)** — Build robust prompts that handle adversarial inputs, edge cases, and failure modes gracefully.
