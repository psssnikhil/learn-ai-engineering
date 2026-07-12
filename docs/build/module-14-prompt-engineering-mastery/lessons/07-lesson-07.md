---
title: Prompt Optimization and Iteration
description: >-
  Learn systematic methods to improve prompt performance through evaluation, A/B
  testing, and iterative refinement
duration: 35 min
difficulty: intermediate
has_code: false
module: module-14
---
# Prompt Optimization and Iteration

## Learning Objectives

By the end of this lesson, you will be able to:
- Build an evaluation dataset to measure prompt quality
- Use systematic A/B testing to compare prompt variants
- Apply common optimization techniques that improve most prompts
- Track prompt performance over time

---

## The Prompt Optimization Loop

```
Write prompt → Test on eval set → Measure metrics → Identify failures → Refine → Repeat
```

Most teams iterate 5-15 times before a prompt reaches production quality.

---

## Building an Evaluation Set

```python
# An eval set is a list of (input, expected_output) pairs
eval_set = [
    {
        "input": "I absolutely love this product! Best purchase ever.",
        "expected": "positive",
    },
    {
        "input": "Terrible experience. Broke after one day.",
        "expected": "negative",
    },
    {
        "input": "It works fine. Nothing special but does the job.",
        "expected": "neutral",
    },
    # ... at least 20-50 examples for reliable results
]

def evaluate_prompt(prompt_template: str, eval_set: list[dict]) -> dict:
    """Run a prompt against an evaluation set and measure accuracy."""
    correct = 0
    results = []

    for item in eval_set:
        prompt = prompt_template.format(text=item["input"])
        response = llm_call(prompt)
        predicted = response.strip().lower()
        is_correct = predicted == item["expected"]
        correct += int(is_correct)
        results.append({
            "input": item["input"][:50],
            "expected": item["expected"],
            "predicted": predicted,
            "correct": is_correct,
        })

    accuracy = correct / len(eval_set)
    return {"accuracy": accuracy, "results": results}
```

---

## A/B Testing Prompts

```python
prompt_v1 = "Classify the sentiment as positive, negative, or neutral:

{text}"

prompt_v2 = """Classify the sentiment of the following text.

Rules:
- "positive": The text expresses satisfaction, happiness, or enthusiasm
- "negative": The text expresses dissatisfaction, frustration, or disappointment  
- "neutral": The text is factual, balanced, or neither positive nor negative

Text: {text}

Sentiment (respond with exactly one word):"""

# Compare
results_v1 = evaluate_prompt(prompt_v1, eval_set)
results_v2 = evaluate_prompt(prompt_v2, eval_set)

print(f"v1 accuracy: {results_v1['accuracy']:.1%}")  # e.g., 72%
print(f"v2 accuracy: {results_v2['accuracy']:.1%}")  # e.g., 91%
```

---

## Common Optimization Techniques

| Technique | When to Use | Typical Improvement |
|-----------|-------------|-------------------|
| **Add explicit rules** | Output is inconsistent | +10-20% accuracy |
| **Add examples (few-shot)** | Model misunderstands the task | +15-25% accuracy |
| **Define output format** | Responses vary in structure | Consistency, easier parsing |
| **Add chain-of-thought** | Reasoning tasks, math | +10-30% on complex tasks |
| **Reduce ambiguity** | Edge cases fail | +5-15% accuracy |
| **Add constraints** | Output is too long/short | Controls output quality |

### Example: Progressive Refinement

```python
# Iteration 1: Basic (72% accuracy)
v1 = "Classify sentiment: {text}"

# Iteration 2: Add definitions (85% accuracy)
v2 = """Classify sentiment as positive, negative, or neutral.
positive = expresses satisfaction or enthusiasm
negative = expresses dissatisfaction or frustration
neutral = factual or balanced

Text: {text}
Sentiment:"""

# Iteration 3: Add examples (91% accuracy)
v3 = """Classify sentiment as positive, negative, or neutral.

Examples:
"Best purchase ever!" → positive
"Broke after one day" → negative
"It does the job" → neutral

Text: {text}
Sentiment:"""

# Iteration 4: Handle edge cases (94% accuracy)
v4 = """Classify sentiment as positive, negative, or neutral.

Rules:
- Sarcasm should be classified by actual intent ("Great, another broken product" = negative)
- Mixed sentiment defaults to the dominant emotion
- Questions without sentiment are neutral

Examples:
"Best purchase ever!" → positive
"Broke after one day" → negative
"It does the job" → neutral

Text: {text}
Sentiment (one word):"""
```

---

## Analyzing Failures

```python
def analyze_failures(eval_results: dict) -> None:
    """Print failed cases to identify patterns."""
    failures = [r for r in eval_results["results"] if not r["correct"]]

    print(f"
{len(failures)} failures out of {len(eval_results['results'])} cases:
")
    for f in failures:
        print(f"  Input:    {f['input']}")
        print(f"  Expected: {f['expected']}")
        print(f"  Got:      {f['predicted']}")
        print()

# Common failure patterns to look for:
# - Sarcasm misclassified
# - Mixed sentiment inconsistent
# - Edge cases (empty text, single words)
# - Format issues (extra words in output)
```

---

## Key Takeaways

- Build an evaluation set before optimizing -- you cannot improve what you do not measure
- A/B test prompt variants systematically, not by gut feeling
- Most prompts improve significantly by adding explicit rules, examples, and output constraints
- Analyze failure cases to find patterns and target your next iteration
- Expect 5-15 iterations to reach production quality

## Resources

- [YouTube: Evaluating LLM Prompts](https://www.youtube.com/watch?v=r-HUnht-Gns) -- Systematic evaluation techniques
- [OpenAI: Prompt Engineering Best Practices](https://platform.openai.com/docs/guides/prompt-engineering) -- Official optimization guide
- [PromptFoo](https://www.promptfoo.dev/) -- Open-source tool for prompt testing and evaluation

---

Next: Handling Edge Cases and Guardrails
