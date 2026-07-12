---
title: Few-Shot and Chain-of-Thought Prompting
description: >-
  Master few-shot examples and chain-of-thought reasoning to dramatically
  improve LLM output quality
duration: 45 min
difficulty: intermediate
has_code: false
---
# Few-Shot and Chain-of-Thought Prompting

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand zero-shot vs few-shot prompting | 45 min | Intermediate |
| Master chain-of-thought (CoT) reasoning | | |
| Learn when to use each technique | | |
| Combine techniques for complex tasks | | |

---

## Zero-Shot vs Few-Shot Prompting

### Zero-Shot: No Examples

You give the model a task with no examples. Works well for simple, well-defined tasks.

```python
# Zero-shot classification
prompt = "Classify this customer review as Positive, Negative, or Neutral:

"
prompt += "Review: 'The product arrived on time but the packaging was damaged.'
"
prompt += "Classification:"
# Output: "Neutral"
```

### Few-Shot: Learning from Examples

You provide examples of input-output pairs. The model learns the pattern and applies it.

```python
# Few-shot classification - much more reliable
prompt = """Classify each customer review as Positive, Negative, or Neutral.

Review: "Absolutely love this product! Best purchase ever."
Classification: Positive

Review: "Terrible quality. Broke after one day. Want a refund."
Classification: Negative

Review: "It's okay. Does what it says, nothing special."
Classification: Neutral

Review: "The product arrived on time but the packaging was damaged."
Classification:"""
# Output: "Neutral" (more reliable because the model sees the pattern)
```

### How Many Examples?

| Examples | Name | Best For |
|----------|------|----------|
| 0 | Zero-shot | Simple tasks, clear instructions |
| 1-3 | Few-shot | Most tasks, establishing format |
| 5-10 | Many-shot | Complex classification, nuanced output |
| 10+ | Many-shot | Specialized domains, precise formatting |

**Rule of thumb**: Start with 3 diverse examples that cover edge cases.

---

## Few-Shot Best Practices

### 1. Choose Diverse Examples

```python
# BAD: All examples are similar
examples = [
    ("Great product!", "Positive"),
    ("Love it!", "Positive"),
    ("Amazing!", "Positive"),
    # Model only sees positive - biased!
]

# GOOD: Cover the full range
examples = [
    ("Great product! Exceeded expectations.", "Positive"),
    ("Broken on arrival. Worst purchase.", "Negative"),
    ("It works. Nothing remarkable.", "Neutral"),
    # Model sees all categories - balanced!
]
```

### 2. Match Your Target Distribution

If 60% of real inputs are negative, your examples should roughly reflect that.

### 3. Include Edge Cases

```python
# Include tricky/ambiguous examples
examples = [
    # Clear cases
    ("Best thing I ever bought!", "Positive"),
    ("Total waste of money.", "Negative"),
    
    # Edge cases - teach the model your definitions
    ("Good product but overpriced.", "Neutral"),  # Mixed -> Neutral
    ("Not bad at all, surprisingly.", "Positive"),  # Double negative -> Positive
    ("I guess it's fine...", "Neutral"),  # Lukewarm -> Neutral
]
```

---

## Chain-of-Thought (CoT) Prompting

Chain-of-thought prompting asks the model to show its reasoning step by step. This dramatically improves performance on tasks requiring logic, math, or multi-step reasoning.

### Without CoT

```python
prompt = """
Q: A store has 15 red shirts and 8 blue shirts. They sell 
3/5 of the red shirts and 1/4 of the blue shirts. How many 
shirts remain in total?

A:"""
# Model might jump to wrong answer
```

### With CoT

```python
prompt = """
Q: A store has 15 red shirts and 8 blue shirts. They sell 
3/5 of the red shirts and 1/4 of the blue shirts. How many 
shirts remain in total?

A: Let me work through this step by step.

Step 1: Calculate red shirts sold
- Started with 15 red shirts
- Sold 3/5 of them: 15 x 3/5 = 9 red shirts sold
- Red shirts remaining: 15 - 9 = 6

Step 2: Calculate blue shirts sold
- Started with 8 blue shirts
- Sold 1/4 of them: 8 x 1/4 = 2 blue shirts sold
- Blue shirts remaining: 8 - 2 = 6

Step 3: Total remaining
- 6 red + 6 blue = 12 shirts remaining

The answer is 12 shirts."""
```

### The Magic Phrase

Simply adding **"Let's think step by step"** to your prompt can significantly improve reasoning:

```python
# Zero-shot CoT - surprisingly effective!
prompt = """
Q: If it takes 5 machines 5 minutes to make 5 widgets, 
how long would it take 100 machines to make 100 widgets?

Let's think step by step.
"""
# Model reasons through it correctly: 5 minutes
# Without CoT, many models answer "100 minutes" (wrong!)
```

---

## Combining Few-Shot + CoT

The most powerful technique: show examples WITH reasoning chains.

```python
prompt = """You are a customer support classifier. Classify tickets 
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
# Model will follow the same structured reasoning pattern
```

---

## When to Use Each Technique

```
Is the task simple and well-defined?
  |-- Yes -> Zero-shot (just clear instructions)
  +-- No
        |-- Does it need specific output format?
        |     +-- Yes -> Few-shot (show format examples)
        +-- Does it require reasoning/logic?
              |-- Yes -> Chain-of-thought
              +-- Both format + reasoning?
                    +-- Few-shot + CoT (most powerful)
```

---

## Practice Exercise

Try these progressively harder prompts:

```python
# 1. Zero-shot: Simple extraction
prompt_1 = "Extract all email addresses from this text: ..."

# 2. Few-shot: Consistent formatting
prompt_2 = """Convert these descriptions to JSON:

Input: "John, 28 years old, lives in NYC, works as an engineer"
Output: {"name": "John", "age": 28, "city": "NYC", "job": "engineer"}

Input: "Sarah is a 35-year-old teacher from Chicago"
Output: {"name": "Sarah", "age": 35, "city": "Chicago", "job": "teacher"}

Input: "Mike, designer, age 42, based in LA"
Output:"""

# 3. CoT: Complex reasoning
prompt_3 = """A company's revenue grew 20% in Q1, dropped 10% in Q2, 
and grew 15% in Q3. If they started at $1M, what's the Q3 revenue?

Let's think step by step."""

# 4. Few-shot + CoT: Combine both for complex classification
# (Try writing this one yourself!)
```

---

## Key Takeaways

- Few-shot examples teach the model your expected pattern and format
- Chain-of-thought prompting dramatically improves reasoning tasks
- "Let's think step by step" is a simple but powerful addition
- Combine few-shot + CoT for the best results on complex tasks
- Choose your technique based on task complexity and type

---

## Next Lesson

**Lesson 3: Advanced Prompt Patterns** - Learn structured output, self-consistency, and prompt chaining for production applications.
