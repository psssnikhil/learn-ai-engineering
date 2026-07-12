---
title: When and Why to Fine-Tune LLMs
description: >-
  Understand when fine-tuning is the right approach, how it compares to prompt
  engineering and RAG, and the cost-benefit tradeoffs
duration: 35 min
difficulty: advanced
has_code: false
---
# When and Why to Fine-Tune LLMs

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand what fine-tuning actually does to a model | 35 min | Advanced |
| Compare fine-tuning vs prompt engineering vs RAG | | |
| Identify when fine-tuning is worth the investment | | |
| Estimate costs and timelines for fine-tuning projects | | |

---

## What is Fine-Tuning?

**Fine-tuning** is the process of further training a pre-trained LLM on a smaller, domain-specific dataset to specialize its behavior for your use case.

```
Pre-trained Model (General Knowledge)
        |
        | + Your Training Data (100s-10,000s of examples)
        | + Training Compute ($10-$10,000+)
        v
Fine-Tuned Model (Specialized for Your Task)
```

### What Fine-Tuning Changes

Fine-tuning adjusts the model's weights to:
- **Learn your style**: Match specific tone, format, terminology
- **Improve on specific tasks**: Better accuracy on your domain
- **Reduce prompt length**: Bake instructions into the model itself
- **Handle edge cases**: Learn from your corrections

### What Fine-Tuning Does NOT Do

- It does NOT give the model new factual knowledge reliably (use RAG for that)
- It does NOT make a small model as capable as a large one
- It does NOT eliminate hallucinations entirely
- It does NOT work well with tiny datasets (< 50 examples)

---

## The Decision Framework: Fine-Tuning vs Alternatives

Before fine-tuning, consider cheaper and faster alternatives:

```
Start Here: Can prompt engineering solve it?
  |
  |-- Yes -> Use prompt engineering (cheapest, fastest)
  |
  +-- No: Does the model need access to specific data?
        |
        |-- Yes -> Use RAG (retrieve relevant context)
        |
        +-- No: Does the model need to behave in a consistently
              different way (style, format, domain expertise)?
                |
                |-- Yes -> Fine-tune
                |
                +-- Still no -> Evaluate if you need a different
                                base model entirely
```

### Comparison Table

| Approach | Cost | Time to Deploy | Best For |
|----------|------|----------------|----------|
| **Prompt Engineering** | Free (just tokens) | Minutes-hours | Format, style, simple tasks |
| **Few-Shot Prompting** | Low (extra tokens) | Minutes | Consistent output format |
| **RAG** | Medium (vector DB) | Days | Factual knowledge, documents |
| **Fine-Tuning** | High ($100-$10K+) | Days-weeks | Behavioral changes, style, domain |
| **Pre-Training** | Very high ($1M+) | Months | Only for model providers |

---

## When Fine-Tuning IS the Right Choice

### 1. Consistent Style or Format

```python
# Problem: You need the model to ALWAYS respond in your company's 
# specific format, and prompt engineering is unreliable

# Example: Medical report formatting
# Without fine-tuning: Model sometimes forgets sections, uses wrong terminology
# With fine-tuning: Consistent, domain-correct formatting every time

training_example = {
    "messages": [
        {"role": "system", "content": "You are a medical report assistant."},
        {"role": "user", "content": "Patient presents with chest pain..."},
        {"role": "assistant", "content": 
            "CLINICAL SUMMARY
"
            "Chief Complaint: Chest pain
"
            "Assessment: [your specific format]
"
            "Plan: [your specific format]
"
            "ICD-10: [relevant codes]"
        }
    ]
}
```

### 2. Reducing Prompt Size (and Cost)

If you have a long system prompt that you send with every request, fine-tuning can bake those instructions into the model:

```
Before fine-tuning:
  System prompt: 2,000 tokens (sent every request)
  x 1M requests/month = 2 BILLION extra input tokens
  Cost: ~$5,000/month just for the system prompt!

After fine-tuning:
  System prompt: 50 tokens (minimal)
  x 1M requests/month = 50M tokens
  Cost: ~$125/month
  
  Savings: ~$4,875/month (easily covers fine-tuning cost)
```

### 3. Domain Expertise

When the model needs to understand specialized terminology, reasoning patterns, or domain conventions that cannot be easily conveyed through prompts.

### 4. Latency Optimization

Fine-tuned smaller models can match larger models on specific tasks, with much faster inference:

```
GPT-4.1 (large):  ~2s response, $10/1M output tokens
Fine-tuned GPT-4.1-mini: ~0.3s response, $1.6/1M output tokens
                         (if fine-tuned well for your task)
```

---

## When Fine-Tuning is NOT the Right Choice

| Situation | Better Alternative |
|-----------|-------------------|
| Model needs to know about your documents | RAG |
| Model needs to follow a specific format | Few-shot prompting first |
| You have fewer than 50 training examples | Prompt engineering |
| You need the model to know current events | RAG + web search |
| You want to try something quickly | Prompt engineering |
| Your task changes frequently | Prompt engineering (more flexible) |

---

## Cost and Timeline Estimates

### OpenAI Fine-Tuning (2025 pricing, approximate)

| Model | Training Cost | Inference Cost | Typical Dataset |
|-------|---------------|----------------|-----------------|
| GPT-4.1-mini | ~$3 per 1M training tokens | 2x base price | 500-5,000 examples |
| GPT-4.1 | ~$25 per 1M training tokens | 2x base price | 500-5,000 examples |

### Timeline

```
Week 1: Data collection and preparation
Week 2: Initial fine-tuning experiments
Week 3: Evaluation and iteration
Week 4: Production deployment

Total: 2-4 weeks for a well-scoped project
```

### Total Budget (Typical)

- **Small project** (500 examples, GPT-4.1-mini): $50-200
- **Medium project** (2,000 examples, GPT-4.1): $500-2,000
- **Large project** (10,000+ examples, multiple iterations): $2,000-10,000+

---

## Key Takeaways

- Fine-tuning specializes a model's behavior, not its knowledge
- Always try prompt engineering and RAG first -- they are cheaper and faster
- Fine-tune when you need consistent style, reduced prompt costs, or domain expertise
- Budget 2-4 weeks and $50-$10,000 depending on scope
- Fine-tuned smaller models can be faster and cheaper than large general models

---

## Next Lesson

**Lesson 2: Preparing Training Data** - Learn how to collect, format, and validate training data for fine-tuning.
