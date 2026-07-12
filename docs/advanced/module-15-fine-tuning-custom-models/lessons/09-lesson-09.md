---
title: Model Distillation and Compression
description: >-
  Learn to transfer knowledge from large teacher models to small, fast student
  models for cost-effective production deployment
duration: 40 min
difficulty: advanced
has_code: false
module: module-15
youtube: 'https://www.youtube.com/watch?v=SYB1FaFcxbE'
objectives:
  - Explain the knowledge distillation process
  - Generate distillation training data from a teacher model
  - Fine-tune a small student model on distilled data
  - Measure quality vs cost tradeoffs
---
# Model Distillation and Compression

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand knowledge distillation for LLMs | 40 min | Advanced |
| Generate high-quality distillation data | | |
| Train a smaller model to mimic a larger one | | |
| Evaluate cost vs quality tradeoffs | | |

---

## What is Knowledge Distillation?

Knowledge distillation transfers the capabilities of a large **teacher** model into a smaller **student** model. The student learns to mimic the teacher's behavior on your specific tasks.

```
Teacher Model (e.g., GPT-4.1, Claude Opus)
  - Expensive: $10-15/1M output tokens
  - Slow: 1-3 seconds per response
  - High quality on everything
        |
        | Generate training examples
        v
Student Model (e.g., Llama-3.1-8B, GPT-4.1-mini fine-tuned)
  - Cheap: $0.50-1.60/1M output tokens  
  - Fast: 0.1-0.5 seconds per response
  - High quality on YOUR specific tasks
```

---

## When to Distill

| Scenario | Distillation Value |
|----------|-------------------|
| High-volume API calls (>100K/day) | Very high — saves 80-90% on costs |
| Latency-sensitive applications | High — smaller models respond faster |
| Need to self-host (privacy) | High — can run on single GPU |
| Diverse, unpredictable tasks | Low — student won't generalize as well |
| Low volume (<1K requests/day) | Low — just use the teacher directly |

---

## Step 1: Generate Distillation Data

The key is to generate a diverse, high-quality dataset using the teacher model:

```python
from openai import OpenAI
import json
import random

client = OpenAI()

def generate_distillation_data(
    prompts, 
    teacher_model="gpt-4.1",
    system_prompt="",
    samples_per_prompt=1
):
    """Generate training data using a teacher model."""
    training_data = []
    
    for prompt in prompts:
        for _ in range(samples_per_prompt):
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model=teacher_model,
                messages=messages,
                temperature=0.7,  # Some variety in responses
            )
            
            teacher_response = response.choices[0].message.content
            
            training_data.append({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": teacher_response}
                ]
            })
    
    return training_data

# Example: distill customer service capabilities
prompts = [
    "What is your return policy?",
    "My order hasn't arrived yet",
    "I want to cancel my subscription",
    "How do I change my shipping address?",
    "The product I received is damaged",
    # ... hundreds more prompts covering your use cases
]

system_prompt = "You are a helpful customer service agent. Be concise, empathetic, and solution-oriented."

data = generate_distillation_data(prompts, system_prompt=system_prompt)

# Save to JSONL
with open("distillation_data.jsonl", "w") as f:
    for example in data:
        f.write(json.dumps(example) + "
")

print(f"Generated {len(data)} distillation examples")
```

### Generating Diverse Prompts

```python
def generate_diverse_prompts(task_description, num_prompts=500):
    """Use the teacher to generate diverse training prompts."""
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{
            "role": "user", 
            "content": f"""Generate {num_prompts} diverse user prompts for this task:
{task_description}

Requirements:
- Cover edge cases and unusual scenarios
- Vary in length, complexity, and tone
- Include some ambiguous or challenging requests
- Mix simple and complex queries

Return as a JSON array of strings."""
        }],
        temperature=0.9,
    )
    
    return json.loads(response.choices[0].message.content)
```

---

## Step 2: Quality Filtering

Not all teacher responses are good. Filter before training:

```python
def filter_quality(data, teacher_model="gpt-4.1"):
    """Use the teacher to quality-check its own responses."""
    filtered = []
    
    for example in data:
        user_msg = next(m for m in example["messages"] if m["role"] == "user")
        asst_msg = next(m for m in example["messages"] if m["role"] == "assistant")
        
        check = client.chat.completions.create(
            model=teacher_model,
            messages=[{
                "role": "user",
                "content": f"""Rate this response quality (1-5):
                
Question: {user_msg["content"]}
Response: {asst_msg["content"]}

Return JSON: {{"score": <1-5>, "reason": "..."}}"""
            }],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(check.choices[0].message.content)
        if result["score"] >= 4:
            filtered.append(example)
    
    print(f"Kept {len(filtered)}/{len(data)} examples (filtered {len(data)-len(filtered)} low-quality)")
    return filtered
```

---

## Step 3: Train the Student

Use the distillation data to fine-tune a smaller model (exactly like Lessons 3-4):

```python
# Option A: Fine-tune GPT-4.1-mini via OpenAI API
job = client.fine_tuning.jobs.create(
    training_file=upload_file("distillation_data.jsonl"),
    model="gpt-4.1-mini",
    hyperparameters={"n_epochs": 3},
    suffix="distilled-cs-agent"
)

# Option B: Fine-tune an open-source model locally
# (See Lesson 4 for the full Hugging Face workflow)
```

---

## Step 4: Evaluate the Student

```python
def evaluate_distillation(test_prompts, system_prompt, student_model, teacher_model="gpt-4.1"):
    """Compare student vs teacher on held-out test prompts."""
    results = {"student_wins": 0, "teacher_wins": 0, "ties": 0}
    
    for prompt in test_prompts:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        teacher_resp = client.chat.completions.create(
            model=teacher_model, messages=messages, temperature=0.3
        ).choices[0].message.content
        
        student_resp = client.chat.completions.create(
            model=student_model, messages=messages, temperature=0.3
        ).choices[0].message.content
        
        # Use teacher to judge (with position randomization to avoid bias)
        if random.random() > 0.5:
            a, b, a_label, b_label = teacher_resp, student_resp, "teacher", "student"
        else:
            a, b, a_label, b_label = student_resp, teacher_resp, "student", "teacher"
        
        judge = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": f"""Compare these two responses:

Prompt: {prompt}

Response A: {a}
Response B: {b}

Which is better? Return JSON: {{"winner": "A" or "B" or "tie"}}"""}],
            temperature=0,
            response_format={"type": "json_object"}
        )
        
        winner = json.loads(judge.choices[0].message.content)["winner"]
        if winner == "A":
            results[f"{a_label}_wins"] += 1
        elif winner == "B":
            results[f"{b_label}_wins"] += 1
        else:
            results["ties"] += 1
    
    total = len(test_prompts)
    print(f"Student wins: {results['student_wins']}/{total}")
    print(f"Teacher wins: {results['teacher_wins']}/{total}")
    print(f"Ties: {results['ties']}/{total}")
    return results
```

---

## Cost Analysis

```python
def calculate_roi(
    daily_requests,
    avg_input_tokens=200,
    avg_output_tokens=300,
    teacher_input_price=2.50,   # per 1M tokens
    teacher_output_price=10.00,
    student_input_price=0.40,
    student_output_price=1.60,
    distillation_cost=200,      # one-time cost to generate data + train
):
    """Calculate ROI of distillation."""
    daily_input = daily_requests * avg_input_tokens / 1_000_000
    daily_output = daily_requests * avg_output_tokens / 1_000_000
    
    teacher_daily = (daily_input * teacher_input_price) + (daily_output * teacher_output_price)
    student_daily = (daily_input * student_input_price) + (daily_output * student_output_price)
    
    savings_per_day = teacher_daily - student_daily
    breakeven_days = distillation_cost / savings_per_day if savings_per_day > 0 else float('inf')
    
    print(f"Daily cost (teacher): ${teacher_daily:.2f}")
    print(f"Daily cost (student): ${student_daily:.2f}")
    print(f"Daily savings: ${savings_per_day:.2f}")
    print(f"Breakeven: {breakeven_days:.0f} days")
    print(f"Annual savings: ${savings_per_day * 365:,.0f}")

# Example: 50K requests/day
calculate_roi(daily_requests=50_000)
```

---

## Resources

- **OpenAI Model Distillation Guide**: [platform.openai.com/docs/guides/distillation](https://platform.openai.com/docs/guides/distillation)
- **Paper: "Distilling the Knowledge in a Neural Network"** (Hinton et al., 2015)
- **Blog: How to Distill GPT-4 into a Smaller Model**: Practical guide with code examples

---

## Key Takeaways

- Knowledge distillation trains a small, cheap model to mimic a large, expensive one on your specific tasks
- Generate 500-5,000 diverse examples using the teacher model as training data
- Quality-filter the teacher's outputs before training
- A well-distilled student can match 80-95% of teacher quality at 10-20% of the cost
- The ROI is strongest for high-volume, narrow-domain applications

---

## Next Lesson

**Lesson 10: Fine-Tuning Best Practices and Case Studies** — Real-world lessons from production fine-tuning projects, common pitfalls, and a decision framework.
