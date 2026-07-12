---
title: Preparing Training Data for Fine-Tuning
description: >-
  Learn how to collect, format, validate, and quality-check training data for
  LLM fine-tuning
duration: 40 min
difficulty: advanced
has_code: false
module: module-15
youtube: 'https://www.youtube.com/watch?v=pW8B4SCkr0k'
objectives:
  - Format training examples in the OpenAI conversational format
  - Implement data validation and quality checks
  - Handle edge cases in training data preparation
  - Calculate token counts and estimate training costs
---
# Preparing Training Data for Fine-Tuning

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand training data formats for fine-tuning | 40 min | Advanced |
| Build a data preparation pipeline | | |
| Validate and quality-check your dataset | | |
| Estimate token usage and costs | | |

---

## Training Data Formats

Most LLM fine-tuning uses the **conversational format** — a list of messages with roles. This is the format used by OpenAI, Anthropic, and most open-source fine-tuning tools.

### OpenAI Format (JSONL)

Each line in your training file is a JSON object with a `messages` array:

```json
{"messages": [{"role": "system", "content": "You are a helpful coding assistant."}, {"role": "user", "content": "Write a Python function to reverse a string"}, {"role": "assistant", "content": "Here's a Python function to reverse a string:

```python
def reverse_string(s: str) -> str:
    return s[::-1]
```

This uses Python's slice notation with a step of -1 to reverse the string."}]}
```

### Key Rules

1. **Each example must have at least 2 messages** (user + assistant)
2. **System messages are optional** but recommended for consistent behavior
3. **The last message must be from the assistant** — this is what the model learns to generate
4. **JSONL format**: one JSON object per line, no trailing commas

---

## Building a Data Preparation Pipeline

### Step 1: Collect Raw Data

Sources for training data:
- **Existing conversations**: Chat logs, support tickets, email threads
- **Expert examples**: Have domain experts write ideal responses
- **Synthetic data**: Use a stronger model to generate examples
- **Curated datasets**: Filter public datasets for quality

### Step 2: Format and Clean

```python
import json
import tiktoken

def format_training_example(system_prompt, user_message, assistant_response):
    """Format a single training example in OpenAI conversational format."""
    example = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message.strip()},
            {"role": "assistant", "content": assistant_response.strip()}
        ]
    }
    return example

def prepare_dataset(raw_examples, system_prompt):
    """Convert raw examples to training format."""
    formatted = []
    for ex in raw_examples:
        formatted_ex = format_training_example(
            system_prompt=system_prompt,
            user_message=ex["input"],
            assistant_response=ex["output"]
        )
        formatted.append(formatted_ex)
    return formatted

# Example usage
raw_data = [
    {
        "input": "What is the return policy?",
        "output": "Our return policy allows returns within 30 days of purchase. Items must be in original condition with tags attached. Refunds are processed within 5-7 business days."
    },
    {
        "input": "How do I track my order?",
        "output": "You can track your order by logging into your account and visiting the 'Orders' section. You will also receive tracking updates via email once your order ships."
    }
]

system_prompt = "You are a helpful customer service agent for an e-commerce company. Respond clearly and concisely."

dataset = prepare_dataset(raw_data, system_prompt)

# Write to JSONL file
with open("training_data.jsonl", "w") as f:
    for example in dataset:
        f.write(json.dumps(example) + "
")

print(f"Prepared {len(dataset)} training examples")
```

---

## Data Validation

Before training, validate every example to avoid wasted compute and poor results.

```python
import json
import tiktoken

def validate_training_file(filepath):
    """Validate a JSONL training file and report issues."""
    encoding = tiktoken.get_encoding("cl100k_base")
    
    errors = []
    warnings = []
    total_tokens = 0
    example_count = 0
    
    with open(filepath, "r") as f:
        for line_num, line in enumerate(f, 1):
            example_count += 1
            
            # Check valid JSON
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {e}")
                continue
            
            # Check required structure
            if "messages" not in data:
                errors.append(f"Line {line_num}: Missing 'messages' key")
                continue
            
            messages = data["messages"]
            
            if len(messages) < 2:
                errors.append(f"Line {line_num}: Need at least 2 messages (user + assistant)")
                continue
            
            # Check last message is from assistant
            if messages[-1]["role"] != "assistant":
                errors.append(f"Line {line_num}: Last message must be from 'assistant'")
            
            # Check valid roles
            valid_roles = {"system", "user", "assistant"}
            for msg in messages:
                if msg.get("role") not in valid_roles:
                    errors.append(f"Line {line_num}: Invalid role '{msg.get('role')}'")
                if not msg.get("content", "").strip():
                    warnings.append(f"Line {line_num}: Empty content for role '{msg.get('role')}'")
            
            # Count tokens
            example_tokens = sum(
                len(encoding.encode(msg["content"])) 
                for msg in messages
            )
            total_tokens += example_tokens
            
            # Warn on very short or very long examples
            if example_tokens < 20:
                warnings.append(f"Line {line_num}: Very short example ({example_tokens} tokens)")
            elif example_tokens > 4096:
                warnings.append(f"Line {line_num}: Very long example ({example_tokens} tokens)")
    
    # Print report
    print(f"=== Validation Report ===")
    print(f"Examples: {example_count}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Avg tokens/example: {total_tokens // max(example_count, 1)}")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    
    for error in errors:
        print(f"  ERROR: {error}")
    for warning in warnings[:10]:
        print(f"  WARNING: {warning}")
    
    return len(errors) == 0

# Run validation
is_valid = validate_training_file("training_data.jsonl")
```

---

## Data Quality Guidelines

### Quantity Recommendations

| Use Case | Minimum Examples | Recommended | Sweet Spot |
|----------|-----------------|-------------|------------|
| Style/format changes | 50 | 200-500 | 300 |
| Domain adaptation | 200 | 500-2,000 | 1,000 |
| Complex reasoning | 500 | 2,000-5,000 | 3,000 |
| Multi-task fine-tuning | 1,000 | 5,000-10,000 | 5,000 |

### Quality Checklist

- **Consistency**: All examples should follow the same format and style
- **Diversity**: Cover the range of inputs the model will see in production
- **Accuracy**: Every assistant response must be correct and high-quality
- **No contradictions**: Examples should not give conflicting instructions
- **Balanced**: Avoid overrepresenting any single pattern or topic

### Common Mistakes

1. **Too few examples**: Under 50 rarely works; 200+ is much more reliable
2. **Low-quality responses**: The model learns to mimic your examples — garbage in, garbage out
3. **Inconsistent formatting**: If your examples use different formats, the model will randomly switch between them
4. **Missing edge cases**: Include examples of how to handle unusual or difficult inputs
5. **Train/test contamination**: Always hold out 10-20% of examples for evaluation

---

## Generating Synthetic Training Data

When you don't have enough real examples, use a stronger model to generate them:

```python
from openai import OpenAI

client = OpenAI()

def generate_synthetic_examples(task_description, num_examples=50):
    """Use GPT-4 to generate training examples for fine-tuning a smaller model."""
    prompt = f"""Generate {num_examples} diverse training examples for the following task:

Task: {task_description}

For each example, provide:
1. A realistic user message
2. An ideal assistant response

Format as JSON array with "input" and "output" fields.
Make examples diverse in topic, length, and complexity.
Include some edge cases and difficult scenarios."""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8
    )
    
    return json.loads(response.choices[0].message.content)

# Generate examples
task = "Customer service agent for a SaaS product. Should be helpful, concise, and escalate complex billing issues to human agents."
synthetic_data = generate_synthetic_examples(task, num_examples=20)
```

---

## Estimating Costs

```python
def estimate_fine_tuning_cost(filepath, model="gpt-4.1-mini", epochs=3):
    """Estimate fine-tuning cost based on token count."""
    encoding = tiktoken.get_encoding("cl100k_base")
    
    total_tokens = 0
    with open(filepath, "r") as f:
        for line in f:
            data = json.loads(line)
            for msg in data["messages"]:
                total_tokens += len(encoding.encode(msg["content"]))
    
    # Approximate pricing (check OpenAI's current pricing page)
    training_cost_per_1m = {"gpt-4.1-mini": 3.00, "gpt-4.1": 25.00}
    cost_per_1m = training_cost_per_1m.get(model, 3.00)
    
    training_tokens = total_tokens * epochs
    estimated_cost = (training_tokens / 1_000_000) * cost_per_1m
    
    print(f"Dataset tokens: {total_tokens:,}")
    print(f"Training tokens ({epochs} epochs): {training_tokens:,}")
    print(f"Estimated cost ({model}): ${estimated_cost:.2f}")
    
    return estimated_cost
```

---

## Resources

- **OpenAI Fine-Tuning Data Guide**: [platform.openai.com/docs/guides/fine-tuning](https://platform.openai.com/docs/guides/fine-tuning)
- **Hugging Face Datasets Library**: For working with large datasets efficiently
- **Argilla**: Open-source tool for labeling and curating training data

---

## Key Takeaways

- Training data must be in JSONL conversational format with user/assistant message pairs
- Always validate your data before training — check format, quality, and token counts
- Aim for 200-500 high-quality examples as a starting point
- Use synthetic data generation to bootstrap when real examples are limited
- Hold out 10-20% of examples for evaluation — never train on your test set

---

## Next Lesson

**Lesson 3: Fine-Tuning with the OpenAI API** — Walk through your first fine-tuning job step by step using the OpenAI platform.
