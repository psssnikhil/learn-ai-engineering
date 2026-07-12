---
title: Evaluation and Benchmarking Fine-Tuned Models
description: >-
  Learn systematic approaches to measure whether your fine-tuning actually
  improved the model using automated metrics, human evaluation, and A/B testing
duration: 40 min
difficulty: advanced
has_code: false
youtube: 'https://www.youtube.com/watch?v=IEVnfrFe2s0'
objectives:
  - Design an evaluation suite for fine-tuned models
  - 'Implement automated metrics (BLEU, ROUGE, exact match)'
  - Set up LLM-as-a-judge evaluation
  - Run A/B comparisons between models
---
# Evaluation and Benchmarking Fine-Tuned Models

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Design evaluation suites for fine-tuned models | 40 min | Advanced |
| Use automated metrics and LLM-as-a-judge | | |
| Detect overfitting, underfitting, and regressions | | |
| Run meaningful A/B comparisons | | |

---

## Why Evaluation Matters

Fine-tuning can make a model better at your task but worse at everything else. Without proper evaluation:
- You might deploy a model that overfits to training examples
- You might miss regressions on edge cases
- You can't justify the cost of fine-tuning vs alternatives

---

## Evaluation Strategy

### The Three-Layer Approach

```
Layer 1: Automated Metrics (fast, cheap, limited)
  - Loss curves, perplexity, exact match, BLEU/ROUGE
  
Layer 2: LLM-as-a-Judge (medium cost, good signal)
  - Use a stronger model to evaluate outputs
  
Layer 3: Human Evaluation (expensive, gold standard)
  - Domain experts rate outputs on specific criteria
```

---

## Layer 1: Automated Metrics

### Training and Validation Loss

```python
import json
import matplotlib.pyplot as plt

def plot_training_curves(log_file):
    """Plot training and validation loss from trainer logs."""
    with open(log_file) as f:
        logs = json.load(f)
    
    train_loss = [(l["step"], l["loss"]) for l in logs if "loss" in l]
    eval_loss = [(l["step"], l["eval_loss"]) for l in logs if "eval_loss" in l]
    
    plt.figure(figsize=(10, 6))
    plt.plot(*zip(*train_loss), label="Training Loss", alpha=0.7)
    plt.plot(*zip(*eval_loss), label="Validation Loss", marker="o")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Fine-Tuning Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("loss_curves.png")
    plt.show()
```

### What the Curves Tell You

| Pattern | Diagnosis | Action |
|---------|-----------|--------|
| Both losses decrease | Good training | Continue |
| Train loss low, val loss high | Overfitting | Reduce epochs, add data |
| Both losses high and flat | Underfitting | Increase rank, learning rate |
| Val loss spikes | Learning rate too high | Reduce learning rate |

### Task-Specific Metrics

```python
from collections import Counter

def evaluate_model(model, tokenizer, test_examples):
    """Run automated evaluation on test examples."""
    results = {
        "exact_match": 0,
        "format_correct": 0,
        "total": len(test_examples)
    }
    
    for example in test_examples:
        # Generate response
        messages = [{"role": "user", "content": example["input"]}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1)
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:], 
            skip_special_tokens=True
        )
        
        # Check exact match
        if response.strip() == example["expected_output"].strip():
            results["exact_match"] += 1
        
        # Check format compliance (customize for your task)
        if validate_format(response, example.get("expected_format")):
            results["format_correct"] += 1
    
    results["exact_match_rate"] = results["exact_match"] / results["total"]
    results["format_rate"] = results["format_correct"] / results["total"]
    
    return results

def validate_format(response, expected_format):
    """Check if response follows the expected format."""
    if not expected_format:
        return True
    # Example: check for required sections
    required_sections = expected_format.get("required_sections", [])
    return all(section.lower() in response.lower() for section in required_sections)
```

---

## Layer 2: LLM-as-a-Judge

Use a stronger model (e.g., GPT-4.1 or Claude) to evaluate your fine-tuned model's outputs.

```python
from openai import OpenAI

client = OpenAI()

def llm_judge(prompt, response_a, response_b, criteria):
    """Use GPT-4.1 to compare two model responses."""
    judge_prompt = f"""You are an expert evaluator. Compare two AI responses to the same prompt.

User Prompt: {prompt}

Response A:
{response_a}

Response B:
{response_b}

Evaluation Criteria:
{criteria}

Rate each response on a scale of 1-5 for each criterion. Then declare a winner.
Output as JSON with format:
{{
  "response_a_scores": {{"criterion1": score, ...}},
  "response_b_scores": {{"criterion1": score, ...}},
  "winner": "A" or "B" or "tie",
  "reasoning": "brief explanation"
}}"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": judge_prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)

# Example evaluation
result = llm_judge(
    prompt="Explain our refund policy",
    response_a="Base model response here...",
    response_b="Fine-tuned model response here...",
    criteria="1. Accuracy to company policy
2. Tone (professional, empathetic)
3. Completeness
4. Conciseness"
)

print(f"Winner: Response {result['winner']}")
print(f"Reasoning: {result['reasoning']}")
```

### Running a Full Evaluation Suite

```python
def run_evaluation_suite(base_model_fn, ft_model_fn, test_cases, criteria):
    """Compare base vs fine-tuned model across all test cases."""
    wins = {"base": 0, "fine_tuned": 0, "tie": 0}
    
    for test in test_cases:
        base_response = base_model_fn(test["prompt"])
        ft_response = ft_model_fn(test["prompt"])
        
        result = llm_judge(
            prompt=test["prompt"],
            response_a=base_response,
            response_b=ft_response,
            criteria=criteria
        )
        
        if result["winner"] == "A":
            wins["base"] += 1
        elif result["winner"] == "B":
            wins["fine_tuned"] += 1
        else:
            wins["tie"] += 1
    
    total = len(test_cases)
    print(f"
=== Evaluation Results ({total} test cases) ===")
    print(f"Fine-tuned wins: {wins['fine_tuned']} ({wins['fine_tuned']/total:.0%})")
    print(f"Base model wins:  {wins['base']} ({wins['base']/total:.0%})")
    print(f"Ties:             {wins['tie']} ({wins['tie']/total:.0%})")
    
    return wins
```

---

## Detecting Regressions

Fine-tuning can cause **catastrophic forgetting** — the model gets better at your task but worse at general capabilities.

```python
# Test general capabilities alongside your specific task
regression_tests = [
    {"prompt": "What is 15 * 23?", "expected": "345", "category": "math"},
    {"prompt": "Translate 'hello' to French", "expected": "bonjour", "category": "translation"},
    {"prompt": "Is this sentence grammatically correct: 'She don't like apples'", "category": "grammar"},
]

# Run these alongside your domain-specific tests
# If general capability scores drop significantly, you may be overfitting
```

---

## Resources

- **Blog: LLM Evaluation Best Practices** — Practical guide to evaluating language models
- **Hugging Face Evaluate Library**: [huggingface.co/docs/evaluate](https://huggingface.co/docs/evaluate)
- **LMSYS Chatbot Arena**: Real-world model comparison methodology
- **Paper: "Judging LLM-as-a-Judge"** — Understanding biases in LLM evaluation

---

## Key Takeaways

- Use a three-layer evaluation strategy: automated metrics, LLM-as-a-judge, and human review
- Always compare against the base model to prove fine-tuning added value
- Watch for overfitting (low training loss, high validation loss) and regressions on general tasks
- LLM-as-a-judge is cost-effective and correlates well with human preferences
- Hold out test examples that were never seen during training

---

## Next Lesson

**Lesson 7: RLHF and Preference Tuning** — Learn how Reinforcement Learning from Human Feedback aligns models with human preferences beyond simple supervised fine-tuning.
