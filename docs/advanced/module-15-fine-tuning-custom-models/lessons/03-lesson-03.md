---
title: Fine-Tuning with the OpenAI API
description: >-
  Walk through your first fine-tuning job step by step using the OpenAI platform
  — upload data, launch training, and test your custom model
duration: 45 min
difficulty: advanced
has_code: false
module: module-15
youtube: 'https://www.youtube.com/watch?v=NRVaRXDoI88'
objectives:
  - Upload a training file to OpenAI
  - Create and monitor a fine-tuning job
  - Test your fine-tuned model with completions
  - Compare fine-tuned vs base model outputs
---
# Fine-Tuning with the OpenAI API

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Upload training data to OpenAI | 45 min | Advanced |
| Launch and monitor a fine-tuning job | | |
| Use your fine-tuned model for inference | | |
| Evaluate and iterate on results | | |

---

## Prerequisites

Before starting, you need:

| Requirement | Details |
|-------------|---------|
| **OpenAI API key** | With fine-tuning permissions enabled |
| **Training data** | Validated JSONL file from Lesson 2 (50+ examples minimum) |
| **Python 3.10+** | With `openai` package installed |
| **Budget awareness** | Fine-tuning costs vary by model; check current pricing |

```bash
pip install openai python-dotenv
```

Create a `.env` file:

```bash
OPENAI_API_KEY=sk-your-key-here
```

**Courses required:**
- Module 15, Lesson 1: When to Fine-Tune vs RAG vs Prompt Engineering
- Module 15, Lesson 2: Preparing Training Data (JSONL format)

---

## What You'll Build

By the end of this lesson, you will have:

- [ ] A training file uploaded to OpenAI's platform
- [ ] A completed fine-tuning job with monitored metrics
- [ ] A fine-tuned model ID ready for inference
- [ ] A side-by-side comparison script (base vs fine-tuned)
- [ ] An evaluation report with at least 10 test prompts

**Example use case:** Fine-tune `gpt-4.1-mini` on customer support conversations so it responds in your company's tone and policy.

---

## Architecture

```
[Training Data]                    [Validation Data]
training_data.jsonl                validation_data.jsonl
       |                                    |
       v                                    v
[OpenAI Files API]  ──────────────>  File IDs stored
       |
       v
[Fine-Tuning Job]
  - Base model: gpt-4.1-mini
  - Hyperparameters: n_epochs, batch_size, lr
  - Suffix: custom model name
       |
       v
[Training Loop]  (10-60 minutes)
  - Training loss decreases
  - Validation loss tracked
  - Events logged
       |
       v
[Fine-Tuned Model]
  ft:gpt-4.1-mini:org:suffix:id
       |
       v
[Inference API]
  - Same chat.completions interface
  - Drop-in replacement for base model
       |
       v
[Evaluation]
  - Compare base vs fine-tuned
  - Measure quality improvement
```

---

## Step 1: Validate and Upload Your Training File

Before uploading, confirm your JSONL format is correct. Each line must be a valid chat conversation:

```python
# validate_data.py
import json

def validate_jsonl(filepath: str) -> dict:
    """Validate training data format before upload."""
    errors = []
    examples = []
    with open(filepath, "r") as f:
        for i, line in enumerate(f, 1):
            try:
                example = json.loads(line.strip())
                if "messages" not in example:
                    errors.append(f"Line {i}: missing 'messages' key")
                    continue
                roles = [m["role"] for m in example["messages"]]
                if roles[-1] != "assistant":
                    errors.append(f"Line {i}: last message must be from assistant")
                examples.append(example)
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: invalid JSON - {e}")
    return {
        "valid": len(errors) == 0,
        "total_examples": len(examples),
        "errors": errors,
    }

result = validate_jsonl("training_data.jsonl")
print(f"Valid: {result['valid']}, Examples: {result['total_examples']}")
if result["errors"]:
    for err in result["errors"][:5]:
        print(f"  ERROR: {err}")
```

Upload once validation passes:

```python
from openai import OpenAI
import time

client = OpenAI()

with open("training_data.jsonl", "rb") as f:
    training_file = client.files.create(
        file=f,
        purpose="fine-tune",
    )

print(f"File uploaded: {training_file.id}")
print(f"Status: {training_file.status}")
print(f"Filename: {training_file.filename}")
print(f"Bytes: {training_file.bytes}")
```

Upload a validation file (strongly recommended):

```python
with open("validation_data.jsonl", "rb") as f:
    validation_file = client.files.create(
        file=f,
        purpose="fine-tune",
    )

print(f"Validation file: {validation_file.id}")
```

---

## Step 2: Create a Fine-Tuning Job

```python
job = client.fine_tuning.jobs.create(
    training_file=training_file.id,
    validation_file=validation_file.id,
    model="gpt-4.1-mini",
    hyperparameters={
        "n_epochs": 3,
        "batch_size": "auto",
        "learning_rate_multiplier": "auto",
    },
    suffix="support-bot-v1",
)

print(f"Job created: {job.id}")
print(f"Status: {job.status}")
print(f"Base model: {job.model}")
```

### Hyperparameters Explained

| Parameter | Description | Default | Guidance |
|-----------|-------------|---------|----------|
| `n_epochs` | Training passes over data | Auto (usually 3) | More epochs = risk of overfitting |
| `batch_size` | Examples per training step | Auto | Larger = more stable, slower |
| `learning_rate_multiplier` | How fast weights update | Auto | Lower = more conservative |

**Rule of thumb for dataset size:**

| Examples | Recommended Epochs |
|----------|-----------------|
| 50-200 | 3-4 |
| 200-1000 | 2-3 |
| 1000+ | 1-2 |

---

## Step 3: Monitor Training Progress

```python
def monitor_fine_tuning(job_id: str, poll_interval: int = 30) -> str | None:
    """Monitor a fine-tuning job until completion."""
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)

        print(f"\nStatus: {job.status}")

        if job.status == "succeeded":
            print(f"Fine-tuned model: {job.fine_tuned_model}")
            if job.trained_tokens:
                print(f"Trained tokens: {job.trained_tokens}")
            return job.fine_tuned_model
        elif job.status == "failed":
            print(f"Error: {job.error}")
            return None
        elif job.status == "cancelled":
            print("Job was cancelled")
            return None

        events = client.fine_tuning.jobs.list_events(
            fine_tuning_job_id=job_id,
            limit=5,
        )
        for event in events.data:
            print(f"  [{event.created_at}] {event.message}")

        print(f"Waiting {poll_interval}s...")
        time.sleep(poll_interval)

fine_tuned_model = monitor_fine_tuning(job.id)
```

### Reading Training Metrics

During training, OpenAI reports loss curves in the dashboard and via events:

| Metric | Healthy Pattern | Warning Sign |
|--------|----------------|--------------|
| **Training loss** | Steady decrease | Stays flat (bad data or too few examples) |
| **Validation loss** | Decreases then plateaus | Increases while training loss drops (overfitting) |
| **Training tokens** | Matches expected count | Much lower than expected (data issues) |

If validation loss increases while training loss keeps dropping, reduce `n_epochs` or add more diverse training data.

---

## Step 4: Test Your Fine-Tuned Model

```python
def compare_models(
    prompt: str,
    system_prompt: str = "",
    base_model: str = "gpt-4.1-mini",
    fine_tuned_model_id: str = None,
) -> dict:
    """Compare base model vs fine-tuned model responses."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    base_response = client.chat.completions.create(
        model=base_model,
        messages=messages,
        temperature=0.7,
    )

    ft_response = client.chat.completions.create(
        model=fine_tuned_model_id,
        messages=messages,
        temperature=0.7,
    )

    return {
        "prompt": prompt,
        "base": base_response.choices[0].message.content,
        "fine_tuned": ft_response.choices[0].message.content,
        "base_tokens": base_response.usage.total_tokens,
        "ft_tokens": ft_response.usage.total_tokens,
    }

# Test with domain-specific prompts
test_prompts = [
    "What is your return policy for electronics?",
    "My order hasn't arrived after 2 weeks. What should I do?",
    "Can I exchange a product I bought 45 days ago?",
    "Do you offer student discounts?",
    "How do I cancel my subscription?",
]

for prompt in test_prompts:
    result = compare_models(
        prompt=prompt,
        system_prompt="You are a customer service agent for TechStore.",
        fine_tuned_model_id=fine_tuned_model,
    )
    print(f"\n--- Prompt: {prompt} ---")
    print(f"BASE:       {result['base'][:200]}")
    print(f"FINE-TUNED: {result['fine_tuned'][:200]}")
```

---

## Step 5: Evaluate and Iterate

Run a structured evaluation before deploying:

```python
# evaluation/compare.py
import json

EVAL_SET = [
    {
        "prompt": "What is your return policy?",
        "expected_keywords": ["30 days", "receipt", "refund"],
    },
    {
        "prompt": "How do I track my order?",
        "expected_keywords": ["tracking", "email", "account"],
    },
]

def evaluate_model(model_id: str, eval_set: list[dict]) -> dict:
    hits = 0
    for case in eval_set:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": case["prompt"]}],
            temperature=0.3,
        )
        answer = response.choices[0].message.content.lower()
        if any(kw.lower() in answer for kw in case["expected_keywords"]):
            hits += 1
    return {"model": model_id, "accuracy": hits / len(eval_set), "total": len(eval_set)}

base_score = evaluate_model("gpt-4.1-mini", EVAL_SET)
ft_score = evaluate_model(fine_tuned_model, EVAL_SET)
print(f"Base accuracy: {base_score['accuracy']:.0%}")
print(f"Fine-tuned accuracy: {ft_score['accuracy']:.0%}")
```

### Iteration Decision Tree

```
Fine-tuned model worse than base?
  ├── Yes → Check data quality, reduce epochs, add more examples
  └── No → Fine-tuned better?
        ├── Slightly better → Add more diverse examples, tune epochs
        └── Much better → Deploy! Monitor in production.
```

---

## Step 6: Manage Your Fine-Tuned Models

```python
# List all fine-tuning jobs
jobs = client.fine_tuning.jobs.list(limit=10)
for j in jobs.data:
    print(f"{j.id} | {j.status} | {j.fine_tuned_model or 'pending'} | {j.model}")

# Retrieve a specific job's details
job_detail = client.fine_tuning.jobs.retrieve(job.id)
print(f"Training file: {job_detail.training_file}")
print(f"Result files: {job_detail.result_files}")

# Cancel a running job
# client.fine_tuning.jobs.cancel("ftjob-abc123")

# Delete a fine-tuned model when no longer needed
# client.models.delete("ft:gpt-4.1-mini:org:support-bot-v1:abc123")
```

---

## Testing Your Build

### Verification Checklist

- [ ] Training file passes validation (no JSON errors, correct message format)
- [ ] Fine-tuning job reaches `succeeded` status
- [ ] Fine-tuned model responds to inference calls
- [ ] Fine-tuned responses differ meaningfully from base model
- [ ] Evaluation accuracy is higher on domain-specific prompts
- [ ] Validation loss did not diverge from training loss

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Job fails immediately | Invalid data format | Run `validate_jsonl()` from Step 1 |
| High training loss | Too few examples or bad data | Add 50+ quality examples |
| Validation loss increases | Overfitting | Reduce epochs to 1-2, add more data |
| Model ignores fine-tuning | System prompt conflicts | Use minimal or no system prompt during inference |
| Slow training | Large dataset or model | Expected — GPT-4.1 takes 30-60 min |
| Identical outputs | Too similar training examples | Increase diversity in training data |

---

## Deployment Notes

### Using Fine-Tuned Models in Production

Fine-tuned models use the same API interface as base models — swap the model ID:

```python
# In your production app
MODEL_ID = "ft:gpt-4.1-mini:org:support-bot-v1:abc123"  # or base model as fallback

response = client.chat.completions.create(
    model=MODEL_ID,
    messages=messages,
    temperature=0.3,  # Lower temperature for consistent support responses
)
```

### Production Checklist

- [ ] Store fine-tuned model ID in environment variables (not hardcoded)
- [ ] Implement fallback to base model if fine-tuned model is unavailable
- [ ] Monitor response quality weekly with an eval set
- [ ] Re-fine-tune when accuracy drops or policies change
- [ ] Track cost: fine-tuned models have different per-token pricing

### Cost Considerations

| Cost Type | When Charged |
|-----------|-------------|
| Training | Per token trained (one-time per job) |
| Inference | Per token input + output (ongoing) |
| Storage | Model stored until you delete it |

Fine-tuning pays off when you need consistent style/format and can reduce prompt length (fewer tokens per request).

---

## Extensions and Challenges

- **Multi-turn fine-tuning**: Include full conversation histories, not just single Q&A pairs
- **DPO fine-tuning**: Use Direct Preference Optimization with chosen/rejected response pairs
- **A/B testing**: Route 50% of traffic to fine-tuned model, compare satisfaction scores
- **Continuous fine-tuning**: Append new high-quality conversations monthly and re-train
- **Function calling fine-tune**: Fine-tune on examples that include tool use patterns

---

## Key Takeaways

- Fine-tuning on OpenAI is a 4-step process: upload data, create job, monitor, test
- Always use a validation file to detect overfitting early
- Start with default hyperparameters — they work well for most use cases
- Compare fine-tuned vs base model outputs to verify improvement
- Training typically takes 10-60 minutes depending on dataset size
- Fine-tuned models are drop-in replacements — same API, different model ID

---

## Next Lesson

**Lesson 4: Fine-Tuning Open-Source Models with Hugging Face** — Learn to fine-tune Llama, Mistral, and other open-source models using the Hugging Face ecosystem.
