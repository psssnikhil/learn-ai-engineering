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
- An OpenAI API key with fine-tuning permissions
- A validated JSONL training file (from Lesson 2)
- Python 3.8+ with the `openai` package installed

```bash
pip install openai
```

---

## Step 1: Upload Your Training File

```python
from openai import OpenAI
import time

client = OpenAI()

# Upload training file
with open("training_data.jsonl", "rb") as f:
    training_file = client.files.create(
        file=f,
        purpose="fine-tune"
    )

print(f"File uploaded: {training_file.id}")
print(f"Status: {training_file.status}")
print(f"Filename: {training_file.filename}")
```

You can also upload a **validation file** (recommended) to track overfitting:

```python
with open("validation_data.jsonl", "rb") as f:
    validation_file = client.files.create(
        file=f,
        purpose="fine-tune"
    )
```

---

## Step 2: Create a Fine-Tuning Job

```python
# Create fine-tuning job
job = client.fine_tuning.jobs.create(
    training_file=training_file.id,
    validation_file=validation_file.id,  # Optional but recommended
    model="gpt-4.1-mini",  # Base model to fine-tune
    hyperparameters={
        "n_epochs": 3,
        "batch_size": "auto",
        "learning_rate_multiplier": "auto"
    },
    suffix="my-custom-model"  # Custom name suffix
)

print(f"Job created: {job.id}")
print(f"Status: {job.status}")
print(f"Model: {job.model}")
```

### Hyperparameters Explained

| Parameter | Description | Default | Guidance |
|-----------|-------------|---------|----------|
| `n_epochs` | Training passes over data | Auto (usually 3) | More epochs = risk of overfitting |
| `batch_size` | Examples per training step | Auto | Larger = more stable, slower |
| `learning_rate_multiplier` | How fast weights update | Auto | Lower = more conservative |

---

## Step 3: Monitor Training Progress

```python
def monitor_fine_tuning(job_id, poll_interval=30):
    """Monitor a fine-tuning job until completion."""
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        
        print(f"Status: {job.status}")
        
        if job.status == "succeeded":
            print(f"Fine-tuned model: {job.fine_tuned_model}")
            return job.fine_tuned_model
        elif job.status == "failed":
            print(f"Error: {job.error}")
            return None
        elif job.status == "cancelled":
            print("Job was cancelled")
            return None
        
        # Show recent training events
        events = client.fine_tuning.jobs.list_events(
            fine_tuning_job_id=job_id, limit=3
        )
        for event in events.data:
            print(f"  [{event.created_at}] {event.message}")
        
        print(f"Waiting {poll_interval}s...")
        time.sleep(poll_interval)

# Monitor the job (typically takes 10-60 minutes)
fine_tuned_model = monitor_fine_tuning(job.id)
```

### Training Metrics

During training, OpenAI reports:
- **Training loss**: Should decrease steadily (model is learning)
- **Validation loss**: Should decrease then plateau (not overfit)
- If validation loss starts increasing while training loss keeps dropping, the model is overfitting

---

## Step 4: Test Your Fine-Tuned Model

```python
def compare_models(prompt, system_prompt="", base_model="gpt-4.1-mini"):
    """Compare base model vs fine-tuned model responses."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    # Base model response
    base_response = client.chat.completions.create(
        model=base_model,
        messages=messages,
        temperature=0.7
    )
    
    # Fine-tuned model response
    ft_response = client.chat.completions.create(
        model=fine_tuned_model,
        messages=messages,
        temperature=0.7
    )
    
    print("=== Base Model ===")
    print(base_response.choices[0].message.content)
    print()
    print("=== Fine-Tuned Model ===")
    print(ft_response.choices[0].message.content)

# Test with a relevant prompt
compare_models(
    prompt="What is your return policy for electronics?",
    system_prompt="You are a customer service agent."
)
```

---

## Step 5: Manage Your Fine-Tuned Models

```python
# List all fine-tuning jobs
jobs = client.fine_tuning.jobs.list(limit=10)
for j in jobs.data:
    print(f"{j.id} | {j.status} | {j.fine_tuned_model or 'pending'}")

# Cancel a running job
# client.fine_tuning.jobs.cancel("ftjob-abc123")

# Delete a fine-tuned model (when no longer needed)
# client.models.delete("ft:gpt-4.1-mini:my-org:my-custom-model:abc123")
```

---

## Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Job fails immediately | Invalid data format | Run validation from Lesson 2 |
| High training loss | Too few examples or bad data | Add more quality examples |
| Validation loss increases | Overfitting | Reduce epochs, add more data |
| Model ignores fine-tuning | System prompt conflicts | Use minimal or no system prompt |
| Slow training | Large dataset or model | Expected — GPT-4.1 takes longer |

---

## Resources

- **OpenAI Fine-Tuning Guide**: [platform.openai.com/docs/guides/fine-tuning](https://platform.openai.com/docs/guides/fine-tuning)
- **OpenAI API Reference**: [platform.openai.com/docs/api-reference/fine-tuning](https://platform.openai.com/docs/api-reference/fine-tuning)
- **Video: Fine-Tuning GPT Models** — Step-by-step walkthrough of the OpenAI fine-tuning workflow

---

## Key Takeaways

- Fine-tuning on OpenAI is a 4-step process: upload data, create job, monitor, test
- Always use a validation file to detect overfitting early
- Start with default hyperparameters — they work well for most use cases
- Compare fine-tuned vs base model outputs to verify improvement
- Training typically takes 10-60 minutes depending on dataset size

---

## Next Lesson

**Lesson 4: Fine-Tuning Open-Source Models with Hugging Face** — Learn to fine-tune Llama, Mistral, and other open-source models using the Hugging Face ecosystem.
