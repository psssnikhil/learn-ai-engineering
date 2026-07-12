---
title: RLHF and Preference Tuning
description: >-
  Learn how Reinforcement Learning from Human Feedback and Direct Preference
  Optimization align models with human preferences
duration: 45 min
difficulty: advanced
has_code: false
youtube: 'https://www.youtube.com/watch?v=2MBJOuVq380'
objectives:
  - Explain the RLHF pipeline and its components
  - Understand DPO as a simpler alternative to RLHF
  - Prepare preference data for alignment training
  - Fine-tune a model using DPO with TRL
---
# RLHF and Preference Tuning

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand the full RLHF pipeline | 45 min | Advanced |
| Compare RLHF, DPO, and other alignment methods | | |
| Prepare preference datasets | | |
| Implement DPO training with TRL | | |

---

## Why Preference Tuning?

Supervised fine-tuning (SFT) teaches a model **what** to say. Preference tuning teaches it **how** to say it — aligning outputs with human preferences for helpfulness, safety, and quality.

```
SFT: "Here is a good response. Learn to generate this."
RLHF/DPO: "Here are two responses. This one is better. Learn why."
```

This is how ChatGPT, Claude, and other chat models were trained to be helpful and safe.

---

## The RLHF Pipeline

### Step 1: Supervised Fine-Tuning (SFT)

Train the base model on high-quality instruction-response pairs (what we covered in Lessons 2-4).

### Step 2: Reward Model Training

Train a separate model to predict human preferences:

```
Input: (prompt, response)
Output: scalar reward score (higher = more preferred)
```

The reward model learns from human comparisons:
- Show annotators two responses to the same prompt
- They pick which one is better
- Train the reward model to assign higher scores to preferred responses

### Step 3: RL Optimization (PPO)

Use the reward model to guide further training via Proximal Policy Optimization:

```
For each training prompt:
  1. Generate a response with the current model
  2. Score it with the reward model
  3. Update the model to increase reward
  4. Add KL penalty to prevent diverging too far from SFT model
```

### The Challenge with RLHF

RLHF is powerful but complex:
- Requires training **three** models (SFT, reward, policy)
- PPO is unstable and sensitive to hyperparameters
- Reward hacking: model finds exploits in the reward model
- Expensive: needs significant compute and human annotation

---

## DPO: A Simpler Alternative

Direct Preference Optimization (DPO) achieves similar results without a reward model or RL.

### How DPO Works

Instead of training a reward model and then doing RL, DPO directly optimizes the language model on preference pairs:

```
For each training example:
  - Prompt: "Explain quantum computing"
  - Chosen response: (the human-preferred response)
  - Rejected response: (the less preferred response)
  
DPO loss pushes the model to:
  - Increase probability of chosen response
  - Decrease probability of rejected response
```

### DPO vs RLHF

| Aspect | RLHF (PPO) | DPO |
|--------|-----------|-----|
| Models needed | 3 (SFT + reward + policy) | 1 (just the policy model) |
| Stability | Unstable, needs tuning | Stable, straightforward |
| Compute cost | High (RL loop) | Lower (standard training) |
| Quality | Slightly better in some cases | Comparable for most tasks |
| Complexity | Very high | Moderate |

---

## Preparing Preference Data

### Data Format

```json
{
  "prompt": "What causes rain?",
  "chosen": "Rain forms when water vapor in the atmosphere condenses into droplets. As warm, moist air rises and cools, the water vapor reaches its dew point and forms clouds. When droplets combine and become heavy enough, they fall as precipitation.",
  "rejected": "Rain is water that falls from the sky. It happens because of clouds. The water cycle is involved somehow."
}
```

### Sources of Preference Data

1. **Human annotation**: Most reliable, most expensive
2. **AI feedback**: Use a stronger model to rank responses
3. **Existing datasets**: UltraFeedback, HH-RLHF, Nectar
4. **Implicit signals**: User upvotes/downvotes, regeneration requests

### Generating Preference Data with AI

```python
from openai import OpenAI
import json

client = OpenAI()

def generate_preference_pair(prompt, num_candidates=4):
    """Generate multiple responses and use GPT-4.1 to rank them."""
    # Generate candidates with different temperatures
    candidates = []
    for temp in [0.3, 0.7, 1.0, 1.2]:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=temp
        )
        candidates.append(response.choices[0].message.content)
    
    # Use GPT-4.1 to rank them
    ranking_prompt = f"""Rank these {len(candidates)} responses to the prompt: "{prompt}"

{chr(10).join(f'Response {i+1}: {c}' for i, c in enumerate(candidates))}

Return JSON: {{"best": <index>, "worst": <index>, "reasoning": "..."}}"""

    ranking = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": ranking_prompt}],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    result = json.loads(ranking.choices[0].message.content)
    
    return {
        "prompt": prompt,
        "chosen": candidates[result["best"] - 1],
        "rejected": candidates[result["worst"] - 1]
    }
```

---

## Implementing DPO with TRL

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig

# Load model (start from your SFT model)
model_name = "./my-sft-model"  # Your supervised fine-tuned model
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Load preference dataset
dataset = load_dataset("json", data_files="preferences.jsonl", split="train")

# LoRA config for DPO (train efficiently)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# DPO training config
dpo_config = DPOConfig(
    output_dir="./dpo-model",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    beta=0.1,  # Controls strength of preference (higher = more conservative)
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    max_length=1024,
    max_prompt_length=512,
)

# Create DPO trainer
trainer = DPOTrainer(
    model=model,
    args=dpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=peft_config,
)

# Train
trainer.train()
trainer.save_model("./dpo-model")
```

### Key Hyperparameter: Beta

The `beta` parameter controls how strongly the model follows preferences:

| Beta | Effect |
|------|--------|
| 0.05 | Strong preference following (risk of overfit) |
| 0.1 | Good default for most tasks |
| 0.2 | Conservative (safer, less dramatic changes) |
| 0.5 | Very conservative (minimal behavior change) |

---

## Resources

- **DPO Paper**: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" (Rafailov et al., 2023)
- **RLHF Paper**: "Training Language Models to Follow Instructions with Human Feedback" (Ouyang et al., 2022)
- **TRL Documentation**: [huggingface.co/docs/trl/dpo_trainer](https://huggingface.co/docs/trl/dpo_trainer)
- **UltraFeedback Dataset**: Large-scale AI preference dataset on Hugging Face

---

## Key Takeaways

- RLHF is the gold standard for alignment but complex (3 models, RL training loop)
- DPO achieves comparable results with just preference pairs and standard training
- Preference data needs a chosen (good) and rejected (bad) response for each prompt
- Use AI feedback from stronger models to generate preference data at scale
- Start with `beta=0.1` and adjust based on how aggressively you want to shift behavior

---

## Next Lesson

**Lesson 8: Deploying Fine-Tuned Models** — Learn to serve your fine-tuned models in production with vLLM, TGI, and cloud inference platforms.
