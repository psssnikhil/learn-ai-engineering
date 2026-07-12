---
title: Fine-Tuning Open-Source Models with Hugging Face
description: >-
  Learn to fine-tune Llama, Mistral, and other open-source models using the
  Hugging Face Transformers and TRL libraries
duration: 50 min
difficulty: advanced
has_code: false
youtube: 'https://www.youtube.com/watch?v=Q9zv369Elqk'
objectives:
  - Set up a Hugging Face fine-tuning environment
  - Load and prepare a model for training with PEFT
  - Fine-tune an open-source LLM on a custom dataset
  - Save and share your fine-tuned model
---
# Fine-Tuning Open-Source Models with Hugging Face

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Set up a fine-tuning environment for open-source models | 50 min | Advanced |
| Use the Transformers and TRL libraries | | |
| Fine-tune Llama or Mistral on custom data | | |
| Push your model to Hugging Face Hub | | |

---

## Why Open-Source Fine-Tuning?

| Factor | OpenAI Fine-Tuning | Open-Source Fine-Tuning |
|--------|-------------------|------------------------|
| Cost | Per-token training fee | GPU rental only ($1-5/hr) |
| Privacy | Data uploaded to OpenAI | Data stays on your machine |
| Control | Limited hyperparameters | Full control over everything |
| Model ownership | Hosted by OpenAI | You own the weights |
| Deployment | OpenAI API only | Deploy anywhere |
| Model choice | GPT-4.1, GPT-4.1-mini | Llama, Mistral, Gemma, Qwen, etc. |

---

## Environment Setup

```bash
# Install required packages
pip install torch transformers datasets accelerate peft trl bitsandbytes

# For GPU monitoring
pip install nvitop
```

### Hardware Requirements

| Approach | GPU VRAM | Suitable GPUs |
|----------|----------|---------------|
| Full fine-tuning (7B model) | 40-80 GB | A100, H100 |
| LoRA fine-tuning (7B model) | 16-24 GB | RTX 4090, A10G, L4 |
| QLoRA fine-tuning (7B model) | 8-12 GB | RTX 3090, T4 |
| QLoRA fine-tuning (3B model) | 6-8 GB | RTX 3060, T4 |

---

## Step-by-Step: Fine-Tuning with SFTTrainer

The `SFTTrainer` from Hugging Face's TRL library is the easiest way to fine-tune open-source models.

### Step 1: Load the Base Model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# Model configuration
model_name = "meta-llama/Llama-3.1-8B-Instruct"

# Quantization config for QLoRA (saves VRAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```

### Step 2: Configure LoRA

```python
# LoRA configuration
lora_config = LoraConfig(
    r=16,                      # Rank of the low-rank matrices
    lora_alpha=32,             # Scaling factor
    target_modules=[           # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 20,971,520 || all params: 8,051,232,768 || 0.26%
```

### Step 3: Prepare Your Dataset

```python
# Load a dataset (or use your own)
dataset = load_dataset("json", data_files="training_data.jsonl", split="train")

# For chat-format datasets, define a formatting function
def format_chat(example):
    """Format a training example as a chat conversation."""
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=False
    )
    return {"text": text}

# Apply formatting
dataset = dataset.map(format_chat)

# Split into train/eval
split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

print(f"Train examples: {len(train_dataset)}")
print(f"Eval examples: {len(eval_dataset)}")
```

### Step 4: Configure Training

```python
training_config = SFTConfig(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=10,
    save_strategy="epoch",
    eval_strategy="epoch",
    bf16=True,
    max_seq_length=2048,
    dataset_text_field="text",
    packing=True,  # Pack short examples together for efficiency
)
```

### Step 5: Train

```python
trainer = SFTTrainer(
    model=model,
    args=training_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)

# Start training
trainer.train()

# Save the LoRA adapter
trainer.save_model("./my-finetuned-model")
tokenizer.save_pretrained("./my-finetuned-model")
```

---

## Testing Your Fine-Tuned Model

```python
from peft import PeftModel

# Load the base model + LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "./my-finetuned-model")

# Generate a response
messages = [
    {"role": "user", "content": "What is your return policy for electronics?"}
]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, do_sample=True)
response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print(response)
```

---

## Pushing to Hugging Face Hub

```python
# Login to Hugging Face
from huggingface_hub import login
login()

# Push the adapter to Hub
model.push_to_hub("your-username/my-finetuned-model")
tokenizer.push_to_hub("your-username/my-finetuned-model")
```

---

## Resources

- **Hugging Face TRL Documentation**: [huggingface.co/docs/trl](https://huggingface.co/docs/trl)
- **PEFT Documentation**: [huggingface.co/docs/peft](https://huggingface.co/docs/peft)
- **Blog: Fine-Tune Llama 3 with Hugging Face**: Comprehensive tutorial on the Hugging Face blog
- **Google Colab Free GPUs**: Free T4 GPUs for small fine-tuning experiments

---

## Key Takeaways

- Open-source fine-tuning gives you full control over model weights, data privacy, and deployment
- QLoRA makes it possible to fine-tune 7B+ models on consumer GPUs (8-16 GB VRAM)
- SFTTrainer from TRL simplifies the training loop to just a few configuration steps
- Always use a validation set and monitor eval loss to detect overfitting
- Push your model to Hugging Face Hub to share and deploy easily

---

## Next Lesson

**Lesson 5: LoRA and Parameter-Efficient Fine-Tuning** — Deep dive into LoRA, QLoRA, and other PEFT techniques that make fine-tuning accessible on limited hardware.
