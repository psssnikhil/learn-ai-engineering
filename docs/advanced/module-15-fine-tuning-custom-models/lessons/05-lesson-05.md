---
title: LoRA and Parameter-Efficient Fine-Tuning (PEFT)
description: >-
  Deep dive into LoRA, QLoRA, and other parameter-efficient techniques that make
  fine-tuning accessible on limited hardware
duration: 40 min
difficulty: advanced
has_code: false
module: module-15
youtube: 'https://www.youtube.com/watch?v=YVU5wAA6Txo'
objectives:
  - Explain how LoRA reduces trainable parameters
  - Configure LoRA hyperparameters for different use cases
  - Implement QLoRA for memory-efficient training
  - Merge LoRA adapters back into the base model
---
# LoRA and Parameter-Efficient Fine-Tuning (PEFT)

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand the math behind LoRA | 40 min | Advanced |
| Choose optimal LoRA hyperparameters | | |
| Use QLoRA for memory-constrained environments | | |
| Merge, stack, and switch between LoRA adapters | | |

---

## The Problem: Full Fine-Tuning is Expensive

A 7B parameter model has ~14 GB of weights (in fp16). Full fine-tuning requires:
- Storing all weights: 14 GB
- Storing gradients: 14 GB
- Optimizer states (Adam): 28 GB
- **Total: ~56 GB VRAM** just for a 7B model

LoRA solves this by training only a tiny fraction of the parameters.

---

## How LoRA Works

LoRA (Low-Rank Adaptation) adds small trainable matrices alongside the frozen base model weights.

```
Standard fine-tuning:
  W_new = W_original + delta_W     (delta_W is the same size as W: huge!)

LoRA fine-tuning:
  W_new = W_original + B * A        (B and A are small matrices)
  
  If W is [4096 x 4096] (16M params):
    A is [4096 x r]  (e.g., r=16: 65K params)
    B is [r x 4096]  (e.g., r=16: 65K params)
    Total: 130K params instead of 16M (99.2% reduction!)
```

### The Key Insight

The weight updates during fine-tuning tend to be **low-rank** — meaning they can be well-approximated by the product of two small matrices. LoRA exploits this by directly learning these small matrices.

---

## LoRA Hyperparameters Explained

### Rank (`r`)

The most important hyperparameter. Controls the expressiveness of the adaptation.

```python
from peft import LoraConfig

# Conservative (simple tasks like format changes)
config_small = LoraConfig(r=8, lora_alpha=16)

# Standard (most fine-tuning tasks)
config_medium = LoraConfig(r=16, lora_alpha=32)

# Aggressive (complex reasoning, multi-task)
config_large = LoraConfig(r=64, lora_alpha=128)
```

| Rank | Trainable Params (7B model) | Best For |
|------|---------------------------|----------|
| 4 | ~2.6M (0.04%) | Simple style changes |
| 8 | ~5.2M (0.07%) | Format standardization |
| 16 | ~10.5M (0.14%) | Domain adaptation (recommended start) |
| 32 | ~21M (0.28%) | Complex task learning |
| 64 | ~42M (0.56%) | Multi-task, heavy adaptation |

### Alpha (`lora_alpha`)

Scaling factor for the LoRA update. Common rule of thumb: set `alpha = 2 * r`.

```python
# The effective update is scaled by: alpha / r
# So alpha=32, r=16 gives a scaling factor of 2.0
# Higher alpha = stronger adaptation (can cause instability)
```

### Target Modules

Which layers to apply LoRA to. More modules = more capacity but more VRAM.

```python
# Attention only (minimum, fastest training)
target_modules = ["q_proj", "v_proj"]

# All attention layers (recommended)
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Attention + MLP layers (maximum adaptation)
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]
```

### Dropout

```python
# Regularization to prevent overfitting
lora_dropout = 0.05  # 5% dropout (good default)
lora_dropout = 0.1   # 10% for very small datasets
```

---

## QLoRA: 4-bit Quantized LoRA

QLoRA combines quantization with LoRA to drastically reduce memory:

```python
import torch
from transformers import BitsAndBytesConfig

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # NormalFloat4 (best for LLMs)
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bf16
    bnb_4bit_use_double_quant=True,    # Quantize the quantization constants
)
```

### Memory Comparison (7B Model)

| Method | VRAM Required | Quality |
|--------|---------------|---------|
| Full fine-tuning (fp16) | ~56 GB | Best |
| LoRA (fp16) | ~18 GB | Very good |
| LoRA (8-bit) | ~12 GB | Good |
| QLoRA (4-bit) | ~8 GB | Good (slight quality trade-off) |

---

## Merging LoRA Adapters

After training, you can merge the LoRA adapter back into the base model for faster inference:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load base model (full precision for merging)
base_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "./my-lora-adapter")

# Merge adapter into base model
merged_model = model.merge_and_unload()

# Save the merged model (now a standalone model)
merged_model.save_pretrained("./my-merged-model")
tokenizer.save_pretrained("./my-merged-model")
```

### Why Merge?

- **No PEFT dependency needed** at inference time
- **Slightly faster inference** (no adapter overhead)
- **Easier deployment** — just a standard model
- **Quantize further** for production (GGUF, AWQ, etc.)

---

## Stacking Multiple LoRA Adapters

You can train multiple LoRA adapters for different tasks and switch between them:

```python
from peft import PeftModel

# Load base model
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# Load first adapter
model = PeftModel.from_pretrained(model, "./customer-service-adapter", adapter_name="customer_service")

# Load second adapter
model.load_adapter("./code-review-adapter", adapter_name="code_review")

# Switch between adapters at runtime
model.set_adapter("customer_service")  # Use customer service adapter
# ... generate response ...

model.set_adapter("code_review")  # Switch to code review adapter
# ... generate response ...
```

---

## Resources

- **LoRA Paper**: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
- **QLoRA Paper**: "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
- **Hugging Face PEFT Docs**: [huggingface.co/docs/peft](https://huggingface.co/docs/peft)
- **Blog: Practical Tips for LoRA**: Sebastian Raschka's guide to LoRA hyperparameters

---

## Key Takeaways

- LoRA trains only 0.1-0.5% of model parameters by learning low-rank update matrices
- Set rank `r=16` and `alpha=32` as a starting point for most tasks
- QLoRA adds 4-bit quantization, cutting VRAM from ~18 GB to ~8 GB for a 7B model
- Target attention layers at minimum; add MLP layers for stronger adaptation
- Merge adapters for deployment, or keep them separate to switch between tasks

---

## Next Lesson

**Lesson 6: Evaluation and Benchmarking Fine-Tuned Models** — Learn systematic approaches to measure whether your fine-tuning actually improved the model.
