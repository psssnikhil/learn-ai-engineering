---
title: Fine-Tuning Open-Source Models with Hugging Face
description: >-
  Learn to fine-tune Llama, Mistral, and other open-source models using the
  Hugging Face Transformers and TRL libraries
duration: 50 min
difficulty: advanced
has_code: false
module: module-15
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

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **GPU access** | Minimum 8 GB VRAM (QLoRA) or 16 GB (LoRA) |
| **Hugging Face account** | With access token for gated models (Llama) |
| **Training data** | JSONL chat-format dataset from Lesson 2 |
| **Python 3.10+** | With CUDA-compatible PyTorch |

**Courses required:**
- Module 15, Lesson 1-2: Fine-tuning fundamentals and data preparation
- Module 15, Lesson 3: OpenAI fine-tuning (for comparison context)

```bash
pip install torch transformers datasets accelerate peft trl bitsandbytes huggingface_hub
huggingface-cli login  # Enter your HF token
```

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

## What You'll Build

By the end of this lesson, you will have:

- [ ] A LoRA-adapted Llama 3.1 8B model trained on custom data
- [ ] Training and validation loss curves showing convergence
- [ ] A saved adapter that can be loaded for inference
- [ ] A comparison of base vs fine-tuned model outputs
- [ ] (Optional) Model pushed to Hugging Face Hub

---

## Architecture

```
[Training Data]  training_data.jsonl
       |
       v
[Dataset Preparation]
  - Load JSONL
  - Apply chat template (Llama format)
  - Train/eval split (90/10)
       |
       v
[Model Loading]
  - Base: meta-llama/Llama-3.1-8B-Instruct
  - Quantization: 4-bit (QLoRA)
  - LoRA adapters on attention + MLP layers
       |
       v
[SFTTrainer]  (Supervised Fine-Tuning)
  - 3 epochs, cosine LR schedule
  - Gradient accumulation for effective batch size
  - Eval after each epoch
       |
       v
[Saved Adapter]  ./my-finetuned-model/
  - adapter_config.json
  - adapter_model.safetensors
       |
       v
[Inference]
  - Load base model + adapter
  - Generate responses
       |
       v
[Hub] (optional)
  - Push adapter to Hugging Face
```

---

## Step 1: Environment Setup and Hardware Check

```bash
pip install torch transformers datasets accelerate peft trl bitsandbytes nvitop
```

Verify GPU availability:

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

### Hardware Requirements

| Approach | GPU VRAM | Suitable GPUs |
|----------|----------|---------------|
| Full fine-tuning (7B model) | 40-80 GB | A100, H100 |
| LoRA fine-tuning (7B model) | 16-24 GB | RTX 4090, A10G, L4 |
| QLoRA fine-tuning (7B model) | 8-12 GB | RTX 3090, T4 |
| QLoRA fine-tuning (3B model) | 6-8 GB | RTX 3060, T4 |

---

## Step 2: Load the Base Model with QLoRA

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model_name = "meta-llama/Llama-3.1-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
model = prepare_model_for_kbit_training(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
```

---

## Step 3: Configure LoRA Adapters

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: ~21M || all params: ~8B || ~0.26% trainable
```

**Why LoRA?** Instead of updating all 8 billion parameters, LoRA trains small adapter matrices (~21M parameters). This reduces VRAM requirements by 4-8x while achieving comparable quality for domain adaptation.

---

## Step 4: Prepare Your Dataset

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files="training_data.jsonl", split="train")

def format_chat(example):
    messages = example["messages"]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

dataset = dataset.map(format_chat)
split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]

print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
print(f"Sample:\n{train_dataset[0]['text'][:300]}...")
```

Ensure your training data uses the correct chat format:

```json
{"messages": [
  {"role": "system", "content": "You are a helpful customer support agent."},
  {"role": "user", "content": "What is your return policy?"},
  {"role": "assistant", "content": "We offer a 30-day return policy for all electronics..."}
]}
```

---

## Step 5: Configure Training

```python
from trl import SFTTrainer, SFTConfig

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
    packing=True,
    report_to="none",
)
```

**Effective batch size** = `per_device_train_batch_size` × `gradient_accumulation_steps` = 4 × 4 = 16.

---

## Step 6: Train

```python
trainer = SFTTrainer(
    model=model,
    args=training_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)

print("Starting training...")
trainer.train()

trainer.save_model("./my-finetuned-model")
tokenizer.save_pretrained("./my-finetuned-model")
print("Training complete. Adapter saved to ./my-finetuned-model/")
```

Monitor these signals during training:

| Signal | Healthy | Problem |
|--------|---------|---------|
| Training loss | Steady decrease | Flat or increasing |
| Eval loss | Decreases then plateaus | Increases (overfitting) |
| GPU memory | Stable utilization | OOM errors (reduce batch size) |
| Training speed | ~1-3 steps/sec on T4 | Very slow (check data loading) |

---

## Step 7: Test Your Fine-Tuned Model

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, "./my-finetuned-model")
model.eval()

def generate_response(user_message: str, system_prompt: str = "") -> str:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
        )
    return tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )

# Compare base vs fine-tuned
print("=== Fine-Tuned ===")
print(generate_response("What is your return policy for electronics?"))

base_only = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=bnb_config, device_map="auto",
)
# Generate with base model for comparison...
```

---

## Step 8: Push to Hugging Face Hub

```python
from huggingface_hub import login

login()  # Or set HF_TOKEN environment variable

model.push_to_hub("your-username/my-finetuned-llama-support")
tokenizer.push_to_hub("your-username/my-finetuned-llama-support")
```

Others can then load your adapter:

```python
model = PeftModel.from_pretrained(
    base_model,
    "your-username/my-finetuned-llama-support",
)
```

---

## Testing Your Build

### Verification Checklist

- [ ] Model loads without OOM errors on your GPU
- [ ] Training loss decreases over epochs
- [ ] Eval loss does not diverge from training loss
- [ ] Fine-tuned model produces domain-relevant responses
- [ ] Adapter files saved correctly (adapter_config.json, adapter_model.safetensors)
- [ ] Inference works after reloading saved adapter

### Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA OOM | Reduce `per_device_train_batch_size` to 1-2, increase `gradient_accumulation_steps` |
| Loss not decreasing | Check data format, increase learning rate to 5e-4 |
| Gibberish output | Training diverged — reduce LR, check for bad data |
| Slow training on Colab | Enable `packing=True`, reduce `max_seq_length` to 1024 |
| Gated model access denied | Request access on Hugging Face model page, re-login |

---

## Deployment Notes

After fine-tuning, you have three deployment paths:

| Path | When to Use |
|------|------------|
| **LoRA adapter + base model** | Development, swapping adapters per task |
| **Merge adapter into base** | Production single-file deployment |
| **Push to Hub** | Sharing, community use, HF Inference Endpoints |

Merge adapter for standalone deployment:

```python
merged_model = model.merge_and_unload()
merged_model.save_pretrained("./my-merged-model")
tokenizer.save_pretrained("./my-merged-model")
```

See Lesson 8 for serving with vLLM or TGI.

---

## Extensions and Challenges

- **Multi-GPU training**: Use `accelerate launch` for data parallelism across GPUs
- **Different base models**: Try Mistral 7B, Gemma 2 9B, or Qwen 2.5 7B
- **DPO training**: Use TRL's `DPOTrainer` with preference pairs for alignment
- **Longer context**: Increase `max_seq_length` to 4096 or 8192 if your GPU allows
- **Custom chat templates**: Fine-tune models without native chat templates using raw text

---

## Key Takeaways

- Open-source fine-tuning gives you full control over model weights, data privacy, and deployment
- QLoRA makes it possible to fine-tune 7B+ models on consumer GPUs (8-16 GB VRAM)
- SFTTrainer from TRL simplifies the training loop to just a few configuration steps
- Always use a validation set and monitor eval loss to detect overfitting
- Push your model to Hugging Face Hub to share and deploy easily
- LoRA adapters are small (~50-200 MB) and can be swapped without reloading the base model

---

## Next Lesson

**Lesson 5: LoRA and Parameter-Efficient Fine-Tuning** — Deep dive into LoRA, QLoRA, and other PEFT techniques that make fine-tuning accessible on limited hardware.
