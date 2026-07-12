---
title: Fine-Tuning Techniques
description: Learn how to adapt pre-trained LLMs for specific tasks
duration: 40 min
difficulty: advanced
has_code: false
youtube: 'https://www.youtube.com/watch?v=eC6vEkmqWMA'
---
# Fine-Tuning LLMs

## What is Fine-Tuning?

**Pre-training**: Learn general language (unsupervised)
**Fine-tuning**: Adapt to specific task (supervised)

```
Pre-trained Model (General knowledge)
 ↓ Fine-tune on task data
Task-Specific Model (Specialized)
```

## Full Fine-Tuning

Update ALL parameters

```python
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

model = AutoModelForCausalLM.from_pretrained("gpt2")

# All parameters trainable
for param in model.parameters():
    param.requires_grad = True

# Training arguments
training_args = TrainingArguments(
    output_dir="./fine-tuned-model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

## LoRA (Low-Rank Adaptation)

**Idea**: Only train small adapter matrices

```
Original: Update W (d × d)
LoRA: Update A (d × r) and B (r × d) where r << d

W_new = W + AB

Parameters: d² → 2dr (much smaller!)
```

**Benefits**:
- 10-100x fewer parameters to train
- Faster training
- Less memory
- Multiple adapters for different tasks

```python
from peft import get_peft_model, LoraConfig

config = LoraConfig(
    r=8,  # Rank
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
)

model = get_peft_model(model, config)

# Only LoRA parameters trainable!
model.print_trainable_parameters()
# trainable params: 294,912 || all params: 124,734,720 || trainable%: 0.24%
```

## Prompt Tuning

Learn continuous prompts (soft prompts)

```
Hard prompt: "Translate to French: {text}"
Soft prompt: [learnable_vectors] {text}
```

Only tune prompt vectors, freeze model!

---

## 📹 Recommended Videos

- [Fine-Tuning LLMs Explained](https://www.youtube.com/watch?v=eC6vEkmqWMA) — Complete guide to fine-tuning
- [LoRA Explained](https://www.youtube.com/watch?v=PXWYUTMt-AU) — Low-rank adaptation for efficient fine-tuning
- [QLoRA Tutorial](https://www.youtube.com/watch?v=XpoKB3usmKc) — Quantized LoRA for consumer GPUs

---

## 📚 Additional Resources

- [LoRA Paper](https://arxiv.org/abs/2106.09685) — Low-rank adaptation of large language models
- [QLoRA Paper](https://arxiv.org/abs/2305.14314) — Efficient finetuning of quantized LLMs
- [Hugging Face PEFT](https://huggingface.co/docs/peft) — Parameter-efficient fine-tuning library
- [OpenAI Fine-Tuning Guide](https://platform.openai.com/docs/guides/fine-tuning) — Fine-tuning with the OpenAI API
