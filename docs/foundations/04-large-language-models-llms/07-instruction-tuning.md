---
title: Instruction Tuning
description: Learn how models like ChatGPT are trained to follow instructions
duration: 30 min
difficulty: advanced
has_code: false
youtube: 'https://www.youtube.com/watch?v=dbo3kNKPaUA'
---
# Instruction Tuning

## The Problem

Pre-trained models complete text, don't follow instructions:

```
Prompt: "Translate to French: Hello"
GPT-3 completion: "Translation services are available..."
❌ Not helpful!
```

## The Solution: Instruction Tuning

Train on (instruction, response) pairs

```
Instruction: "Translate to French: Hello"
Response: "Bonjour"
✅ Follows instruction!
```

## Dataset Format

```json
{
  "instruction": "Summarize this text in one sentence",
  "input": "Long article text here...",
  "output": "Brief one-sentence summary"
}
```

## Training Process

1. **Collect demonstrations**: Humans write good responses
2. **Supervised fine-tuning (SFT)**: Train model on demonstrations
3. **Result**: Model that follows instructions!

## InstructGPT / ChatGPT Approach

**3-stage process**:

### Stage 1: Supervised Fine-Tuning (SFT)
- Collect ~13K high-quality (prompt, response) pairs
- Fine-tune GPT-3 on these examples

### Stage 2: Reward Model Training
- Collect rankings: Which response is better?
- Train reward model to predict human preferences

### Stage 3: Reinforcement Learning (PPO)
- Use reward model to improve via RL
- Model learns to generate high-reward responses

## Key Datasets

**FLAN**: 62 text tasks, 1.8M examples
**InstructGPT**: Human-written examples
**Alpaca**: 52K synthetic instructions (from GPT-4)
**Dolly**: 15K human-generated
**OpenAssistant**: Community-driven

---

## 📹 Recommended Videos

- [Instruction Tuning and RLHF](https://www.youtube.com/watch?v=dbo3kNKPaUA) — Stanford CS224N lecture
- [How ChatGPT Was Trained](https://www.youtube.com/watch?v=VPRSBzXzavo) — From GPT-3 to ChatGPT
- [FLAN and Instruction Tuning](https://www.youtube.com/watch?v=RYtF-F5fON4) — Google's instruction tuning approach

---

## 📚 Additional Resources

- [InstructGPT Paper](https://arxiv.org/abs/2203.02155) — Training language models to follow instructions
- [FLAN Paper](https://arxiv.org/abs/2109.01652) — Finetuned language models are zero-shot learners
- [Alpaca: Stanford](https://crfm.stanford.edu/2023/03/13/alpaca.html) — Instruction-following LLaMA model
