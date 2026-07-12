---
title: From Transformers to LLMs
description: Learn how the Transformer architecture evolved into modern LLMs
duration: 25 min
difficulty: intermediate
has_code: false
youtube: 'https://www.youtube.com/watch?v=SZorAJ4I-sA'
---
# From Transformers to LLMs

## The Evolution

**2017: Transformer** → **2018: BERT/GPT-1** → **2024: ChatGPT**

## Key Innovations

### 1. Decoder-Only Architecture (GPT)
```
Original Transformer: Encoder + Decoder
GPT simplification: Decoder only!

Why? Generation doesn't need encoder
```

### 2. Causal (Autoregressive) Attention
```
Each token can only attend to PREVIOUS tokens

"The cat sat on"
- "cat" can see: ["The"]
- "sat" can see: ["The", "cat"]  
- "on" can see: ["The", "cat", "sat"]
```

### 3. Massive Scale
```
GPT-1:   117M parameters
GPT-2:   1.5B parameters
GPT-3:   175B parameters
GPT-4:   ~1.7T parameters (rumored)
```

### 4. Pre-training + Fine-tuning
```
Step 1: Pre-train on massive text (learn language)
Step 2: Fine-tune on specific tasks (follow instructions)
```

## Architecture Comparison

**BERT (Encoder-only)**:
- Bidirectional attention
- Best for understanding
- Input → Classification/Extraction

**GPT (Decoder-only)**:
- Causal attention  
- Best for generation
- Prompt → Completion

**T5 (Encoder-Decoder)**:
- Both types
- Best for seq2seq
- Input → Transformed output

## Training Objective

**GPT**: Next token prediction
```
Input:  "The cat sat on the"
Target: "mat"

Loss: Cross-entropy between prediction and target
```

**BERT**: Masked language modeling
```
Input:  "The [MASK] sat on the mat"
Target: "cat"
```

---

## 📚 Additional Resources

- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) — Jay Alammar's visual guide
- [BERT Paper](https://arxiv.org/abs/1810.04805) — BERT: Pre-training of deep bidirectional transformers
- [GPT-3 Paper](https://arxiv.org/abs/2005.14165) — Language models are few-shot learners
