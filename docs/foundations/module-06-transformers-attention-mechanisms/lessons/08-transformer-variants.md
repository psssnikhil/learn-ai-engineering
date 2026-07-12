---
title: Transformer Variants
description: Explore different Transformer architectures and their use cases
duration: 30 min
difficulty: intermediate
has_code: false
module: module-06
youtube: 'https://www.youtube.com/watch?v=MN__lSncZBs'
---
# Transformer Variants

## 1. Encoder-Only (BERT)

**Use**: Understanding tasks
- Text classification
- Named entity recognition
- Question answering

```
Input → Encoder → [CLS] token → Classification
```

**Example**: BERT, RoBERTa, ALBERT

## 2. Decoder-Only (GPT)

**Use**: Generation tasks
- Text generation
- Completion
- Few-shot learning

```
Prompt → Decoder → Next token prediction
```

**Example**: GPT-2, GPT-3, GPT-4, LLaMA

## 3. Encoder-Decoder (T5)

**Use**: Seq2seq tasks
- Translation
- Summarization
- Any input→output mapping

```
Input → Encoder → Decoder → Output
```

**Example**: T5, BART, mT5

## Efficiency Improvements

### Linformer
- O(n) instead of O(n²)
- Linear projection of K, V

### Performer
- Random feature attention
- Linear complexity

### Flash Attention
- Memory-efficient attention
- Faster on modern GPUs

## Key Differences

| Model | Architecture | Best For |
|-------|-------------|----------|
| BERT | Encoder-only | Understanding |
| GPT | Decoder-only | Generation |
| T5 | Enc-Dec | Seq2seq |
| RoBERTa | Encoder | Classification |
| BART | Enc-Dec | Summarization |

---

## 📹 Recommended Videos

- [BERT, GPT, T5 Explained](https://www.youtube.com/watch?v=MN__lSncZBs) — Comparison of major transformer variants
- [BERT Explained](https://www.youtube.com/watch?v=xI0HHN5XKDo) — CodeEmporium deep dive
- [GPT vs BERT vs T5](https://www.youtube.com/watch?v=TQQlZhbC5ps) — When to use which architecture

---

## 📚 Additional Resources

- [BERT Paper](https://arxiv.org/abs/1810.04805) — Devlin et al. original BERT paper
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — Language models as multitask learners
- [T5 Paper](https://arxiv.org/abs/1910.10683) — Exploring transfer learning with a unified text-to-text transformer
- [Hugging Face Model Hub](https://huggingface.co/models) — Browse and use pre-trained transformer models
