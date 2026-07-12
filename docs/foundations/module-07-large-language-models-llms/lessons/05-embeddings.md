---
title: Embeddings & Representations
description: Understand how tokens are converted to vectors and how LLMs represent meaning
duration: 25 min
difficulty: intermediate
has_code: false
module: module-07
youtube: 'https://www.youtube.com/watch?v=wjZofJX0v4M'
---
# Embeddings & Representations

## Token to Vector

```
Token ID: 1234
 ↓
Embedding Matrix [vocab_size × d_model]
 ↓
Vector: [0.2, -0.5, 0.1, ..., 0.3] (d_model dimensions)
```

## Learned Representations

LLMs learn to place similar words close together!

```
king - man + woman ≈ queen
Paris - France + Italy ≈ Rome
```

## Contextual Embeddings

**Static** (Word2Vec): "bank" always same vector
**Contextual** (BERT/GPT): "bank" vector depends on context

```
"river bank" → vector_A
"bank account" → vector_B
vector_A ≠ vector_B
```

## Implementation

```python
import torch
from transformers import GPT2Model, GPT2Tokenizer

model = GPT2Model.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

text = "The cat sat on the mat"
inputs = tokenizer(text, return_tensors='pt')

# Get embeddings
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state  # (1, seq_len, 768)

print(f"Embedding shape: {embeddings.shape}")
# Each token has a 768-dimensional vector!
```

---

## 📹 Recommended Videos

- [Word Embeddings Explained Visually](https://www.youtube.com/watch?v=wjZofJX0v4M) — 3Blue1Brown-style visualization of Word2Vec
- [Embeddings - What They Are and Why They Matter](https://www.youtube.com/watch?v=OATCgQtNX2o) — Practical embeddings overview
- [Sentence Transformers Tutorial](https://www.youtube.com/watch?v=HP9RJvBMHZA) — Using modern embedding models

---

## 📚 Additional Resources

- [Word2Vec Paper](https://arxiv.org/abs/1301.3781) — Efficient estimation of word representations
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — Benchmark for embedding models
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings) — Using OpenAI's embedding API
