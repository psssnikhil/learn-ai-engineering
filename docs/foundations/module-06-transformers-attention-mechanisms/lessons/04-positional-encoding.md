---
title: Positional Encoding
description: Understand how Transformers encode sequence order without recurrence
duration: 25 min
difficulty: intermediate
has_code: true
module: module-06
youtube: 'https://www.youtube.com/watch?v=dichIcUZfOw'
---
# Positional Encoding

## The Problem

Self-attention has **no sense of order**!

```
"Dog bites man" 
vs 
"Man bites dog"

Without positional info: SAME attention patterns! ❌
```

---

## Solution: Add Position Information

```
Word embedding + Position embedding = Final embedding
```

## Sinusoidal Positional Encoding

**Formula**:
```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Properties**:
- ✅ Unique for each position
- ✅ Works for any sequence length
- ✅ Relative positions encoded
- ✅ No learnable parameters

## Implementation

```python
def positional_encoding(seq_len, d_model):
    """
    Generate sinusoidal positional encodings
    """
    PE = np.zeros((seq_len, d_model))
    
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            # Even dimensions: sine
            PE[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            
            # Odd dimensions: cosine
            if i + 1 < d_model:
                PE[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    
    return PE


# Usage
seq_len = 100
d_model = 512

PE = positional_encoding(seq_len, d_model)

# Add to embeddings
embeddings = get_word_embeddings(tokens)  # (seq_len, d_model)
input_to_transformer = embeddings + PE
```

---

## Visualization

```python
import matplotlib.pyplot as plt

PE = positional_encoding(100, 128)

plt.figure(figsize=(12, 8))
plt.imshow(PE, aspect='auto', cmap='RdBu')
plt.xlabel('Embedding Dimension')
plt.ylabel('Position in Sequence')
plt.title('Positional Encoding Heatmap')
plt.colorbar()
plt.show()
```

Pattern: Wave-like, different frequencies for different dimensions!

---

## 📚 Additional Resources

- [Positional Encoding Explained](https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models/) — Machine Learning Mastery guide
- [RoPE: Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — Modern positional encoding used in LLaMA
- [ALiBi Paper](https://arxiv.org/abs/2108.12409) — Attention with linear biases
