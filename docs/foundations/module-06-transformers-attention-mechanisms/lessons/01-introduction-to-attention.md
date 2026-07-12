---
title: Introduction to Attention Mechanisms
description: >-
  Understand the revolutionary attention mechanism that transformed AI and
  powers modern language models
duration: 30 min
difficulty: intermediate
has_code: true
module: module-06
youtube: 'https://www.youtube.com/watch?v=XfpMkf4rD6E'
objectives:
  - Explain the limitations of RNNs that attention solves
  - Describe how attention weights are calculated
  - Implement basic attention mechanism
---
# Introduction to Attention Mechanisms

## The Problem with RNNs

Imagine reading a book and trying to remember everything from the first page when you're on page 300. That's what RNNs try to do!

**RNN Limitation**:
```
Input:  "The cat, which we bought last year, is sleeping"
Output: "Le chat dort"

Problem: By the time RNN reaches "sleeping", it has forgotten "cat"!
```

**The bottleneck**:
- All information compressed into fixed-size hidden state
- Long-range dependencies get lost
- Sequential processing (slow!)

---

## The Attention Breakthrough

**Key Idea**: Instead of remembering everything, just **pay attention** to relevant parts!

```
Translation: "The cat is sleeping"
         ↓
When translating "chat":
- Look at: "cat" ✅ (high attention)
- Ignore: "is", "sleeping" (low attention)
```

**Benefits**:
- ✅ Access any part of input directly
- ✅ Parallel processing
- ✅ Interpretable (see what model focuses on)
- ✅ Handles long sequences

---

## How Attention Works: The Intuition

Think of it like a **spotlight** that can focus on different words:

```
Input: "The cat sat on the mat"

When processing "mat":
Attention weights:
- The: 0.05  ░
- cat: 0.10  ░░
- sat: 0.15  ░░░
- on:  0.60  ████████████
- the: 0.05  ░
- mat: 0.05  ░
```

The model **attends** most to "on" because it's most relevant!

---

## Attention Formula (Simplified)

```
1. Calculate similarity between query and each key
2. Convert to attention weights (softmax)
3. Use weights to combine values
```

**Math**:
```
Attention(Q, K, V) = softmax(Q·K^T) × V
```

Where:
- **Q** (Query): "What am I looking for?"
- **K** (Keys): "What do I have?"
- **V** (Values): "What should I return?"

---

## Visual Example

```
Sentence: "I love deep learning"

Query: "What does 'love' relate to?"

Keys:        Similarity:    Weights:     Values:
"I"          0.2           0.05    →    embedding_I
"love"       0.9           0.70    →    embedding_love  
"deep"       0.3           0.10    →    embedding_deep
"learning"   0.4           0.15    →    embedding_learning

Output = 0.05*emb_I + 0.70*emb_love + 0.10*emb_deep + 0.15*emb_learning
```

---

## Code: Basic Attention

```python
import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

def attention(query, keys, values):
    """
    Simple attention mechanism
    
    Args:
        query: (d,) - what we're looking for
        keys: (n, d) - what we have
        values: (n, d) - what to return
    
    Returns:
        output: (d,) - weighted combination of values
        weights: (n,) - attention weights
    """
    # Calculate similarity scores
    scores = keys @ query  # (n,)
    
    # Convert to attention weights
    weights = softmax(scores)  # (n,)
    
    # Weighted sum of values
    output = weights @ values  # (d,)
    
    return output, weights


# Example
query = np.array([1, 0])  # Looking for something
keys = np.array([
    [1, 0],  # Very similar!
    [0, 1],  # Not similar
    [0.7, 0.3]  # Somewhat similar
])
values = np.array([
    [10, 20],  # Value 1
    [30, 40],  # Value 2
    [50, 60]   # Value 3
])

output, weights = attention(query, keys, values)

print("Attention weights:", weights)
# [0.56, 0.19, 0.25] - highest weight on first key!

print("Output:", output)
# Weighted combination of values
```

---

## Attention in Practice

### Machine Translation

```
English: "I love AI"
French:  "J'aime l'IA"

When generating "aime" (love):
Attention weights on English words:
- I: 0.1
- love: 0.8  ← Focus here!
- AI: 0.1
```

### Text Summarization

```
Document: "Apple released new iPhone. The phone has great camera..."

Summarizing "phone features":
Attention:
- "Apple": 0.05
- "released": 0.05
- "iPhone": 0.20
- "phone": 0.15
- "great": 0.25  ← Important!
- "camera": 0.30 ← Very important!
```

---

## Types of Attention

### 1. Additive Attention (Bahdanau)
```python
score = W·tanh(W₁·query + W₂·key)
```

### 2. Multiplicative Attention (Luong)
```python
score = query·key
```

### 3. Scaled Dot-Product Attention (Transformer)
```python
score = (query·key) / √d
```
(We'll cover this in detail next lesson!)

---

## Complete Implementation

```python
import numpy as np

class Attention:
    def __init__(self, hidden_dim):
        # Learnable weights (would be trained)
        self.W_q = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.W_k = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.W_v = np.random.randn(hidden_dim, hidden_dim) * 0.01
    
    def forward(self, query, keys, values):
        """
        Args:
            query: (batch, d_model)
            keys: (batch, seq_len, d_model)
            values: (batch, seq_len, d_model)
        """
        # Project query, keys, values
        Q = query @ self.W_q  # (batch, d_model)
        K = keys @ self.W_k   # (batch, seq_len, d_model)
        V = values @ self.W_v # (batch, seq_len, d_model)
        
        # Calculate attention scores
        scores = K @ Q.T  # (batch, seq_len)
        
        # Attention weights
        weights = self.softmax(scores)  # (batch, seq_len)
        
        # Weighted sum
        output = weights.T @ V  # (batch, d_model)
        
        return output, weights
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# Usage
attention = Attention(hidden_dim=64)

# Single query
query = np.random.randn(1, 64)

# Sequence of keys and values
keys = np.random.randn(1, 10, 64)  # 10 time steps
values = np.random.randn(1, 10, 64)

output, weights = attention.forward(query, keys, values)

print("Output shape:", output.shape)  # (1, 64)
print("Attention weights shape:", weights.shape)  # (1, 10)
print("Attention weights sum:", weights.sum())  # Should be ~1.0
```

---

## Visualizing Attention

```python
import matplotlib.pyplot as plt
import seaborn as sns

def plot_attention(attention_weights, source_words, target_words):
    """
    Visualize attention weights as heatmap
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(attention_weights, 
                xticklabels=source_words,
                yticklabels=target_words,
                cmap='Blues',
                annot=True,
                fmt='.2f')
    plt.xlabel('Source (Input)')
    plt.ylabel('Target (Output)')
    plt.title('Attention Weights')
    plt.show()


# Example: Translation
source = ["I", "love", "AI"]
target = ["Je", "aime", "l'IA"]

# Simulated attention weights
weights = np.array([
    [0.8, 0.1, 0.1],  # "Je" attends to "I"
    [0.1, 0.8, 0.1],  # "aime" attends to "love"
    [0.1, 0.1, 0.8]   # "l'IA" attends to "AI"
])

plot_attention(weights, source, target)
```

---

## Why Attention Changed Everything

### Before Attention (2014)
```
RNN Encoder → Fixed Vector → RNN Decoder
            ↓
    Information Bottleneck!
```

### After Attention (2015+)
```
RNN Encoder → All Hidden States → Attention → RNN Decoder
              ↓                     ↓
        No bottleneck!      Focus on relevant parts!
```

### Impact
- **2015**: Attention for machine translation
- **2017**: Transformers (attention is all you need!)
- **2018**: BERT, GPT
- **2022**: ChatGPT
- **2024**: All modern LLMs use attention!

---

## 📹 Watch Next

- [StatQuest: Attention for Neural Networks](https://www.youtube.com/watch?v=XfpMkf4rD6E)
- [3Blue1Brown: Attention in Transformers](https://www.youtube.com/watch?v=eMlx5fFNoYc)
- [Illustrated Attention](https://www.youtube.com/watch?v=4Bdc55j80l8)

---

## 🎯 Key Takeaways

1. **Attention** solves RNN's long-range dependency problem
2. **Query-Key-Value** paradigm: search, match, retrieve
3. **Attention weights** show what model focuses on
4. **Parallel processing** makes it fast
5. **Foundation** for Transformers and all modern LLMs
6. Attention = `softmax(similarity(Q,K)) × V`

---

## 📹 Recommended Videos

- [Attention in Neural Networks](https://www.youtube.com/watch?v=W2rWgXJBZhU) — Computerphile clear explanation
- [Attention Is All You Need (paper walkthrough)](https://www.youtube.com/watch?v=iDulhoQ2pro) — Yannic Kilcher's deep paper review
- [Visual Guide to Transformer Neural Networks](https://www.youtube.com/watch?v=dichIcUZfOw) — Hedu AI animated walkthrough

---

## 📚 Additional Resources

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar's must-read visual guide
- [Attention Is All You Need (original paper)](https://arxiv.org/abs/1706.03762) — Vaswani et al. 2017
- [Visualizing Attention](https://distill.pub/2016/augmented-rnns/) — Distill.pub interactive article on attention mechanisms

---

## 🚀 Next Lesson

**Lesson 2**: Self-Attention Mechanism
- How attention works within a sequence
- The key innovation of Transformers
- Implementing self-attention
- Query, Key, Value in detail

**Get ready to dive deeper!** 🧠
