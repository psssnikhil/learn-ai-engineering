---
title: Self-Attention Mechanism
description: >-
  Master self-attention, the core innovation that enables Transformers to
  process sequences in parallel
duration: 35 min
difficulty: intermediate
has_code: true
youtube: 'https://www.youtube.com/watch?v=yGTUuEx3GkA'
objectives:
  - Explain how self-attention differs from regular attention
  - 'Calculate Query, Key, Value matrices'
  - Implement scaled dot-product attention
---
# Self-Attention: The Heart of Transformers

## What is Self-Attention?

**Regular Attention**: Look at a different sequence
**Self-Attention**: Look at the same sequence (itself!)

```
Sentence: "The cat sat on the mat"

Self-attention asks for each word:
"Which OTHER words in this sentence are most relevant to me?"

For "cat":
- "The" (before cat): 0.3  ← grammatical link
- "sat" (action): 0.5     ← cat is doing this
- "mat" (location): 0.2   ← less relevant
```

---

## Why Self-Attention?

**Captures relationships within a sequence**:

```
"The animal didn't cross the street because it was too tired"

Q: What does "it" refer to?

Self-attention weights:
- animal: 0.7  ← High! "it" = "animal"
- street: 0.1  ← Low
- tired: 0.2   ← Related but not the referent
```

---

## The Three Matrices: Q, K, V

Every word creates three vectors:

**Query (Q)**: "What am I looking for?"
**Key (K)**: "What do I offer?"
**Value (V)**: "What information do I contain?"

```python
# For each word embedding
Q = embedding @ W_q  # Query: what I seek
K = embedding @ W_k  # Key: what I have
V = embedding @ W_v  # Value: my information
```

---

## Scaled Dot-Product Attention

**Formula**:
```
Attention(Q,K,V) = softmax(QK^T / √d_k) V
```

**Steps**:
1. **Similarity**: Q·K^T (how similar is each word to others?)
2. **Scale**: Divide by √d_k (prevents vanishing gradients)
3. **Normalize**: Softmax (convert to probabilities)
4. **Combine**: Multiply by V (weighted sum of values)

---

## Step-by-Step Example

```python
import numpy as np

# Simple example: 3 words, dimension=4
sentence = ["The", "cat", "sat"]

# Embeddings (3, 4)
embeddings = np.array([
    [1, 0, 0, 0],  # The
    [0, 1, 0, 0],  # cat
    [0, 0, 1, 0]   # sat
])

# Weight matrices (4, 4)
W_q = np.random.randn(4, 4) * 0.1
W_k = np.random.randn(4, 4) * 0.1
W_v = np.random.randn(4, 4) * 0.1

# Create Q, K, V
Q = embeddings @ W_q  # (3, 4)
K = embeddings @ W_k  # (3, 4)
V = embeddings @ W_v  # (3, 4)

# Calculate attention
d_k = Q.shape[-1]  # 4
scores = Q @ K.T / np.sqrt(d_k)  # (3, 3)

# Softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

attention_weights = softmax(scores)  # (3, 3)

# Final output
output = attention_weights @ V  # (3, 4)

print("Attention weights (each row = one word):")
print(attention_weights)
#       The   cat   sat
# The  [[0.33, 0.33, 0.34],
# cat   [0.30, 0.40, 0.30],
# sat   [0.32, 0.35, 0.33]]
```

---

## Complete Implementation

```python
import numpy as np

class SelfAttention:
    def __init__(self, d_model, d_k=None):
        """
        Args:
            d_model: Dimension of embeddings
            d_k: Dimension of Q, K (default: same as d_model)
        """
        self.d_model = d_model
        self.d_k = d_k if d_k else d_model
        
        # Learnable weight matrices
        self.W_q = np.random.randn(d_model, self.d_k) / np.sqrt(d_model)
        self.W_k = np.random.randn(d_model, self.d_k) / np.sqrt(d_model)
        self.W_v = np.random.randn(d_model, self.d_k) / np.sqrt(d_model)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Input embeddings (batch_size, seq_len, d_model)
            mask: Optional mask (batch_size, seq_len, seq_len)
        
        Returns:
            output: (batch_size, seq_len, d_k)
            attention_weights: (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Project to Q, K, V
        Q = x @ self.W_q  # (batch, seq_len, d_k)
        K = x @ self.W_k  # (batch, seq_len, d_k)
        V = x @ self.W_v  # (batch, seq_len, d_k)
        
        # Calculate attention scores
        scores = Q @ K.transpose(0, 2, 1)  # (batch, seq_len, seq_len)
        scores = scores / np.sqrt(self.d_k)  # Scale
        
        # Apply mask (if provided)
        if mask is not None:
            scores = scores + (mask * -1e9)  # Large negative number
        
        # Attention weights
        attention_weights = self.softmax(scores)  # (batch, seq_len, seq_len)
        
        # Weighted sum of values
        output = attention_weights @ V  # (batch, seq_len, d_k)
        
        return output, attention_weights
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# Usage
d_model = 512
seq_len = 10
batch_size = 2

attention = SelfAttention(d_model)

# Input: (batch, sequence, embedding_dim)
x = np.random.randn(batch_size, seq_len, d_model)

# Forward pass
output, weights = attention.forward(x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention weights shape: {weights.shape}")
print(f"Attention weights sum (should be 1): {weights[0].sum(axis=1)}")
```

---

## Masking in Self-Attention

**Why mask?** Prevent attention to certain positions.

### 1. Padding Mask

```python
# Sentence: "I love AI <PAD> <PAD>"
# Don't attend to padding tokens!

sentence = ["I", "love", "AI", "<PAD>", "<PAD>"]
padding_mask = np.array([
    [0, 0, 0, 1, 1],  # 1 = masked
    [0, 0, 0, 1, 1],
    [0, 0, 0, 1, 1],
    [0, 0, 0, 1, 1],
    [0, 0, 0, 1, 1]
])

# Apply: scores = scores + (mask * -1e9)
# Softmax will make masked positions ≈ 0
```

### 2. Look-Ahead Mask (for GPT-style models)

```python
# Don't look at future words!

look_ahead_mask = np.array([
    [0, 1, 1],  # Word 1: can only see itself
    [0, 0, 1],  # Word 2: can see word 1, 2
    [0, 0, 0]   # Word 3: can see all
])

# Upper triangle = future tokens = masked!
```

---

## Visualizing Self-Attention

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_attention(attention_weights, tokens):
    """
    Visualize which words attend to which
    """
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        attention_weights,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='YlOrRd',
        annot=True,
        fmt='.2f',
        cbar_kws={'label': 'Attention Weight'}
    )
    
    plt.xlabel('Key (attending to)')
    plt.ylabel('Query (attending from)')
    plt.title('Self-Attention Heatmap')
    plt.tight_layout()
    plt.show()


# Example
tokens = ["The", "cat", "sat", "on", "mat"]
# Simulated attention weights (5x5)
attention = np.array([
    [0.5, 0.3, 0.1, 0.05, 0.05],  # "The" attends mostly to itself & "cat"
    [0.2, 0.5, 0.2, 0.05, 0.05],  # "cat" attends to itself & neighbors
    [0.1, 0.2, 0.4, 0.2, 0.1],    # "sat" balanced
    [0.05, 0.05, 0.2, 0.5, 0.2],  # "on" attends to "sat" & "mat"
    [0.05, 0.05, 0.1, 0.3, 0.5]   # "mat" attends to "on" & itself
])

visualize_attention(attention, tokens)
```

---

## Complexity Analysis

**Time Complexity**: O(n² · d)
- n = sequence length
- d = embedding dimension

**Why n²?** Every token attends to every other token!

```
Sequence length: 100
Attention matrix: 100 × 100 = 10,000 comparisons

Sequence length: 1000
Attention matrix: 1000 × 1000 = 1,000,000 comparisons!
```

**Problem**: Long sequences are expensive!

**Solutions** (we'll cover later):
- Sparse attention
- Efficient Transformers (Linformer, Performer)
- Sliding window attention

---

## Self-Attention vs RNN

| Feature | RNN | Self-Attention |
|---------|-----|----------------|
| **Parallelization** | ❌ Sequential | ✅ Fully parallel |
| **Long dependencies** | ❌ Vanishing gradients | ✅ Direct connections |
| **Complexity** | O(n·d²) | O(n²·d) |
| **Memory** | O(1) per step | O(n²) |
| **Speed** | Slow | Fast (on GPU) |

---

## 🎯 Key Takeaways

1. **Self-attention** relates words within the same sequence
2. **Q, K, V** matrices enable flexible attention
3. **Scaling** by √d_k prevents gradient issues
4. **Masking** controls what tokens can see
5. **O(n²)** complexity for sequence length n
6. **Parallel** processing makes it GPU-friendly

---

## 🚀 Next Lesson

**Lesson 3**: Multi-Head Attention
- Multiple attention "heads"
- Different representation subspaces
- How it improves model capacity
- Implementation from scratch

**Let's go!** 💪

---

## 📚 Additional Resources

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar's visual guide to self-attention
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Original transformer paper
- [Lilian Weng: Attention](https://lilianweng.github.io/posts/2018-06-24-attention/) — Comprehensive attention survey
