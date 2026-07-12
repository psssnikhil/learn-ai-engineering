---
title: Self-Attention and Multi-Head Attention
description: >-
  Dive deep into the mechanics of self-attention and understand why multiple
  attention heads capture different types of linguistic relationships
duration: 45 min
difficulty: intermediate
has_code: true
module: module-00
---
# Self-Attention and Multi-Head Attention

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand self-attention vs cross-attention | 45 min | Intermediate |
| Implement scaled dot-product attention in code | | |
| Understand why multiple heads are better than one | | |
| See what different attention heads learn to detect | | |

---

## Self-Attention: A Word Looks at Its Own Sentence

In the previous lesson, we introduced attention in general. **Self-attention** is the specific variant where a sequence attends to **itself** — each word in a sentence looks at every other word in that same sentence to build a context-aware representation.

```
Self-attention: words in a sentence attend to OTHER words in the SAME sentence

Input:  "The cat sat on the mat"

For each word, self-attention asks:
  "Which other words in THIS sentence should I pay attention to?"

"sat" looks at: "The"(low) "cat"(high) "sat"(medium) "on"(low) "the"(low) "mat"(medium)
→ New vector for "sat" now encodes: "cat did the sitting, on the mat"
```

This is different from **cross-attention** (used in translation models), where words in one language attend to words in another language.

---

## Full Self-Attention Implementation

Let us implement self-attention from scratch to make the mechanics concrete:

```python
import numpy as np

def softmax(x, axis=-1):
    """Row-wise softmax."""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def self_attention(X, W_Q, W_K, W_V):
    """
    Scaled dot-product self-attention.

    X: (seq_len, d_model) — input embeddings for each token
    W_Q, W_K, W_V: (d_model, d_k) — learned projection matrices

    Returns:
        output: (seq_len, d_k) — context-aware representations
        weights: (seq_len, seq_len) — attention weight matrix
    """
    Q = X @ W_Q   # (seq_len, d_k) — Queries
    K = X @ W_K   # (seq_len, d_k) — Keys
    V = X @ W_V   # (seq_len, d_k) — Values

    d_k = K.shape[-1]

    # Compute attention scores: (seq_len, seq_len)
    scores = Q @ K.T / np.sqrt(d_k)

    # Normalize to probabilities
    weights = softmax(scores)

    # Weighted sum of values
    output = weights @ V  # (seq_len, d_k)

    return output, weights

# Example: 3 words, 4-dimensional embeddings
np.random.seed(42)
seq_len, d_model, d_k = 3, 4, 4

# Input embeddings (normally from an embedding layer)
X = np.random.randn(seq_len, d_model)

# Learned weight matrices
W_Q = np.random.randn(d_model, d_k) * 0.1
W_K = np.random.randn(d_model, d_k) * 0.1
W_V = np.random.randn(d_model, d_k) * 0.1

output, weights = self_attention(X, W_Q, W_K, W_V)

print("Attention weights (each row sums to 1):")
print(np.round(weights, 3))
print("
Output shape:", output.shape)
```

---

## The Limitation of Single-Head Attention

A single attention mechanism can only learn ONE type of relationship at a time. But language has many simultaneous relationships:

```
"The cat that I adopted from the shelter yesterday sat on the mat"

Relationships a model needs to capture simultaneously:
1. Syntactic:   "cat" → "sat" (subject-verb agreement)
2. Referential: "that" → "cat" (relative clause reference)
3. Temporal:    "adopted" → "yesterday" (when did it happen?)
4. Spatial:     "sat" → "on the mat" (where?)
5. Source:      "adopted" → "from the shelter" (origin)
```

A single set of Q, K, V weights tries to capture all of these at once — it is like trying to listen to five conversations with one ear. This is where **Multi-Head Attention** comes in.

---

## Multi-Head Attention: Multiple Perspectives at Once

Multi-Head Attention runs several attention operations **in parallel**, each with its own learned Q, K, V weight matrices. Each "head" can specialize in detecting a different type of relationship.

```
Multi-Head Attention (h=4 heads):

Head 1 (Q1, K1, V1): learns syntactic relationships (subject-verb)
Head 2 (Q2, K2, V2): learns positional/proximity patterns
Head 3 (Q3, K3, V3): learns referential relationships (pronouns)
Head 4 (Q4, K4, V4): learns semantic similarity

Each head produces its own output → concatenated → projected
```

```python
def multi_head_attention(X, heads):
    """
    Multi-head self-attention.

    X: (seq_len, d_model) — input embeddings
    heads: list of (W_Q, W_K, W_V) tuples, one per head

    Returns:
        output: (seq_len, d_model) — context-aware representations
    """
    head_outputs = []

    for W_Q, W_K, W_V in heads:
        head_output, _ = self_attention(X, W_Q, W_K, W_V)
        head_outputs.append(head_output)

    # Concatenate all head outputs along the feature dimension
    concatenated = np.concatenate(head_outputs, axis=-1)
    # (seq_len, d_k * num_heads)

    # Project back to d_model dimensions
    W_O = np.random.randn(concatenated.shape[-1], X.shape[-1]) * 0.1
    output = concatenated @ W_O
    # (seq_len, d_model)

    return output

# Set up 4 attention heads, each with smaller dimensions
np.random.seed(42)
d_model = 16
d_k = 4  # Each head works in d_model / num_heads dimensions
num_heads = 4

X = np.random.randn(5, d_model)  # 5 tokens, 16-dim embeddings

# Create weight matrices for each head
heads = []
for _ in range(num_heads):
    W_Q = np.random.randn(d_model, d_k) * 0.1
    W_K = np.random.randn(d_model, d_k) * 0.1
    W_V = np.random.randn(d_model, d_k) * 0.1
    heads.append((W_Q, W_K, W_V))

output = multi_head_attention(X, heads)
print("Input shape:", X.shape)    # (5, 16)
print("Output shape:", output.shape)  # (5, 16) — same shape, but now context-aware
```

---

## What Different Heads Learn

Research has shown that different attention heads in trained Transformers specialize in different linguistic patterns:

| Head Type | What It Detects | Example |
|-----------|----------------|---------|
| **Positional** | Adjacent words | "New" → "York" |
| **Syntactic** | Subject-verb relationships | "The cats" → "are" |
| **Coreference** | Pronoun resolution | "she" → "Marie Curie" |
| **Semantic** | Meaning similarity | "happy" → "joyful" |
| **Separator** | Sentence boundaries | "." → next sentence start |

This specialization emerges naturally during training — it is not programmed in. The model discovers that having specialized heads is useful for reducing prediction error.

---

## The Math: Multi-Head Attention Formula

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W^O

where head_i = Attention(X · W_i^Q, X · W_i^K, X · W_i^V)
```

In practice, with `d_model = 512` and `h = 8` heads:
- Each head operates on `d_k = d_model / h = 64` dimensions
- 8 heads × 64 dimensions = 512 dimensions after concatenation
- `W^O` projects back to `d_model = 512`

The total computation is similar to single-head attention with full dimensionality, but the model gets 8 different "perspectives" on the relationships in the data.

---

## Causal (Masked) Self-Attention

In language generation (GPT, Claude), the model must not look at future words when predicting the next word. **Causal masking** prevents this by setting future attention scores to negative infinity before softmax:

```python
def causal_self_attention(X, W_Q, W_K, W_V):
    """Self-attention with causal mask (for autoregressive generation)."""
    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)

    # Create causal mask: -inf for positions where j > i
    seq_len = X.shape[0]
    mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)
    scores = scores + mask

    weights = softmax(scores)
    output = weights @ V
    return output, weights

# With causal masking:
# Word 1 can only attend to: [word 1]
# Word 2 can only attend to: [word 1, word 2]
# Word 3 can only attend to: [word 1, word 2, word 3]
# ...and so on
```

This is why GPT-style models are called **autoregressive** — they generate one token at a time, only looking at previous tokens.

---

## Key Takeaways

- Self-attention lets each word attend to every other word in the same sequence
- A single attention head can only capture one type of relationship
- Multi-head attention runs several attention operations in parallel, each learning different patterns
- Different heads naturally specialize: syntax, coreference, proximity, semantics
- Causal masking prevents looking at future tokens during generation
- The computation is fully parallelizable — all heads and all positions compute simultaneously

## Resources

- [YouTube: Attention in Transformers, Visually Explained](https://www.youtube.com/watch?v=eMlx5fFNoYc) -- 3Blue1Brown's visual deep dive into how attention works
- [YouTube: Multi-Head Attention Explained](https://www.youtube.com/watch?v=mMa2PmYJlCo) -- CodeEmporium's clear explanation of multi-head attention
- [Jay Alammar: The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) -- Detailed visual walkthrough of attention within the Transformer
- [YouTube: Stanford CS224N — Self-Attention and Transformers](https://www.youtube.com/watch?v=ptuGllU5SQQ) -- Stanford lecture on self-attention mechanics

---

Next: The Complete Transformer Architecture
