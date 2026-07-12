---
title: Self-Attention and Multi-Head Attention
description: >-
  Implement scaled dot-product self-attention and multi-head attention from
  scratch, understand why multiple heads emerge as specialists, and implement
  causal masking for autoregressive generation
duration: 75 min
difficulty: intermediate
has_code: true
module: module-00
---
# Self-Attention and Multi-Head Attention

## Prerequisites

- [Lesson 05: The Attention Mechanism](05-attention-mechanism.md) — Q, K, V framework, attention formula, numerical walkthrough

## What You'll Learn

| Objective | Why It Matters |
|-----------|---------------|
| Understand self-attention as a specific variant of attention | Unambiguous definition used in every Transformer paper |
| Implement self-attention with correct shape annotations | Shape errors are the most common implementation bug |
| Understand why a single head has limited expressivity | Motivates multi-head attention from first principles |
| Implement multi-head attention end-to-end | The exact computation in GPT, BERT, and every modern LLM |
| Implement causal masking | Required for understanding how GPT generates text |
| Analyze what different heads learn | Connects to interpretability research |

---

## Self-Attention vs. Cross-Attention

The previous lesson introduced attention in general. Now we need to be precise:

- **Self-attention**: queries, keys, and values all come from the **same** sequence
  - Used in the Transformer encoder (BERT-style) and in the GPT decoder
  - Each word in a sentence attends to every other word in that **same** sentence

- **Cross-attention**: queries come from one sequence; keys and values from **another**
  - Used in encoder-decoder models (original Transformer, T5) for the decoder to attend to the encoder output
  - Also the mechanism underlying RAG: the model attends to retrieved documents

```
Self-attention (decoder in GPT):
  Input: "The cat sat"
  Query source:  same sequence ["The", "cat", "sat"]
  Key/Value source: same sequence ["The", "cat", "sat"]
  → Each word attends to earlier words (with causal mask)

Cross-attention (translation model):
  Query source:  target sequence ["Le", "chat"]
  Key/Value source: source sequence ["The", "cat", "sat"]
  → Each target word attends to all source words
```

---

## Self-Attention: Complete Implementation

```python
import numpy as np

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    exp_x = np.exp(x - x.max(axis=axis, keepdims=True))
    return exp_x / exp_x.sum(axis=axis, keepdims=True)

def self_attention(X: np.ndarray,
                   W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray,
                   mask: np.ndarray = None):
    """
    Scaled dot-product self-attention.

    Args:
        X:   (seq_len, d_model)  — input token embeddings
        W_Q: (d_model, d_k)      — query projection
        W_K: (d_model, d_k)      — key projection
        W_V: (d_model, d_v)      — value projection
        mask:(seq_len, seq_len)  — optional causal mask (-inf for masked positions)

    Returns:
        output:  (seq_len, d_v)       — context-enriched representations
        weights: (seq_len, seq_len)   — attention weights for visualization
    """
    # Project inputs to Q, K, V
    Q = X @ W_Q    # (seq_len, d_k)
    K = X @ W_K    # (seq_len, d_k)
    V = X @ W_V    # (seq_len, d_v)

    d_k = K.shape[-1]

    # Compute scaled scores
    scores = Q @ K.T / np.sqrt(d_k)   # (seq_len, seq_len)

    # Apply mask if provided (set masked positions to -∞ before softmax)
    if mask is not None:
        scores = scores + mask

    # Convert scores to attention weights
    weights = softmax(scores, axis=-1)  # (seq_len, seq_len), rows sum to 1

    # Weighted combination of values
    output = weights @ V               # (seq_len, d_v)

    return output, weights

# ------ Example ------
np.random.seed(42)
seq_len, d_model, d_k, d_v = 5, 8, 4, 4

X   = np.random.randn(seq_len, d_model)
W_Q = np.random.randn(d_model, d_k) * 0.1
W_K = np.random.randn(d_model, d_k) * 0.1
W_V = np.random.randn(d_model, d_v) * 0.1

output, weights = self_attention(X, W_Q, W_K, W_V)

print(f"Input shape:   {X.shape}")       # (5, 8)
print(f"Output shape:  {output.shape}")  # (5, 4)
print(f"Weights shape: {weights.shape}") # (5, 5)
print(f"\nAttention weights (each row sums to 1):")
print(weights.round(3))
print(f"\nRow sums: {weights.sum(axis=1).round(4)}")  # all 1.0
```

### Shape Annotations — Critical for Implementation

Getting shapes wrong is the most common source of bugs in Transformer code. Let us be explicit:

```
For a single-head attention with:
  seq_len = n, d_model = d, d_k = d_k, d_v = d_v

X:      (n, d)                  — n tokens, each d-dimensional
W_Q:    (d, d_k)
W_K:    (d, d_k)
W_V:    (d, d_v)

Q = X @ W_Q:    (n, d) @ (d, d_k)  →  (n, d_k)   ← n query vectors
K = X @ W_K:    (n, d) @ (d, d_k)  →  (n, d_k)   ← n key vectors
V = X @ W_V:    (n, d) @ (d, d_v)  →  (n, d_v)   ← n value vectors

scores = Q @ K.T:   (n, d_k) @ (d_k, n)  →  (n, n)   ← ALL pairwise scores
weights = softmax(scores/√d_k):    (n, n)            ← row-stochastic matrix
output = weights @ V:   (n, n) @ (n, d_v)  →  (n, d_v) ← context-enriched

In GPT-3: n=2048, d=12288, d_k=128 (=d/96 heads), 96 layers
  → scores matrix per head per layer: 2048×2048 = 4M floats
  → across 96 heads × 96 layers: ~37 billion floats total
  → this is why KV caching is essential for efficient inference
```

---

## The Limitation of Single-Head Attention

A single set of Q, K, V weight matrices can only learn one "projection" of similarity. But language has many simultaneous relationship types that a single projection cannot capture at once:

```
"The cat that I adopted from the shelter yesterday sat on the mat"

A complete model needs to capture:
  1. Syntactic:   "cat" → "sat"        (subject-verb agreement)
  2. Referential: "that" → "cat"       (relative clause reference)
  3. Temporal:    "adopted" → "yesterday" (when did the action happen)
  4. Spatial:     "sat" → "on the mat" (where)
  5. Semantic:    "shelter" → "adopted" (adoption comes from shelters)
```

A single attention head optimizes its Q, K, V matrices to minimize total loss. With multiple competing relationship types, it can only approximate a weighted average — excelling at none of them fully.

**Analogy**: Trying to listen to five conversations simultaneously with a single microphone pointed in one direction. You need five microphones pointing in different directions.

---

## Multi-Head Attention

Run multiple attention operations in parallel, each with its own Q, K, V projections. Each head can specialize independently:

\[
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
\]

\[
\text{head}_i = \text{Attention}(X W_i^Q,\ X W_i^K,\ X W_i^V)
\]

Each head operates in a lower-dimensional subspace: if d_model = 512 and h = 8 heads, each head uses d_k = d_model / h = 64. After concatenation, the output is (seq_len, d_model) again — same shape as the input.

```python
def multi_head_attention(X: np.ndarray,
                         W_Qs: list, W_Ks: list, W_Vs: list,
                         W_O: np.ndarray):
    """
    Multi-head self-attention.

    Args:
        X:     (seq_len, d_model)
        W_Qs:  list of h matrices, each (d_model, d_k)
        W_Ks:  list of h matrices, each (d_model, d_k)
        W_Vs:  list of h matrices, each (d_model, d_v)
        W_O:   (h * d_v, d_model)  — output projection

    Returns:
        (seq_len, d_model) — context-aware representations at model dimension
    """
    head_outputs = []

    for W_Q, W_K, W_V in zip(W_Qs, W_Ks, W_Vs):
        head_out, _ = self_attention(X, W_Q, W_K, W_V)  # (seq_len, d_v)
        head_outputs.append(head_out)

    # Concatenate along feature dimension: (seq_len, h * d_v)
    concat = np.concatenate(head_outputs, axis=-1)

    # Project back to d_model: (seq_len, h*d_v) @ (h*d_v, d_model) → (seq_len, d_model)
    output = concat @ W_O

    return output

# ------ Example with 4 heads ------
np.random.seed(42)
d_model  = 16
num_heads = 4
d_k       = d_model // num_heads   # 4 per head
d_v       = d_model // num_heads   # 4 per head

X = np.random.randn(5, d_model)   # 5 tokens, 16-dim embeddings

W_Qs = [np.random.randn(d_model, d_k) * 0.1 for _ in range(num_heads)]
W_Ks = [np.random.randn(d_model, d_k) * 0.1 for _ in range(num_heads)]
W_Vs = [np.random.randn(d_model, d_v) * 0.1 for _ in range(num_heads)]
W_O  = np.random.randn(num_heads * d_v, d_model) * 0.1   # (16, 16)

output = multi_head_attention(X, W_Qs, W_Ks, W_Vs, W_O)
print(f"Input shape:  {X.shape}")       # (5, 16)
print(f"Output shape: {output.shape}")  # (5, 16) — same shape as input!
```

!!! note "Computational Efficiency of Multi-Head Attention"
    With h heads each at dimension d_k = d_model/h, the total computation is the same order as a single head at full dimension d_model. The advantage is representational diversity — multiple independent projections of the relationship space — without additional compute cost.

---

## What Different Heads Learn

Research (Clark et al. 2019; Michel et al. 2019) has shown that attention heads in trained Transformers develop interpretable specializations:

| Head Type | Pattern Detected | Example |
|-----------|----------------|---------|
| **Positional** | Adjacent tokens | "New" → "York", tokens next to each other |
| **Syntactic (subject-verb)** | Grammatical agreement | "The cats **are**" — "cats" → "are" |
| **Coreference** | Pronoun resolution | "Marie Curie…**she**" — "she" → "Marie Curie" |
| **Semantic similarity** | Related meanings | "happy" → "joyful", "dog" → "animal" |
| **Separator/delimiter** | Sentence boundaries | Periods, special tokens |
| **Direct object** | Verb-object structure | "ate" → "pizza" |

These specializations **emerge from training on next-token prediction** — they are not programmed. The model discovers that having dedicated heads for different relationship types reduces prediction error.

!!! warning "Not All Heads Are Interpretable"
    Only some heads show clean, interpretable patterns. Many heads appear to compute complex, distributed representations that do not map cleanly to linguistic categories. Attention is a mechanism, not a perfect linguistic analysis tool.

---

## Causal (Masked) Self-Attention

For autoregressive generation (GPT, Claude, LLaMA), the model must predict the next token using only *previously seen* tokens. Allowing the model to "peek" at future tokens during training would make the task trivially easy, and the model would not learn to generate coherently.

**Solution**: apply a causal mask that sets all future attention scores to −∞ before softmax, which makes their softmax output effectively 0.

```python
def make_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create a causal mask: -inf for future positions, 0 for current/past.

    For position i, the mask allows attending to positions j ≤ i.
    np.triu extracts the upper triangle (j > i), which we set to -inf.
    """
    mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)
    return mask

def causal_self_attention(X: np.ndarray,
                          W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray):
    """Self-attention with causal masking for autoregressive generation."""
    seq_len = X.shape[0]
    mask = make_causal_mask(seq_len)
    return self_attention(X, W_Q, W_K, W_V, mask=mask)

# Example: 4 tokens, showing what each position can attend to
seq_len = 4
mask = make_causal_mask(seq_len)
print("Causal mask:")
print(mask)
# [[  0., -inf, -inf, -inf],   ← position 0: can only attend to itself
#  [  0.,   0., -inf, -inf],   ← position 1: can attend to 0 and 1
#  [  0.,   0.,   0., -inf],   ← position 2: can attend to 0, 1, 2
#  [  0.,   0.,   0.,   0.]]   ← position 3: can attend to all

# After softmax, -inf → 0, so future positions get exactly 0 attention
X_demo = np.random.randn(seq_len, 8)
W_Q_demo = np.random.randn(8, 4) * 0.1
W_K_demo = np.random.randn(8, 4) * 0.1
W_V_demo = np.random.randn(8, 4) * 0.1

output, weights = causal_self_attention(X_demo, W_Q_demo, W_K_demo, W_V_demo)

print("\nCausal attention weights:")
print(weights.round(3))
# Upper triangle is all zeros (future positions masked out)
# Lower triangle sums to 1 for each row
```

### Why Causal Masking Enables Efficient Training

Without causal masking, training would require sequential processing (you can only give the model the tokens up to position t). With causal masking, the entire sequence is processed in parallel:

```
Training sequence: "The cat sat on the mat"
Without mask: process positions 0,1,2,3,4,5 separately → 6 forward passes
With mask:    process all positions simultaneously → 1 forward pass
              position i automatically only sees positions ≤ i (by mask)

For a batch of 32 sequences, 1024 tokens each:
  Without mask: 32 × 1024 = 32,768 forward passes
  With mask:    32 forward passes (one per example in batch)

This is why GPT training can process trillions of tokens efficiently.
```

---

## Multi-Head Attention in Real Models

| Model | d_model | Heads | d_k per head | Layers |
|-------|---------|-------|-------------|--------|
| GPT-2 (117M) | 768 | 12 | 64 | 12 |
| GPT-2 (1.5B) | 1600 | 25 | 64 | 48 |
| GPT-3 (175B) | 12288 | 96 | 128 | 96 |
| LLaMA 3 (8B) | 4096 | 32 | 128 | 32 |
| LLaMA 3 (70B) | 8192 | 64 | 128 | 80 |

Note: d_k = d_model / heads in all cases. This keeps total computation constant regardless of number of heads.

!!! note "Grouped Query Attention (GQA)"
    Modern efficient models (LLaMA 3, Mistral, Gemma) use **Grouped Query Attention**: multiple query heads share a smaller number of key/value heads. For example, 8 query heads sharing 1 KV head. This reduces KV cache size by 8x without significant quality loss. The extreme case (1 KV head for all queries) is called **Multi-Query Attention (MQA)**.

---

## Putting It Together: One Transformer Layer

```python
def transformer_encoder_layer(X: np.ndarray,
                               W_Qs, W_Ks, W_Vs, W_O,   # multi-head attention weights
                               W1, b1, W2, b2,           # feed-forward weights
                               gamma1, beta1,             # layer norm params (attention)
                               gamma2, beta2):            # layer norm params (ff)
    """
    One Transformer encoder layer.
    X: (seq_len, d_model)
    Returns: (seq_len, d_model)
    """
    def layer_norm(x, gamma, beta, eps=1e-6):
        """Normalize across the feature dimension."""
        mean = x.mean(axis=-1, keepdims=True)
        std  = x.std(axis=-1, keepdims=True)
        return gamma * (x - mean) / (std + eps) + beta

    # ── Sub-layer 1: Multi-Head Self-Attention ──
    attn_out = multi_head_attention(X, W_Qs, W_Ks, W_Vs, W_O)
    X = layer_norm(X + attn_out, gamma1, beta1)   # residual + norm

    # ── Sub-layer 2: Feed-Forward Network (applied position-wise) ──
    ff_hidden = np.maximum(0, X @ W1 + b1)         # (seq_len, d_ff)
    ff_out    = ff_hidden @ W2 + b2                 # (seq_len, d_model)
    X = layer_norm(X + ff_out, gamma2, beta2)       # residual + norm

    return X
```

---

## Edge Cases and Misconceptions

**"More heads always = better."** Not necessarily. Research has shown that pruning 50-90% of attention heads in trained BERT models causes minimal quality degradation. Many heads are redundant. The right number of heads is a hyperparameter tuned empirically.

**"The output projection W_O is optional."** W_O is essential. Without it, each head's output exists in a d_v-dimensional subspace and you cannot mix information across heads. W_O projects the concatenated outputs back to d_model and allows the model to recombine information from all heads.

**"Causal masking is only for the decoder."** In encoder-decoder architectures, the encoder uses full (bidirectional) self-attention, and the decoder uses causal self-attention. But GPT-style decoder-only models use causal attention everywhere. "Encoder-only" does not mean "uses causal attention."

**"d_k and d_v must be equal."** No — they can differ. In most implementations they are kept equal for simplicity. The attention weight matrix is (n, n) regardless of d_k; d_v only determines the output dimension per head.

---

## Production Connection

| Concept | Production Relevance |
|---------|---------------------|
| **KV cache** | At inference time, past K and V matrices are cached to avoid recomputation. Size = 2 × seq_len × d_v × num_heads × num_layers × bytes_per_param |
| **Flash Attention** | Reorders the attention computation to minimize memory reads/writes, achieving 2-4x speedup without changing the math |
| **GQA / MQA** | Reduces KV cache size, enabling longer contexts or faster serving |
| **Attention sink** | Observed phenomenon where the first token receives disproportionately high attention — relevant for understanding "system prompt" effects |

---

## Key Takeaways

- Self-attention is attention where Q, K, and V all come from the same input sequence
- Single-head attention can only learn one type of token relationship simultaneously
- Multi-head attention runs h parallel attention computations in lower-dimensional subspaces, concatenates, and projects — same compute cost as single-head at full dimension but richer representations
- Different heads naturally specialize in syntactic, semantic, positional, and referential relationships
- Causal masking (setting future positions to −∞) enables autoregressive training to use a single forward pass over the full sequence instead of n sequential passes
- In production, KV caching stores K and V for past tokens; GQA reduces this cache at slight quality cost

---

## Further Reading

- [Jay Alammar: The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — the definitive visual walkthrough of multi-head attention (read the full article)
- [Clark et al. (2019): What Does BERT Look At?](https://arxiv.org/abs/1906.04341) — analysis of what different attention heads learn
- [Dao et al. (2022): Flash Attention](https://arxiv.org/abs/2205.14135) — the algorithm that made long-context Transformers practical
- [3Blue1Brown: Attention in Transformers, Visually Explained](https://www.youtube.com/watch?v=eMlx5fFNoYc) — excellent visual complement to this lesson

---

**Next:** [The Complete Transformer Architecture](07-transformer-architecture.md)
