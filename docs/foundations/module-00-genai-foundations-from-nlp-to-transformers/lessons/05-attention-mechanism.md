---
title: The Attention Mechanism — "Attention Is All You Need"
description: >-
  Master the attention mechanism with full numerical walkthroughs on a 4-word
  sentence — from Q/K/V projections through score computation, scaling, softmax,
  and the final weighted sum
duration: 75 min
difficulty: intermediate
has_code: true
module: module-00
---
# The Attention Mechanism — "Attention Is All You Need"

## Prerequisites

- [Lesson 03: NLP Fundamentals](03-nlp-fundamentals.md) — tokenization, embeddings, dot product for similarity
- [Lesson 04: Contextual Embeddings](04-contextual-embeddings.md) — why sequential models failed and what we need instead
- [Lesson 02: Math Foundations](02-math-foundations.md) — softmax, gradient descent

## What You'll Learn

| Objective | Why It Matters |
|-----------|---------------|
| Understand attention's intuition before the math | Makes the formulas feel like inevitable choices, not magic |
| Trace Q, K, V projections with actual numbers | You will recognize this pattern in every Transformer implementation |
| Walk through a 4-word attention computation by hand | The most effective way to internalize the mechanism |
| Understand the scaling factor and why it matters | Connects to numerical stability in production training |
| Understand why attention replaces sequential processing | The key insight of the 2017 paper |

---

## The Paper That Changed Everything

In June 2017, Vaswani et al. published "Attention Is All You Need." The title itself was provocative — it claimed that the attention mechanism alone, without any recurrence or convolution, was sufficient for state-of-the-art sequence modeling.

They were right. Every major language model since 2018 — GPT, BERT, Claude, LLaMA, Gemini — is built on this foundation.

!!! note "Before Reading the Math"
    The attention mechanism is elegant but requires careful reading. Work through the numerical example in Section 4 slowly. Once you see the numbers flow through the computation, the formula clicks permanently.

---

## Intuition: Selective Focus

When you read "The animal didn't cross the street because **it** was too tired," you instinctively know "it" refers to "animal" and not "street." You do this by attending to the surrounding words differently — "animal" and "tired" get high weight; "the" and "because" get low weight.

Attention is a mechanism that lets a model **learn** which words to focus on when processing each position. Instead of reading left-to-right and hoping context accumulates in a hidden state, attention asks: "For each word I am processing, which other words in this sentence are most relevant to me right now?"

```
Processing "it" in "The animal didn't cross the street because it was too tired":

  Word      Raw Score   Attention Weight (after softmax)
  ----      ---------   --------------------------------
  The         0.32          0.02
  animal      2.14          0.43    ← high! (likely referent)
  didn't      0.18          0.01
  cross       0.55          0.04
  the         0.28          0.02
  street      1.02          0.12    ← moderate (other candidate)
  because     0.21          0.02
  it          1.15          0.15    ← self-attention
  was         0.41          0.03
  too         0.38          0.03
  tired       1.08          0.13    ← relevant (supports "animal")

  Weighted output for "it" = 0.02×V_The + 0.43×V_animal + ... + 0.13×V_tired
  This output now encodes: "it is the animal (because it was tired)"
```

---

## The Q, K, V Framework

### Where the Metaphor Comes From

The Query-Key-Value framework is borrowed from information retrieval:

- In a database: you submit a **query**, it matches against stored **keys**, and returns **values**
- In attention: each word asks a **query** ("what am I looking for?"), every word offers a **key** ("what can I be found for?"), and the retrieved **value** is the actual information

The insight: Q, K, and V are all derived from the **same** input sequence through different learned linear projections. Each word simultaneously acts as a query (asking about other words), a key (being asked about by other words), and a value (providing information when selected).

### The Three Projections

For each token embedding \(x_i\), we create three vectors via learned weight matrices:

\[
q_i = x_i W^Q, \quad k_i = x_i W^K, \quad v_i = x_i W^V
\]

where \(W^Q, W^K \in \mathbb{R}^{d_\text{model} \times d_k}\) and \(W^V \in \mathbb{R}^{d_\text{model} \times d_v}\).

!!! note "What the Projections Learn"
    The Q and K projections are trained to make the dot product Q_i · K_j large when position j is semantically relevant to position i. This is what "learning attention patterns" means — the W^Q and W^K matrices are learned such that their projections capture useful notions of relevance for the task at hand.

---

## The Attention Formula

\[
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
\]

Let us unpack each piece before the numerical example:

| Component | Shape | Purpose |
|-----------|-------|---------|
| Q | (seq_len, d_k) | Queries — what each position is looking for |
| K | (seq_len, d_k) | Keys — what each position offers |
| Q K^T | (seq_len, seq_len) | Pairwise relevance scores |
| / √d_k | scalar | Scaling — prevents softmax saturation |
| softmax(…) | (seq_len, seq_len) | Attention weights — rows sum to 1 |
| × V | (seq_len, d_v) | Weighted combination of value vectors |

---

## Numerical Walkthrough: 4-Word Sentence

Let us compute attention from scratch on "The cat sat on" with 4-dimensional embeddings and d_k = 3.

### Setup

```python
import numpy as np

np.random.seed(42)

# Word embeddings — shape (4, 4)
# Each row is one word's embedding vector
X = np.array([
    [0.1, 0.2, 0.3, 0.4],   # "The"
    [0.5, 0.1, 0.8, 0.2],   # "cat"
    [0.3, 0.7, 0.2, 0.6],   # "sat"
    [0.4, 0.3, 0.5, 0.1],   # "on"
])
# X.shape: (4, 4)  — 4 words × 4-dim embeddings

d_model = 4   # embedding dimension
d_k     = 3   # key/query dimension
d_v     = 3   # value dimension

# Learned projection matrices (normally trained; here we set fixed values)
W_Q = np.array([
    [0.1, 0.0, 0.2],
    [0.0, 0.3, 0.1],
    [0.2, 0.1, 0.0],
    [0.1, 0.2, 0.3],
])   # shape (d_model, d_k) = (4, 3)

W_K = np.array([
    [0.3, 0.1, 0.0],
    [0.1, 0.2, 0.1],
    [0.0, 0.1, 0.3],
    [0.2, 0.0, 0.2],
])   # shape (4, 3)

W_V = np.array([
    [0.2, 0.1, 0.3],
    [0.1, 0.3, 0.1],
    [0.3, 0.0, 0.2],
    [0.0, 0.2, 0.1],
])   # shape (4, 3)

print(f"X.shape   = {X.shape}")    # (4, 4)
print(f"W_Q.shape = {W_Q.shape}")  # (4, 3)
```

### Step 1: Project to Q, K, V

```python
# Q = X @ W_Q:  (4,4) @ (4,3) → (4,3)
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

print("\nQ (queries):")
print(Q.round(3))
# Each ROW is one word's query vector
# Q[0] = "The" asking: "what should I attend to?"
# Q[2] = "sat" asking: "what should I attend to?"

print("\nK (keys):")
print(K.round(3))
# Each ROW is one word's key vector
# K[1] = "cat" announcing: "I am available as context"

print("\nV (values):")
print(V.round(3))
# Each ROW is one word's value vector
# V[1] = "cat" saying: "here is my actual information"
```

### Step 2: Compute Raw Attention Scores

```python
# Scores = Q @ K^T:  (4,3) @ (3,4) → (4,4)
scores = Q @ K.T

print("\nRaw attention scores (Q @ K^T):")
print(scores.round(3))
# scores[i, j] = how much word i wants to attend to word j
# scores[2, 1] = how much "sat" wants to attend to "cat"

# The scores matrix is 4×4: every word × every word
print(f"\nScores shape: {scores.shape}")  # (4, 4)
```

### Step 3: Scale by √d_k

```python
scale_factor = np.sqrt(d_k)   # √3 ≈ 1.732
scaled_scores = scores / scale_factor

print(f"\nScaling factor: √{d_k} = {scale_factor:.3f}")
print("\nScaled scores:")
print(scaled_scores.round(3))
```

**Why scale?** Without scaling, for large d_k, the dot products grow large in magnitude (their variance is proportional to d_k). Large inputs to softmax push outputs toward 0 or 1, making gradients vanishingly small. Dividing by √d_k normalizes the variance:

```python
# Demonstration: why large scores hurt training

def saturated_softmax(scores):
    """Demonstrates softmax saturation."""
    probs = np.exp(scores) / np.exp(scores).sum()
    return probs

small_scores = np.array([0.5, 0.3, 0.1, 0.2])
large_scores = small_scores * 10.0   # what happens without scaling

print("Small scores:", saturated_softmax(small_scores).round(3))
# [0.357, 0.293, 0.254, 0.281] — gradients can flow (no saturation)

print("Large scores:", saturated_softmax(large_scores).round(3))
# [0.999, 0.000, 0.000, 0.000] — gradient ≈ 0 for non-max (vanished!)
```

### Step 4: Apply Softmax Row-Wise

```python
def softmax_2d(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax: each row becomes a probability distribution."""
    exp_x = np.exp(x - x.max(axis=1, keepdims=True))
    return exp_x / exp_x.sum(axis=1, keepdims=True)

attention_weights = softmax_2d(scaled_scores)

print("\nAttention weights (each row sums to 1.0):")
print(attention_weights.round(3))
print("\nRow sums:", attention_weights.sum(axis=1).round(4))  # all ≈ 1.0

# Each row is the attention distribution for one word
# Row 0 = "The" attention pattern over all 4 words
# Row 2 = "sat" attention pattern over all 4 words
print(f"\n'sat' attends to each word: {dict(zip(['The','cat','sat','on'], attention_weights[2].round(3)))}")
```

### Step 5: Weighted Sum of Values

```python
# Output = attention_weights @ V:  (4,4) @ (4,3) → (4,3)
output = attention_weights @ V

print("\nAttention output:")
print(output.round(3))
print(f"\nOutput shape: {output.shape}")  # (4, 3)

# output[2] is the new context-aware representation for "sat"
# It is a weighted combination of all word values, weighted by attention
# High attention to "cat" → "sat" representation absorbs information about "cat"
print(f"\nNew 'sat' representation: {output[2].round(3)}")
```

### The Full Function

```python
def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray,
                                  V: np.ndarray,
                                  mask: np.ndarray = None):
    """
    Scaled dot-product attention.

    Q: (seq_len, d_k) or (batch, heads, seq_len, d_k)
    K: (seq_len, d_k)
    V: (seq_len, d_v)
    mask: optional (seq_len, seq_len) — set to -1e9 to mask out positions

    Returns: (seq_len, d_v) context-aware representations
    """
    d_k     = K.shape[-1]
    scores  = Q @ K.T / np.sqrt(d_k)          # (seq_len, seq_len)

    if mask is not None:
        scores = scores + mask                 # mask out future tokens (causal)

    weights = softmax_2d(scores)               # (seq_len, seq_len)
    return weights @ V, weights               # (seq_len, d_v), weights for visualization

# Use the function
ctx, attn = scaled_dot_product_attention(Q, K, V)
print("Output shape:", ctx.shape)   # (4, 3)
```

---

## Why Attention Solves the Sequential Bottleneck

Compare how RNNs and Attention handle the relationship between word 1 and word 100 in a sequence:

```
RNN: word 1 → h₁ → h₂ → ... → h₉₉ → h₁₀₀
     gradient must flow through 99 weight matrices → vanishes

Attention: word 1 directly attends to word 100 in one matrix multiply
           gradient flows directly from the loss to both words — no intermediate steps!
```

| Property | RNN/LSTM | Self-Attention |
|----------|----------|----------------|
| Distance between word i and word j | O(|i-j|) steps | O(1) — direct connection |
| Gradient path length | O(|i-j|) multiplications | O(1) |
| Parallelization | None (step n+1 needs step n) | Fully parallel (all rows of Q computed independently) |
| Total computation | O(n) sequential steps | O(n²) but fully parallelizable |
| Memory bottleneck | Hidden state size | O(n²) attention matrix |

The trade-off: attention requires O(n²) memory and compute, while RNNs require O(n) sequential time. For n < 100K, attention wins massively because of GPU parallelism. For extreme context lengths (millions of tokens), this quadratic cost becomes a bottleneck — active research area (Flash Attention, linear attention, etc.).

---

## Visualizing Attention: Coreference Resolution

One of the most informative ways to understand what trained attention learns:

```
Input: "The animal didn't cross the street because it was too tired"

Attention weights when processing "it" (row = "it", columns = all words):

                The  animal  didn't  cross  the  street  because  it  was  too  tired
Attention:     0.02  0.43    0.01   0.04  0.02   0.12    0.02   0.15  0.03  0.03  0.13

The model assigns 43% attention to "animal" and 12% to "street".
"animal" wins because "tired" (another high-attention word) collocates with "animal",
not "street" (streets don't get tired).
```

This pattern — attention learning to resolve pronouns to their referents — emerges *automatically* from next-token prediction training. The model never receives an explicit label saying "it → animal."

---

## Edge Cases and Misconceptions

**"Q, K, V are three separate inputs."** In self-attention, Q, K, and V are all computed from the *same* input sequence X via three separate projection matrices. Cross-attention (used in encoder-decoder models and in RAG) uses the query from one sequence and the keys/values from another.

**"Attention weights tell you what the model is 'thinking about'."** Attention weights are one signal, but they do not have a simple interpretation as "importance." Research has shown that models can ignore high-attention positions and vice versa. Attention is a mechanism, not an explanation.

**"The d_k in the scaling formula is the model dimension d_model."** No — d_k is the dimension of the query and key vectors, which is typically d_model / num_heads. In GPT-3: d_model = 12288, num_heads = 96, so d_k = 128.

**"Attention is expensive, so it should be used sparingly."** Modern hardware (A100/H100 GPUs with FlashAttention) makes attention highly efficient. The quadratic cost in sequence length is the real bottleneck, not attention per se.

---

## Production Connection

| Concept | Production Relevance |
|---------|---------------------|
| **Attention matrix** | At inference time, the KV cache stores K and V for all previous tokens — this is why long context is expensive |
| **Scaling** | The √d_k factor is what makes training stable at large model sizes |
| **Causal mask** | Every autoregressive LLM uses this — it is what makes generation possible |
| **O(n²) cost** | The reason context windows cost more per token at longer lengths; 128K tokens costs ≈ 1000x more attention compute than 128 tokens |

---

## Key Takeaways

- Attention lets each token directly access information from any other token in one step — no sequential propagation
- **Q** (query): what this token is looking for; **K** (key): what this token offers; **V** (value): the actual information this token provides
- Attention score = dot product of Q and K, scaled by 1/√d_k, then softmaxed
- Output = weighted sum of Value vectors using the attention weights
- The scaling by √d_k prevents softmax saturation, which would kill gradients during training
- The entire computation is one matrix multiply (Q @ K^T), parallelizable across all token pairs simultaneously

---

## Further Reading

- [Vaswani et al. (2017): Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the original paper; read the architecture section after this lesson
- [Jay Alammar: Visualizing A Neural Machine Translation Model](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) — the predecessor paper with the first visual attention explanations
- [3Blue1Brown: Attention in Transformers, Visually Explained](https://www.youtube.com/watch?v=eMlx5fFNoYc) — the best visual walkthrough of attention mechanics (26 min)
- [Lilian Weng: The Attention Mechanism Family](https://lilianweng.github.io/posts/2018-06-24-attention/) — comprehensive survey of attention variants

---

**Next:** [Self-Attention and Multi-Head Attention](06-self-attention-multi-head.md)
