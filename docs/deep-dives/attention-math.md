---
title: "Attention Math: Full QKV Derivation"
description: >-
  Derive scaled dot-product attention from scratch — shapes, projections,
  softmax numerics, and a complete numpy walkthrough on a 4-word sentence
---

# Attention Math: Full QKV Derivation

**Prerequisite**: [Transformers and Attention Mechanisms](../foundations/module-06-transformers-attention-mechanisms/index.md)

**What you'll get**: After this page you can derive the attention formula,
explain every shape, and reproduce the softmax numerics on a toy example
without looking anything up.

---

## Why Attention Exists

A neural network processes a sequence token by token. At position `i`, the
model needs to decide: *which other positions in the sequence contain information
relevant to understanding token `i`?*

Early sequence models (RNNs) answered this by compressing all context into a
fixed-size hidden vector — an information bottleneck. Attention replaces this
with a **dynamic weighted average**: at each position, look at *all* other
positions simultaneously and take a weighted sum of their representations,
where the weights reflect relevance.

The genius of the Transformer formulation is making this relevance score
**differentiable and parameterized** — the model learns what to pay attention to.

---

## The Three Matrices: Q, K, V

Every token's embedding `x_i ∈ ℝ^d_model` is linearly projected into three
different vector spaces:

```
Q_i = x_i · W_Q    (query:  "what am I looking for?")
K_i = x_i · W_K    (key:    "what do I contain?")
V_i = x_i · W_V    (value:  "what information do I provide?")
```

Shapes:
```
x_i:  (d_model,)          e.g. 512
W_Q:  (d_model, d_k)      e.g. 512 × 64
W_K:  (d_model, d_k)      e.g. 512 × 64
W_V:  (d_model, d_v)      e.g. 512 × 64

Q_i, K_i, V_i: (d_k,) or (d_v,)
```

For a full sequence of length `n`, we stack all token vectors into matrices:

```
X:  (n, d_model)
Q = X · W_Q   →  (n, d_k)
K = X · W_K   →  (n, d_k)
V = X · W_V   →  (n, d_v)
```

### Intuition for the Three Roles

Think of a library search:

- **Query** = your search request ("books about gradient descent")
- **Key** = the index card title for each book ("Optimization Methods", "Deep Learning", ...)
- **Value** = the actual content of the book

The attention mechanism computes how well each query matches each key, then
returns a weighted sum of the values. High-scoring keys contribute more to
the output.

The separation into Q, K, V (three different learned projections) lets the
model learn different representations for "what I'm searching for" vs.
"what I can be found as" vs. "what I actually output" — a powerful asymmetry.

!!! note "Why not just use X directly?"
    If Q = K = V = X, attention degenerates to a fixed weighting based only on
    raw embedding similarity — no learning about what features matter for each
    role. The projections W_Q, W_K, W_V are the learnable parameters.

---

## Scaled Dot-Product Attention: Full Derivation

### Step 1: Attention Scores

For each query-key pair, compute a **raw score** (logit) via dot product:

```
score(q_i, k_j) = q_i · k_j = Σ_d (q_i[d] × k_j[d])
```

In matrix form for the full sequence:

```
Scores = Q · K^T          shape: (n, n)

Scores[i, j] = q_i · k_j = "how much should token i attend to token j?"
```

### Step 2: Scaling

The raw dot product can get very large when `d_k` is large. To see why:

If `q_i` and `k_j` are independent random vectors with elements drawn from
`N(0, 1)`, then:

```
q_i · k_j = Σ_{d=1}^{d_k} q_i[d] × k_j[d]

E[q_i · k_j]     = 0              (zero mean)
Var[q_i · k_j]   = d_k            (variance grows with d_k)
Std[q_i · k_j]   = √(d_k)
```

With `d_k = 64`, raw dot products have standard deviation 8. Feeding large
values into softmax pushes it into the saturation region where gradients
vanish (near-zero gradients on all but the largest score).

Fix: divide by √(d_k):

```
Scaled_Scores = Q · K^T / √(d_k)   shape: (n, n)
```

After scaling, scores have standard deviation ≈ 1, keeping softmax in its
gradient-rich linear regime.

### Step 3: Masking (Optional)

For **decoder self-attention** (causal/autoregressive), we prevent token `i`
from attending to future token `j > i`. Add −∞ to all future positions:

```
if i < j:  Scaled_Scores[i, j] = -∞
```

After softmax, e^(−∞) = 0, so these positions contribute zero to the output.

For **encoder self-attention** and **cross-attention**, no causal mask is needed.

### Step 4: Softmax — Row-Wise

Apply softmax to each row (each query position attends over all key positions):

```
Weights[i, :] = softmax(Scaled_Scores[i, :])

softmax(z)_j = exp(z_j) / Σ_k exp(z_k)
```

Properties of these weights:
- All non-negative: `Weights[i, j] ≥ 0`
- Sum to 1 per row: `Σ_j Weights[i, j] = 1`
- This is a **probability distribution** over positions

### Step 5: Weighted Sum of Values

```
Output = Weights · V        shape: (n, d_v)

Output[i] = Σ_j Weights[i, j] × V[j]
```

`Output[i]` is a weighted combination of all value vectors, where positions
most relevant to query `i` get the highest weights.

### Full Formula

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

Shape summary for a sequence of length `n = 4`, `d_k = d_v = 3`:

```
Q:       (4, 3)
K:       (4, 3)
V:       (4, 3)
Q·K^T:   (4, 4)   ← score matrix: how much each token attends to each other
/√d_k:   (4, 4)   ← scaled
softmax: (4, 4)   ← attention weights
·V:      (4, 3)   ← output: each token's new representation
```

---

## Numerical Walkthrough

Let's compute attention by hand on the sentence **"The cat sat down"**.
We'll use tiny dimensions so every number fits on the page.

```python
import numpy as np

np.random.seed(42)

# ── 1. Setup ────────────────────────────────────────────────────────────────
n = 4          # sequence length: ["The", "cat", "sat", "down"]
d_model = 6    # embedding dimension
d_k = 3        # key/query dimension per head
d_v = 3        # value dimension per head

tokens = ["The", "cat", "sat", "down"]

# Toy embeddings (in practice from a learned embedding table)
X = np.array([
    [0.1, 0.8, 0.3, 0.5, 0.2, 0.9],   # "The"
    [0.9, 0.1, 0.7, 0.2, 0.8, 0.3],   # "cat"
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],   # "sat"
    [0.3, 0.7, 0.2, 0.9, 0.1, 0.6],   # "down"
])
# X shape: (4, 6)

# ── 2. Projection matrices (learned; random here for illustration) ────────
W_Q = np.random.randn(d_model, d_k) * 0.5   # (6, 3)
W_K = np.random.randn(d_model, d_k) * 0.5   # (6, 3)
W_V = np.random.randn(d_model, d_v) * 0.5   # (6, 3)

# ── 3. Compute Q, K, V ────────────────────────────────────────────────────
Q = X @ W_Q    # (4, 3)
K = X @ W_K    # (4, 3)
V = X @ W_V    # (4, 3)

print("Q (queries):")
for i, tok in enumerate(tokens):
    print(f"  {tok:6s}: {Q[i].round(3)}")

# ── 4. Scaled dot-product scores ──────────────────────────────────────────
scores = Q @ K.T / np.sqrt(d_k)   # (4, 4)

print("\nAttention scores (scaled, before softmax):")
print("         The    cat    sat   down")
for i, tok in enumerate(tokens):
    row = scores[i].round(3)
    print(f"  {tok:6s}: {row}")

# ── 5. Softmax ────────────────────────────────────────────────────────────
def softmax(x: np.ndarray) -> np.ndarray:
    # Numerically stable: subtract row max before exp
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

weights = softmax(scores)   # (4, 4)

print("\nAttention weights (post-softmax, each row sums to 1.0):")
print("         The    cat    sat   down")
for i, tok in enumerate(tokens):
    row = weights[i].round(3)
    print(f"  {tok:6s}: {row}  (sum={row.sum():.3f})")

# ── 6. Weighted sum of values ─────────────────────────────────────────────
output = weights @ V    # (4, 3)

print("\nAttention output (each token's new representation):")
for i, tok in enumerate(tokens):
    print(f"  {tok:6s}: {output[i].round(4)}")
```

**What to notice in the output:**

1. The score matrix `scores[i, j]` measures how much token `i` should attend
   to token `j`. High scores → high attention weight.
2. Each row of `weights` sums to exactly 1.0 — it's a valid probability
   distribution.
3. `output[i]` is a linear combination of all value vectors `V[j]`, weighted
   by how much token `i` attends to token `j`. The output for "cat" will be
   pulled toward the value representations of tokens it attends to most (likely
   nearby tokens and semantically related ones).
4. With random projections, the pattern looks random — the *learning* is in
   W_Q, W_K, W_V that get trained to make attention weights semantically meaningful.

---

## Multi-Head Attention

Single-head attention can only learn one type of relationship at a time.
**Multi-head attention** runs `h` independent attention heads in parallel,
each with its own W_Q^(i), W_K^(i), W_V^(i) projections:

```python
def multi_head_attention(X, W_Qs, W_Ks, W_Vs, W_O):
    """
    X:     (n, d_model)
    W_Qs:  list of h matrices, each (d_model, d_k)
    W_Ks:  list of h matrices, each (d_model, d_k)
    W_Vs:  list of h matrices, each (d_model, d_v)
    W_O:   (h * d_v, d_model)  — output projection
    """
    h = len(W_Qs)
    head_outputs = []

    for i in range(h):
        Q_i = X @ W_Qs[i]    # (n, d_k)
        K_i = X @ W_Ks[i]    # (n, d_k)
        V_i = X @ W_Vs[i]    # (n, d_v)

        scores_i = Q_i @ K_i.T / np.sqrt(d_k)   # (n, n)
        weights_i = softmax(scores_i)             # (n, n)
        head_i = weights_i @ V_i                  # (n, d_v)
        head_outputs.append(head_i)

    # Concatenate heads: (n, h * d_v)
    concat = np.concatenate(head_outputs, axis=-1)

    # Project back to d_model: (n, d_model)
    return concat @ W_O


# In GPT-2 small: h=12 heads, d_model=768, d_k=d_v=64
# Total parameters per attention layer:
#   W_Q + W_K + W_V per head: 3 × (768 × 64) × 12 = 1,769,472
#   W_O: (12 × 64) × 768 = 589,824
#   Total: ~2.36M per layer
```

Each head can specialize: one head might learn syntactic dependencies (subject-verb),
another semantic similarity (synonyms), another coreference (he/she → noun).

!!! note "Why d_k = d_model / h?"
    With `h` heads each of dimension `d_k = d_model / h`, the total computation
    is the same as one head with `d_k = d_model`. Multi-head attention adds
    representational power (multiple relationship types) at no extra FLOP cost.

---

## Common Misconceptions

### "Q, K, V are the input embeddings"

No. Q, K, V are the result of *projecting* the input embeddings through three
learned weight matrices. The input embeddings are X; Q = X·W_Q etc.

### "Attention is just similarity search"

Attention is *parameterized* similarity: the Q and K projections are learned
to make certain similarities more or less prominent. It's not cosine similarity
on the raw embeddings.

### "The softmax makes attention a hard selection"

Softmax is *soft* selection — every position gets some weight. Only when one
score is much larger than all others does the distribution approach one-hot.
In practice, attention is distributed across several positions.

### "You need causal masking in all Transformers"

Only in **decoders** doing autoregressive generation. Encoders (BERT-style) use
full bidirectional attention — every token can attend to every other token,
including future ones.

---

## Key Takeaways

- **Q, K, V** are three learned linear projections of the same input, letting
  the model represent "what I seek", "what I offer", and "what I provide" separately.
- **Scaling by √d_k** prevents softmax saturation when d_k is large.
- **Attention weights** are a per-query probability distribution over all positions —
  the output is a weighted sum of value vectors.
- **Multi-head attention** runs h independent attention operations in parallel
  and concatenates outputs, allowing the model to capture multiple relationship
  types simultaneously.
- The entire operation is differentiable end-to-end; W_Q, W_K, W_V, W_O are
  learned by gradient descent.

---

## Further Reading

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jalammar's visual walkthrough
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al. (2017), the original paper
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html) — full paper as executable code
- [Flash Attention](https://arxiv.org/abs/2205.14135) — IO-aware exact attention; the production implementation

← [Deep Dives Hub](index.md)
