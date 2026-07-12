---
title: Implementing Attention from Scratch
description: >-
  Build a complete, annotated multi-head attention mechanism in pure NumPy and
  PyTorch—every line explained, every shape verified
duration: 75 min
difficulty: advanced
has_code: true
module: module-06
youtube: 'https://www.youtube.com/watch?v=kCc8FmEb1nY'
objectives:
  - Implement scaled dot-product attention with shape annotations
  - Build multi-head attention with correct head splitting and merging
  - Add causal masking for decoder-style generation
  - Assemble a full Transformer block (MHA + FFN + LayerNorm + residual)
  - Profile and compare NumPy vs PyTorch implementations
---

# Implementing Attention from Scratch

## Prerequisites

- [Lesson 02: Self-Attention](./02-self-attention.md) — Q, K, V projections, scaled dot-product
- [Lesson 03: Multi-Head Attention](./03-multi-head-attention.md) — head splitting, output projection
- [Lesson 05: Complete Transformer Architecture](./05-transformer-architecture.md) — full block structure

## What You'll Learn

| Component | What you'll implement |
|-----------|----------------------|
| Scaled dot-product attention | Core attention function with masking |
| Multi-head attention | Full MHA class with Q/K/V/O projections |
| Causal mask | Upper-triangular mask for autoregressive decoding |
| Transformer block | MHA + FFN + LayerNorm + residuals |
| PyTorch version | Production-ready module with `F.scaled_dot_product_attention` |

---

## Why Build It from Scratch?

Reading a paper is not the same as debugging a shape mismatch at 3 AM. Building attention from scratch forces you to answer:

- Why does `Q @ K.T` give shape `(n, n)` not `(n, d_k)`?
- Why does dividing by `√d_k` matter numerically?
- Why does the causal mask use `-1e9` not `-inf`?
- What does head splitting actually look like in memory?

These questions only have meaningful answers when you see them in code.

---

## Step 1: Numerically Stable Softmax

Every attention implementation needs a solid softmax. The naive version overflows for large inputs.

```python
import numpy as np


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax.

    Subtracts max before exponentiating — shifts the largest value to exp(0)=1,
    preventing overflow while leaving softmax output unchanged (shift invariance).

    x : any shape
    returns: same shape, values sum to 1 along `axis`
    """
    x_max = x.max(axis=axis, keepdims=True)     # (B, h, n, 1) if x is (B,h,n,n)
    exp_x  = np.exp(x - x_max)                  # same shape
    return exp_x / exp_x.sum(axis=axis, keepdims=True)


# Sanity check: large inputs should not overflow
x_large = np.array([1000.0, 1001.0, 1002.0])
print("Stable softmax:", softmax(x_large))
# Should print values near [0.09, 0.24, 0.67] — not NaN

# Verify sum == 1
print("Sum:", softmax(x_large).sum())  # 1.0000
```

---

## Step 2: Scaled Dot-Product Attention

This is the core function. Everything else is just plumbing around it.

```python
def scaled_dot_product_attention(
    Q: np.ndarray,               # (B, h, T_q, d_k)  — queries
    K: np.ndarray,               # (B, h, T_k, d_k)  — keys
    V: np.ndarray,               # (B, h, T_k, d_v)  — values
    mask: np.ndarray | None = None,  # (B, h, T_q, T_k) bool — True = mask out
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scaled dot-product attention (Vaswani et al. 2017, Eq. 1).

    Returns
    -------
    output  : (B, h, T_q, d_v)   — weighted combination of values
    weights : (B, h, T_q, T_k)   — attention distribution (rows sum to 1)
    """
    d_k = Q.shape[-1]

    # Step 1: dot products between all queries and keys
    # (B, h, T_q, d_k) × (B, h, d_k, T_k) → (B, h, T_q, T_k)
    scores = Q @ K.transpose(0, 1, 3, 2)

    # Step 2: scale to prevent softmax saturation
    scores = scores / np.sqrt(d_k)

    # Step 3: apply mask (e.g. causal mask for decoder)
    if mask is not None:
        scores = np.where(mask, -1e9, scores)  # masked positions → -∞ → softmax ≈ 0

    # Step 4: softmax over keys (last dimension)
    weights = softmax(scores, axis=-1)   # (B, h, T_q, T_k)

    # Step 5: weighted sum of values
    # (B, h, T_q, T_k) × (B, h, T_k, d_v) → (B, h, T_q, d_v)
    output = weights @ V

    return output, weights


# Quick shape test
B, h, T, d_k, d_v = 2, 4, 6, 16, 16
Q = np.random.randn(B, h, T, d_k)
K = np.random.randn(B, h, T, d_k)
V = np.random.randn(B, h, T, d_v)

out, w = scaled_dot_product_attention(Q, K, V)
print(f"Output:  {out.shape}")    # (2, 4, 6, 16)
print(f"Weights: {w.shape}")      # (2, 4, 6, 6)
print(f"Row sum: {w[0, 0].sum(axis=-1)}")  # all 1.0
```

---

## Step 3: Causal Mask

For autoregressive (decoder-style) generation, token `i` must not attend to token `j > i`.

```python
def causal_mask(n: int) -> np.ndarray:
    """
    Upper-triangular boolean mask for causal attention.

    True  = MASK OUT (this position should be ignored)
    False = visible

    Shape: (n, n)

    Visual for n=4:
    [[F, T, T, T],   token 0 sees only itself
     [F, F, T, T],   token 1 sees 0,1
     [F, F, F, T],   token 2 sees 0,1,2
     [F, F, F, F]]   token 3 sees all
    """
    return np.triu(np.ones((n, n), dtype=bool), k=1)


# Visualize
mask_4 = causal_mask(4)
print(mask_4.astype(int))
# [[0 1 1 1]
#  [0 0 1 1]
#  [0 0 0 1]
#  [0 0 0 0]]

# Apply causal mask to attention
B, h, T = 1, 1, 4
Q_c = np.random.randn(B, h, T, 8)
K_c = np.random.randn(B, h, T, 8)
V_c = np.random.randn(B, h, T, 8)

# Broadcast mask to (1, 1, T, T) for batch and head dimensions
mask = causal_mask(T)[np.newaxis, np.newaxis, :, :]  # (1, 1, 4, 4)

out_c, w_c = scaled_dot_product_attention(Q_c, K_c, V_c, mask=mask)
print("\nCausal weights (should be upper-tri ≈ 0):")
print(w_c[0, 0].round(4))
```

---

## Step 4: Multi-Head Attention (Complete Class)

```python
class MultiHeadAttention:
    """
    Multi-head scaled dot-product attention (Vaswani et al. 2017, Section 3.2).

    Parameters
    ----------
    d_model   : int  — total embedding dimension (e.g. 512)
    num_heads : int  — number of parallel attention heads (e.g. 8)

    Key constraint: d_model must be divisible by num_heads.
    Each head operates on a d_k = d_model // num_heads dimensional subspace.
    """

    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads  # dimension per head

        # Initialization scale: keep variance ≈ 1 at layer input
        scale = 1.0 / np.sqrt(d_model)

        # Fused projection matrices: project all heads at once
        # W_Q, W_K, W_V: (d_model, d_model)  — same as (d_model, h × d_k)
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale  # output projection

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Reshape and transpose to separate heads.

        x      : (B, n, d_model)
        returns: (B, num_heads, n, d_k)

        Memory layout: contiguous reshape then transpose.
        The reshape treats d_model as (num_heads, d_k) — each head gets
        a contiguous slice of the projected representation.
        """
        B, n, _ = x.shape
        # Reshape: (B, n, d_model) → (B, n, h, d_k)
        x = x.reshape(B, n, self.num_heads, self.d_k)
        # Transpose: (B, n, h, d_k) → (B, h, n, d_k)
        return x.transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Reverse of _split_heads.

        x      : (B, num_heads, n, d_k)
        returns: (B, n, d_model)
        """
        B, h, n, d_k = x.shape
        # Transpose: (B, h, n, d_k) → (B, n, h, d_k)
        x = x.transpose(0, 2, 1, 3)
        # Reshape: (B, n, h, d_k) → (B, n, d_model)
        return x.reshape(B, n, h * d_k)

    def forward(
        self,
        query: np.ndarray,              # (B, T_q, d_model)
        key:   np.ndarray | None = None,  # (B, T_k, d_model) — None → self-attention
        value: np.ndarray | None = None,  # (B, T_k, d_model)
        mask:  np.ndarray | None = None,  # (B, 1, T_q, T_k) or (B, h, T_q, T_k)
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run multi-head attention.

        If key and value are None, performs self-attention (query=key=value=input).
        If key/value are provided, performs cross-attention.

        Returns
        -------
        output  : (B, T_q, d_model)
        weights : (B, num_heads, T_q, T_k)
        """
        # Default to self-attention if no separate key/value
        if key is None:
            key = query
        if value is None:
            value = query

        # 1. Linear projections — all heads computed at once
        Q = query @ self.W_Q   # (B, T_q, d_model)
        K = key   @ self.W_K   # (B, T_k, d_model)
        V = value @ self.W_V   # (B, T_k, d_model)

        # 2. Split into heads
        Q = self._split_heads(Q)   # (B, h, T_q, d_k)
        K = self._split_heads(K)   # (B, h, T_k, d_k)
        V = self._split_heads(V)   # (B, h, T_k, d_k)

        # 3. Attention per head
        heads, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        # heads   : (B, h, T_q, d_k)
        # weights : (B, h, T_q, T_k)

        # 4. Merge heads and apply output projection
        merged = self._merge_heads(heads)  # (B, T_q, d_model)
        output = merged @ self.W_O         # (B, T_q, d_model)

        return output, weights


# ── Full shape test ────────────────────────────────────────────────────────────
B, T, d_model, h = 2, 10, 512, 8

mha = MultiHeadAttention(d_model=d_model, num_heads=h)

x = np.random.randn(B, T, d_model)
output, weights = mha.forward(x)

print(f"Input:   {x.shape}")       # (2, 10, 512)
print(f"Output:  {output.shape}")  # (2, 10, 512) — same shape as input
print(f"Weights: {weights.shape}") # (2, 8, 10, 10)
print(f"Row sums (should be 1): {weights[0, 0].sum(axis=-1).round(4)}")
```

---

## Step 5: Layer Normalization and Feed-Forward

```python
class LayerNorm:
    """
    Layer normalization (Ba et al. 2016).

    Unlike BatchNorm, operates independently per token (last dimension).
    No cross-sample dependencies — safe for variable-length sequences.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        self.gamma = np.ones(d_model)   # learnable scale  (B, n, d_model) → broadcast
        self.beta  = np.zeros(d_model)  # learnable shift
        self.eps   = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x : (B, n, d_model) → (B, n, d_model)"""
        mean = x.mean(axis=-1, keepdims=True)     # (B, n, 1)
        std  = x.std(axis=-1, keepdims=True) + self.eps
        return self.gamma * (x - mean) / std + self.beta


class FeedForward:
    """
    Position-wise feed-forward network (Vaswani et al., Section 3.3).

    Two linear layers with ReLU: expand to d_ff, project back to d_model.
    Applied identically and independently to each position.
    """

    def __init__(self, d_model: int, d_ff: int):
        scale = 1.0 / np.sqrt(d_model)
        self.W1 = np.random.randn(d_model, d_ff)    * scale  # (d_model, d_ff)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff,   d_model)  * scale  # (d_ff, d_model)
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x : (B, n, d_model) → (B, n, d_model)

        Step 1: expand  → (B, n, d_ff)   using ReLU
        Step 2: project → (B, n, d_model)
        """
        h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU activation
        return h @ self.W2 + self.b2
```

---

## Step 6: Complete Transformer Block

```python
class TransformerBlock:
    """
    One Transformer encoder block (Pre-LayerNorm variant).

    Sub-layer 1: Multi-Head Self-Attention
    Sub-layer 2: Feed-Forward Network
    Both use residual connections and layer normalization.

    Pre-LN formulation (GPT-2 style):
        x = x + Sublayer(LayerNorm(x))
    This is more stable than Post-LN during training.
    """

    def __init__(
        self,
        d_model:   int,
        num_heads: int,
        d_ff:      int,
        dropout:   float = 0.0,
    ):
        self.attn  = MultiHeadAttention(d_model, num_heads)
        self.ffn   = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = dropout

    def _apply_dropout(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if not training or self.dropout == 0.0:
            return x
        mask = np.random.rand(*x.shape) > self.dropout
        return x * mask / (1.0 - self.dropout)

    def forward(
        self,
        x:        np.ndarray,              # (B, n, d_model)
        mask:     np.ndarray | None = None, # causal or padding mask
        training: bool = True,
    ) -> np.ndarray:                       # (B, n, d_model)
        """
        Forward pass through one Transformer encoder block.

        Shapes at each step:
            Input x           : (B, n, d_model)
            norm1(x)          : (B, n, d_model)
            attn output       : (B, n, d_model)
            x + attn (residual): (B, n, d_model)  ← no shape change
            norm2(x)          : (B, n, d_model)
            ffn output        : (B, n, d_model)
            x + ffn (residual): (B, n, d_model)  ← output same as input
        """
        # --- Sub-layer 1: Multi-Head Self-Attention ---
        x_normed   = self.norm1.forward(x)
        attn_out, _ = self.attn.forward(x_normed, mask=mask)
        attn_out   = self._apply_dropout(attn_out, training)
        x = x + attn_out          # residual connection

        # --- Sub-layer 2: Feed-Forward ---
        x_normed  = self.norm2.forward(x)
        ffn_out   = self.ffn.forward(x_normed)
        ffn_out   = self._apply_dropout(ffn_out, training)
        x = x + ffn_out           # residual connection

        return x                  # (B, n, d_model)


# ── End-to-end test ────────────────────────────────────────────────────────────
B, n, d_model, h, d_ff = 2, 8, 256, 4, 1024

block = TransformerBlock(d_model=d_model, num_heads=h, d_ff=d_ff)

x    = np.random.randn(B, n, d_model)
mask = causal_mask(n)[np.newaxis, np.newaxis, :, :]   # (1, 1, 8, 8)

out  = block.forward(x, mask=mask)

print(f"Block input:  {x.shape}")    # (2, 8, 256)
print(f"Block output: {out.shape}")  # (2, 8, 256) — same shape
print(f"Output range: [{out.min():.2f}, {out.max():.2f}]")
```

---

## Step 7: PyTorch Production Version

PyTorch provides `F.scaled_dot_product_attention` (PyTorch ≥ 2.0), which dispatches to Flash Attention when available.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttentionTorch(nn.Module):
    """
    Multi-head attention in PyTorch using F.scaled_dot_product_attention.

    This automatically uses Flash Attention on CUDA when available,
    reducing memory from O(n²) to O(n) through tiled computation.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads
        self.dropout   = dropout

        # Fused QKV projection (3× faster than 3 separate matmuls)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x:       torch.Tensor,              # (B, n, d_model)
        mask:    torch.Tensor | None = None, # (B, 1, n, n) bool — True = MASK
        is_causal: bool = False,            # use built-in causal mask
    ) -> tuple[torch.Tensor, None]:
        """
        Returns
        -------
        output  : (B, n, d_model)
        weights : None — not returned when using F.scaled_dot_product_attention
                  (Flash Attention never materializes the full weight matrix)
        """
        B, n, _ = x.shape
        h = self.num_heads

        # 1. Fused QKV projection
        qkv = self.qkv_proj(x)              # (B, n, 3 * d_model)
        Q, K, V = qkv.chunk(3, dim=-1)      # each (B, n, d_model)

        # 2. Reshape to (B, h, n, d_k) for multi-head attention
        def reshape(t):
            return t.view(B, n, h, self.d_k).transpose(1, 2)  # (B, h, n, d_k)

        Q, K, V = reshape(Q), reshape(K), reshape(V)

        # 3. F.scaled_dot_product_attention:
        #    - Applies √d_k scaling internally
        #    - Supports is_causal for free (no need to construct mask)
        #    - Uses Flash Attention kernel when on CUDA
        attn_mask = None
        if mask is not None:
            # Convert bool mask to additive mask (-inf where True)
            attn_mask = torch.where(mask, torch.tensor(float("-inf")), torch.zeros_like(mask, dtype=torch.float))

        output = F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )   # (B, h, n, d_k)

        # 4. Merge heads and project
        output = output.transpose(1, 2).contiguous().view(B, n, self.d_model)
        output = self.out_proj(output)     # (B, n, d_model)

        return output, None


class TransformerBlockTorch(nn.Module):
    """Full Transformer encoder block in PyTorch."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn  = MultiHeadAttentionTorch(d_model, num_heads, dropout)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),                # GELU preferred over ReLU in modern LLMs
            nn.Linear(d_ff, d_model),
        )
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        """x : (B, n, d_model) → (B, n, d_model)"""
        # Attention sublayer with Pre-LN
        attn_out, _ = self.attn(self.norm1(x), is_causal=is_causal)
        x = x + self.dropout(attn_out)

        # FFN sublayer with Pre-LN
        x = x + self.dropout(self.ffn(self.norm2(x)))

        return x


# Test
B, n, d_model, h, d_ff = 2, 16, 512, 8, 2048
block = TransformerBlockTorch(d_model, h, d_ff)

x = torch.randn(B, n, d_model)
out = block(x, is_causal=True)

print(f"Input:  {x.shape}")    # torch.Size([2, 16, 512])
print(f"Output: {out.shape}")  # torch.Size([2, 16, 512])
```

---

## Numerical Walkthrough: 3 Tokens, 2 Heads

Let's trace exactly what happens with a tiny example: 3 tokens, `d_model=4`, `h=2`, `d_k=2`.

```python
np.random.seed(0)
d_model, h, d_k = 4, 2, 2

# Toy input: 3 tokens with 4-dim embeddings
X = np.array([
    [1.0, 0.0, 1.0, 0.0],  # token 0
    [0.0, 1.0, 0.0, 1.0],  # token 1
    [1.0, 1.0, 0.0, 0.0],  # token 2
])  # (3, 4)

# Use identity projection for transparency
W_Q = W_K = W_V = np.eye(4)

Q_full = X @ W_Q  # (3, 4) — same as X
K_full = X @ W_K
V_full = X @ W_V

# Split into 2 heads by slicing d_model dimension
Q0, Q1 = Q_full[:, :2], Q_full[:, 2:]  # each (3, 2)
K0, K1 = K_full[:, :2], K_full[:, 2:]
V0, V1 = V_full[:, :2], V_full[:, 2:]

# Compute attention for head 0
scores0 = Q0 @ K0.T / np.sqrt(d_k)    # (3, 3)
print("Head 0 scores:\n", scores0.round(3))

weights0 = softmax(scores0, axis=-1)   # (3, 3)
print("Head 0 weights:\n", weights0.round(3))

head0_out = weights0 @ V0              # (3, 2)
print("Head 0 output:\n", head0_out.round(3))

# Similarly for head 1
scores1  = Q1 @ K1.T / np.sqrt(d_k)
weights1 = softmax(scores1, axis=-1)
head1_out = weights1 @ V1

# Concatenate heads
concat = np.concatenate([head0_out, head1_out], axis=-1)  # (3, 4)
print("Concatenated output:\n", concat.round(3))
```

---

## Common Bugs and How to Spot Them

!!! warning "Bug: Forgetting to scale by √d_k"
    Symptoms: training loss doesn't decrease; model produces uniform attention. Fix: check `scores / np.sqrt(d_k)` is present.

!!! warning "Bug: Causal mask applied in wrong direction"
    If you use `np.tril` instead of `np.triu` for the mask, you mask the *past* instead of the *future*. Always verify: token at position 0 should see only itself (all other positions masked).

!!! warning "Bug: Forgetting to broadcast mask over batch and heads"
    A mask of shape `(n, n)` needs to be `(1, 1, n, n)` for broadcasting over `(B, h, n, n)` attention scores.

!!! note "Numerical check: attention weights should sum to 1"
    After softmax, `weights.sum(axis=-1)` must be all 1s. If you see values like 0.97 or 1.03, you have a numerical precision issue in your softmax.

---

## Performance Comparison

| Implementation | Memory | Speed | Notes |
|----------------|--------|-------|-------|
| NumPy (this lesson) | O(n²) | Slow (CPU) | Educational, fully transparent |
| PyTorch naive | O(n²) | Fast (GPU matmuls) | Good baseline |
| PyTorch `F.scaled_dot_product_attention` | O(n²) storage, O(n) compute tiles | 2–4× faster | Default in PyTorch ≥ 2.0 |
| Flash Attention 2 | O(n) HBM | 4–8× faster than naive | Requires CUDA, used in production |

---

## Production Connection

**This code is the core of every LLM you use.** GPT-4, Claude, LLaMA-3 — all use multi-head attention with causal masking and residual connections exactly as shown here. The PyTorch implementation above, when scaled to billions of parameters and thousands of GPUs, *is* the production system.

**Flash Attention** (Dao et al. 2022, 2023) is the most important optimization beyond the basic algorithm. It avoids materializing the `(n, n)` attention matrix in GPU High-Bandwidth Memory (HBM) by tiling the computation. For `n=8192` and `h=32`, naive attention requires `32 × 8192² × 4 bytes ≈ 8.6 GB` just for attention weights. Flash Attention brings this to near-zero HBM usage.

**KV cache** during inference: the `K` and `V` matrices computed for previous tokens are cached and reused. Each new token only computes its own `Q`, then attends to all cached `K` and `V`. This reduces generation from O(T²) per token to O(T) per token.

---

## Key Takeaways

1. **Scaled dot-product attention** is a 5-line function: `Q @ K.T / √d_k → softmax → @ V`.
2. **Shape discipline** is everything: track `(B, h, T, d_k)` at every step; wrong shapes cause silent errors.
3. **Multi-head attention** splits d_model into h subspaces, computes attention per subspace, then merges — 4 weight matrices total (Q, K, V, O).
4. **Causal masking** uses upper-triangular True mask: `-1e9` before softmax → nearly-zero weight → token can't see the future.
5. **Residual connections** guarantee the output shape equals the input shape — this is what makes stacking blocks trivial.
6. **PyTorch's `F.scaled_dot_product_attention`** is the right choice for any real system — it dispatches to Flash Attention and is memory-efficient.

---

## Further Reading

- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — Harvard NLP full implementation with annotation
- [Karpathy: Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) — complete GPT in ~300 lines
- [Flash Attention 2 paper](https://arxiv.org/abs/2307.08691) — Dao 2023, why memory-efficient attention matters
- [deep-dive: attention-math.md](../../../deep-dives/attention-math.md) — full mathematical derivations including backpropagation
- [PyTorch SDPA docs](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) — API reference with Flash Attention dispatch

---

## Next Lesson

**[Lesson 10: Scaling Laws & Efficient Transformers](./10-scaling-laws.md)** — how performance scales with parameters, data, and compute; Flash Attention, mixed precision, and the engineering decisions that make LLMs possible.
