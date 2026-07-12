---
title: "Backpropagation Calculus: Chain Rule Through a 2-Layer Net"
description: >-
  Derive every gradient in a 2-layer neural network by hand — chain rule,
  Jacobians, vanishing gradients, and a complete numpy verification
---

# Backpropagation Calculus

**Prerequisite**: [Neural Networks & Deep Learning Fundamentals](../foundations/module-05-neural-networks-deep-learning-fundamentals/index.md)

**What you'll get**: After this page you can derive every weight gradient in a
2-layer network on paper, understand why vanishing gradients happen at the
calculus level, and verify your derivations with numpy.

---

## Intuition First

Training a neural network is an optimization problem: minimize a loss function
`L(θ)` over model parameters `θ`. Gradient descent requires `∂L/∂θ` — the
gradient of the loss with respect to every parameter.

For a single neuron, this is straightforward. The challenge is that a deep
network is a *composition* of many functions:

```
output = f_n(f_{n-1}(... f_2(f_1(input)) ...))
```

**Backpropagation** is just the chain rule applied systematically to this
composition, starting from the loss and working backward through each layer.
The chain rule is the only calculus you need.

---

## Network Architecture

We'll analyze a 2-layer fully-connected network for binary classification:

```
Input:    x ∈ ℝ^(n_0)

Layer 1:  z1 = W1·x + b1         linear
          a1 = σ(z1)             activation (sigmoid)

Layer 2:  z2 = W2·a1 + b2        linear
          a2 = σ(z2)             activation (sigmoid)

Loss:     L = BCE(a2, y)         binary cross-entropy
```

Parameters to learn: `W1 (n_1 × n_0)`, `b1 (n_1,)`, `W2 (n_2 × n_1)`, `b2 (n_2,)`.

For concreteness, we'll use:
- `n_0 = 2` (2 input features)
- `n_1 = 3` (3 hidden units)
- `n_2 = 1` (1 output, binary classification)

---

## Forward Pass

### Definitions

**Sigmoid activation**:
```
σ(z) = 1 / (1 + exp(-z))
```

**Binary cross-entropy loss** (for a single example):
```
L = -[y · log(a2) + (1 - y) · log(1 - a2)]
```

### Forward computation graph

```
x → [W1, b1] → z1 → σ → a1 → [W2, b2] → z2 → σ → a2 → L
```

Each arrow is a function; backprop reverses the arrows, multiplying local
derivatives (chain rule).

---

## Backward Pass: Chain Rule Derivation

### The Chain Rule

For composed functions `L = f(g(x))`:
```
dL/dx = (dL/df) · (df/dg) · (dg/dx)
```

In neural networks, each "function" is a layer, and we chain these
local derivatives from loss to inputs.

We define **error signals** (δ = backpropagated gradient) at each layer.

---

### Step 1: Gradient of L w.r.t. a2

```
L = -[y·log(a2) + (1-y)·log(1-a2)]

∂L/∂a2 = -y/a2 + (1-y)/(1-a2)
        = (a2 - y) / (a2·(1-a2))
```

### Step 2: Sigmoid Derivative — a Key Identity

```
σ(z) = 1 / (1 + e^(-z))

σ'(z) = σ(z) · (1 - σ(z))
```

Proof:
```
dσ/dz = d/dz [1 + e^(-z)]^{-1}
      = e^(-z) / (1 + e^(-z))^2
      = [1/(1+e^(-z))] · [e^(-z)/(1+e^(-z))]
      = σ(z) · (1 - σ(z))
```

!!! note "Why this matters"
    This elegant identity means we never need to recompute `e^(-z)` during
    backprop — we already have `σ(z) = a`, so `σ'(z) = a(1-a)`.

### Step 3: Gradient of L w.r.t. z2 (Layer 2 pre-activation)

By chain rule:
```
∂L/∂z2 = ∂L/∂a2 · ∂a2/∂z2
        = (a2 - y)/(a2(1-a2)) · a2(1-a2)
        = a2 - y
```

This is a beautiful result: the gradient of BCE loss combined with sigmoid
activation is simply `(prediction - target)`. The ugly terms cancel.

Define `δ2 = ∂L/∂z2 = a2 - y`.   Shape: `(n_2,) = (1,)`

### Step 4: Gradients w.r.t. W2 and b2

The linear layer `z2 = W2·a1 + b2`:

```
∂L/∂W2 = δ2 · a1^T         outer product
∂L/∂b2 = δ2
```

**Derivation for W2:**
```
z2[k] = Σ_j W2[k,j] · a1[j] + b2[k]

∂z2[k]/∂W2[k,j] = a1[j]

∂L/∂W2[k,j] = ∂L/∂z2[k] · ∂z2[k]/∂W2[k,j]
             = δ2[k] · a1[j]
```

In matrix form: `∂L/∂W2 = δ2 ⊗ a1^T`   shape: `(n_2, n_1) = (1, 3)`

**Derivation for b2:**
```
∂z2[k]/∂b2[k] = 1
∂L/∂b2 = δ2                              shape: (n_2,) = (1,)
```

### Step 5: Backpropagate to Layer 1

Now we need to pass the error signal back through W2 to layer 1.

```
∂L/∂a1 = W2^T · δ2        shape: (n_1,) = (3,)
```

**Derivation:**
```
z2[k] = Σ_j W2[k,j] · a1[j]

∂z2[k]/∂a1[j] = W2[k,j]

∂L/∂a1[j] = Σ_k ∂L/∂z2[k] · ∂z2[k]/∂a1[j]
           = Σ_k δ2[k] · W2[k,j]
           = (W2^T · δ2)[j]
```

### Step 6: Gradient of L w.r.t. z1

Apply the sigmoid derivative at layer 1:
```
δ1 = ∂L/∂z1 = ∂L/∂a1 ⊙ σ'(z1)
             = (W2^T · δ2) ⊙ a1·(1-a1)
```

where `⊙` is element-wise multiplication (Hadamard product).

Shape: `(n_1,) = (3,)`.

### Step 7: Gradients w.r.t. W1 and b1

Same pattern as W2/b2:
```
∂L/∂W1 = δ1 · x^T         shape: (n_1, n_0) = (3, 2)
∂L/∂b1 = δ1               shape: (n_1,) = (3,)
```

### Complete Gradient Summary

```
Forward:
  z1 = W1·x + b1
  a1 = σ(z1)
  z2 = W2·a1 + b2
  a2 = σ(z2)
  L  = BCE(a2, y)

Backward:
  δ2 = a2 - y                          ← clean: BCE + sigmoid simplification
  
  ∂L/∂W2 = δ2 ⊗ a1^T
  ∂L/∂b2 = δ2
  
  δ1 = (W2^T · δ2) ⊙ a1·(1-a1)       ← W2 transposes the error; sigmoid gates it
  
  ∂L/∂W1 = δ1 ⊗ x^T
  ∂L/∂b1 = δ1
```

---

## Numerical Verification

```python
import numpy as np

# ── Network setup ─────────────────────────────────────────────────────────
np.random.seed(0)

n0, n1, n2 = 2, 3, 1

# Random weights
W1 = np.random.randn(n1, n0) * 0.5   # (3, 2)
b1 = np.zeros(n1)                      # (3,)
W2 = np.random.randn(n2, n1) * 0.5   # (1, 3)
b2 = np.zeros(n2)                      # (1,)

# Single training example
x = np.array([0.5, -0.3])    # (2,)
y = np.array([1.0])           # (1,) — true label

# ── Activation functions ──────────────────────────────────────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def bce_loss(a, y):
    eps = 1e-8  # numerical stability
    return -np.mean(y * np.log(a + eps) + (1 - y) * np.log(1 - a + eps))

# ── Forward pass ──────────────────────────────────────────────────────────
z1 = W1 @ x + b1      # (3,)
a1 = sigmoid(z1)       # (3,)
z2 = W2 @ a1 + b2     # (1,)
a2 = sigmoid(z2)       # (1,)
L  = bce_loss(a2, y)

print("=== Forward pass ===")
print(f"z1: {z1.round(4)}")
print(f"a1: {a1.round(4)}")
print(f"z2: {z2.round(4)}")
print(f"a2: {a2.round(4)}")
print(f"L:  {L:.6f}")

# ── Analytical backward pass (our derivation) ─────────────────────────────
delta2 = a2 - y                                    # (1,)

dW2 = np.outer(delta2, a1)                         # (1, 3)
db2 = delta2.copy()                                 # (1,)

delta1 = (W2.T @ delta2) * (a1 * (1 - a1))        # (3,)

dW1 = np.outer(delta1, x)                          # (3, 2)
db1 = delta1.copy()                                 # (3,)

print("\n=== Analytical gradients ===")
print(f"dW2: {dW2.round(6)}")
print(f"db2: {db2.round(6)}")
print(f"dW1:\n{dW1.round(6)}")
print(f"db1: {db1.round(6)}")

# ── Numerical gradient check (finite differences) ─────────────────────────
def numerical_grad(param, eps=1e-5):
    """Compute gradient of L w.r.t. param using finite differences."""
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old_val = param[idx]

        param[idx] = old_val + eps
        z1_ = W1 @ x + b1; a1_ = sigmoid(z1_)
        z2_ = W2 @ a1_ + b2; a2_ = sigmoid(z2_)
        L_plus = bce_loss(a2_, y)

        param[idx] = old_val - eps
        z1_ = W1 @ x + b1; a1_ = sigmoid(z1_)
        z2_ = W2 @ a1_ + b2; a2_ = sigmoid(z2_)
        L_minus = bce_loss(a2_, y)

        grad[idx] = (L_plus - L_minus) / (2 * eps)
        param[idx] = old_val
        it.iternext()
    return grad

dW2_num = numerical_grad(W2)
db2_num = numerical_grad(b2)
dW1_num = numerical_grad(W1)
db1_num = numerical_grad(b1)

print("\n=== Gradient check (max |analytical - numerical|) ===")
print(f"dW2 error: {np.max(np.abs(dW2 - dW2_num)):.2e}")
print(f"db2 error: {np.max(np.abs(db2 - db2_num)):.2e}")
print(f"dW1 error: {np.max(np.abs(dW1 - dW1_num)):.2e}")
print(f"db1 error: {np.max(np.abs(db1 - db1_num)):.2e}")
# All errors should be < 1e-8 if the derivation is correct
```

**Expected output**: all gradient errors < 1e-8, confirming our derivation.

---

## Why Vanishing Gradients Happen

Look at the gradient for layer 1:
```
δ1 = (W2^T · δ2) ⊙ a1·(1-a1)
```

The term `a1·(1-a1)` is the sigmoid derivative. Since `σ(z) ∈ (0, 1)`,
the maximum of `σ(z)·(1-σ(z))` is **0.25** (achieved at z=0).

For a network with `L` layers, the gradient at layer 1 includes L-1 such terms:

```
δ1 ≈ (...) × 0.25^{L-1} × (...)
```

With L=10 layers: 0.25^9 ≈ 3.8 × 10^{-6}. The gradient shrinks by a factor of
~4 at every sigmoid layer as we backpropagate deeper.

**Consequences:**
- Early layers learn very slowly (their gradients are tiny)
- Deep networks with sigmoid activations effectively only train the last few layers
- Weights in early layers barely move from their initialization

**Solutions engineers use:**

| Problem | Solution | How it helps |
|---------|----------|-------------|
| Sigmoid saturation | ReLU (`max(0, z)`) | Derivative is exactly 1 for z>0 — no shrinkage |
| Residual vanishing | Skip connections | Gradient can flow through the skip path unchanged |
| Initialization scale | Xavier/He init | Sets initial weight scale to keep gradients ∼1 |
| Normalization | BatchNorm, LayerNorm | Re-centers/rescales activations to prevent saturation |

**ReLU derivative:**
```
σ(z) = max(0, z)
σ'(z) = 1 if z > 0, else 0
```

With ReLU, the gradient flowing through an active neuron is multiplied by 1 —
no shrinkage. This is why replacing sigmoid with ReLU was a major empirical
breakthrough in deep learning (Glorot & Bengio, 2011).

!!! warning "Exploding gradients"
    The opposite problem: if weight matrices have large singular values,
    `W2^T · δ2` can amplify the gradient at each layer, causing updates that
    overflow. Solutions: gradient clipping (`max_norm`), careful initialization,
    BatchNorm.

---

## Batch Gradient Descent

So far we've computed gradients for a single example `(x, y)`. In practice,
we compute over a batch of `m` examples and average:

```
∂L/∂W2 = (1/m) × δ2_batch · a1_batch^T

where:
  δ2_batch: (m, n_2)   — all error signals
  a1_batch: (m, n_1)   — all layer 1 activations
```

```python
def forward_batch(X_batch, y_batch, W1, b1, W2, b2):
    """X_batch: (m, n0), y_batch: (m, 1)"""
    Z1 = X_batch @ W1.T + b1    # (m, n1)
    A1 = sigmoid(Z1)              # (m, n1)
    Z2 = A1 @ W2.T + b2          # (m, n2)
    A2 = sigmoid(Z2)              # (m, n2)
    L  = bce_loss(A2, y_batch)
    return Z1, A1, Z2, A2, L

def backward_batch(X_batch, y_batch, A1, A2, W2):
    m = X_batch.shape[0]
    delta2 = A2 - y_batch                              # (m, n2)
    dW2 = (delta2.T @ A1) / m                         # (n2, n1)
    db2 = delta2.mean(axis=0)                          # (n2,)
    delta1 = (delta2 @ W2) * (A1 * (1 - A1))          # (m, n1)
    dW1 = (delta1.T @ X_batch) / m                    # (n1, n0)
    db1 = delta1.mean(axis=0)                          # (n1,)
    return dW1, db1, dW2, db2
```

Note: dividing by `m` converts the total gradient to the mean gradient across
the batch. This controls the magnitude of gradient updates independently of batch size.

---

## Key Takeaways

- **Backpropagation is the chain rule**, systematically applied from loss to parameters.
- The error signal `δ_l = ∂L/∂z_l` propagates backward via `W^T` (transpose)
  and is gated by the local activation derivative.
- **BCE loss + sigmoid output** simplifies to `δ2 = a2 - y` — a beautiful
  cancellation worth understanding.
- **Vanishing gradients** arise because sigmoid's derivative is bounded by 0.25,
  compounding exponentially over many layers.
- **ReLU** (and its variants: Leaky ReLU, GELU, SiLU) solved vanishing gradients
  by having derivative = 1 for positive inputs.
- **Gradient checking** (finite differences) is the ground truth for validating
  backprop implementations — always run it when implementing from scratch.
- Modern frameworks (PyTorch, JAX) compute these exact gradients via automatic
  differentiation; understanding the derivation tells you *why* the autograd
  graph is built the way it is.

---

## Further Reading

- [Yes You Should Understand Backprop](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b) — Andrej Karpathy
- [micrograd](https://github.com/karpathy/micrograd) — 50-line autograd engine you can read in one sitting
- [The Matrix Calculus You Need For Deep Learning](https://explained.ai/matrix-calculus/) — Parr & Howard, comprehensive reference
- [Understanding the Difficulty of Training Deep Feedforward Networks](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) — Glorot & Bengio (2010), vanishing gradients and Xavier initialization

← [Deep Dives Hub](index.md)
