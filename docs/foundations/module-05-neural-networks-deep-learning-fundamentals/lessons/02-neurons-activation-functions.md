---
title: Neurons, Activation Functions, and Forward Propagation
description: >-
  Understand exactly what activation functions do mathematically, when to use
  each one and why, trace a complete forward pass with numerical values, and
  understand what happens if you remove non-linearity
duration: 75 min
difficulty: beginner
has_code: true
module: module-05
---
# Neurons, Activation Functions, and Forward Propagation

## Prerequisites

- [Lesson 01: Introduction to Neural Networks](01-introduction-to-neural-networks.md) — neuron structure, layers, forward pass overview
- [Module 00 Lesson 01: Prerequisites](../../module-00-genai-foundations-from-nlp-to-transformers/lessons/01-prerequisites.md) — matrix multiplication

## What You'll Learn

| Objective | Why It Matters |
|-----------|---------------|
| Understand why activation functions are mathematically necessary | Without this, stacking layers is pointless |
| Know the properties of each major activation function | Choosing the wrong one causes training failure |
| Choose the right activation for each layer type | A single wrong choice at the output layer breaks the whole model |
| Trace a complete forward pass with actual numbers | Required for understanding backpropagation |
| Understand dead neurons and gradient saturation | The most common activation-related training failure |

---

## Why Activation Functions Are Necessary — The Core Argument

Without activation functions, no matter how many layers you stack, the network computes a linear transformation. Stacking linear layers collapses to a single linear layer:

\[
\mathbf{h}_1 = X W_1 + b_1
\]
\[
\mathbf{h}_2 = \mathbf{h}_1 W_2 + b_2 = (XW_1 + b_1)W_2 + b_2 = X(W_1 W_2) + (b_1 W_2 + b_2)
\]

The composition is still just `X @ W_combined + b_combined`. Three layers without activation = one layer.

```python
import numpy as np

def demonstrate_linear_collapse():
    """
    Show that stacking linear layers without activation functions
    is equivalent to a single linear layer.
    """
    np.random.seed(42)
    X   = np.random.randn(5, 3)   # 5 examples, 3 features
    W1  = np.random.randn(3, 4)
    b1  = np.random.randn(4)
    W2  = np.random.randn(4, 2)
    b2  = np.random.randn(2)
    W3  = np.random.randn(2, 1)
    b3  = np.random.randn(1)

    # Three-layer linear network (no activation)
    h1      = X @ W1 + b1
    h2      = h1 @ W2 + b2
    output3 = h2 @ W3 + b3    # shape (5, 1)

    # Equivalent single-layer operation:
    W_combined = W1 @ W2 @ W3
    b_combined = b1 @ W2 @ W3 + b2 @ W3 + b3
    output1    = X @ W_combined + b_combined

    # They are identical:
    print("Three layers equal single layer?",
          np.allclose(output3, output1, atol=1e-10))   # True

demonstrate_linear_collapse()
```

**The conclusion**: to model non-linear relationships (which are required for almost every real-world task), you must introduce non-linearity via activation functions. The activation makes the composition of layers genuinely richer than any single linear layer.

---

## The Major Activation Functions

### 1. ReLU (Rectified Linear Unit) — The Modern Default

\[
\text{ReLU}(z) = \max(0, z)
\]

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

def relu_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivative: 0 for z < 0, 1 for z > 0, undefined (choose 0) at z=0.
    This is the 'subgradient' — used in training.
    """
    return (z > 0).astype(float)

z_test = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
print("z:           ", z_test)
print("ReLU(z):     ", relu(z_test))          # [0.0, 0.0, 0.0, 1.0, 3.0]
print("ReLU'(z):    ", relu_derivative(z_test))  # [0.0, 0.0, 0.0, 1.0, 1.0]
```

**Why ReLU is dominant:**
- **Gradient does not vanish** for positive values — the derivative is exactly 1, so gradients flow unchanged through active neurons
- **Computationally cheap** — just a comparison, no exponentials
- **Sparse activation** — roughly half the neurons output 0, which reduces effective model size
- **Empirically superior** in practice on most tasks (established by AlexNet 2012 and subsequent research)

**The dead neuron problem:**

```python
def demonstrate_dead_neurons():
    """
    Show how neurons can permanently deactivate ('die').
    Once z < 0 for all inputs, the gradient is 0 everywhere →
    the weights never get updated → the neuron stays dead forever.
    """
    np.random.seed(7)

    # A neuron with these weights
    w = np.array([-2.0, -3.0, -1.0])   # all strongly negative
    b = -1.0

    # For any reasonable input, z = w^T x + b will be very negative
    X = np.random.randn(100, 3)   # 100 examples
    z = X @ w + b

    print(f"Fraction of inputs where neuron is active: {(z > 0).mean():.1%}")
    # → 0.0% — this neuron is completely dead

    print(f"Gradient of loss w.r.t. w: {relu_derivative(z).mean():.1f}")
    # → 0.0 — no gradient flows through, weights never update

demonstrate_dead_neurons()
```

Dead neurons are caused by large learning rates, poor initialization, or large negative biases. Solutions: Leaky ReLU, lower learning rate, careful weight initialization.

---

### 2. Leaky ReLU — Fixing Dead Neurons

\[
\text{LeakyReLU}(z) = \begin{cases} z & z > 0 \\ \alpha z & z \leq 0 \end{cases}
\]

where \(\alpha\) is typically 0.01:

```python
def leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    return np.where(z > 0, 1.0, alpha)

z_test = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
print("z:              ", z_test)
print("LeakyReLU:      ", leaky_relu(z_test))           # [-0.05, -0.01, 0.0, 1.0, 5.0]
print("LeakyReLU':     ", leaky_relu_derivative(z_test)) # [0.01, 0.01, 0.01, 1.0, 1.0]
# Key: gradient is 0.01 (not 0) for z < 0 → neuron cannot fully die
```

---

### 3. Sigmoid — The Probability Activation

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

```python
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # clip for numerical stability

def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    """
    Key identity: σ'(z) = σ(z)(1 - σ(z))
    Maximum value: 0.25 (at z=0)
    Vanishes exponentially as |z| → ∞
    """
    s = sigmoid(z)
    return s * (1 - s)

z_test = np.array([-10.0, -5.0, 0.0, 5.0, 10.0])
print("z:         ", z_test)
print("σ(z):      ", sigmoid(z_test).round(4))           # [0.0, 0.0067, 0.5, 0.9933, 1.0]
print("σ'(z):     ", sigmoid_derivative(z_test).round(4)) # [0.0, 0.0066, 0.25, 0.0066, 0.0]
# Note: gradient vanishes for |z| > 5 — gradient saturation!
```

**Gradient saturation problem:**

```python
def show_sigmoid_saturation():
    """
    When z is large, σ'(z) ≈ 0.
    If a neuron's output saturates (near 0 or 1), gradients effectively stop flowing.
    """
    z_values = np.array([-20, -10, -5, -2, 0, 2, 5, 10, 20])
    grads    = sigmoid_derivative(z_values)

    for z, g in zip(z_values, grads):
        print(f"z={z:5.0f}: σ'(z) = {g:.6f}  {'← near ZERO!' if abs(z) > 5 else ''}")

show_sigmoid_saturation()
# z = -20: σ'(z) = 0.000000  ← near ZERO!
# z =  -5: σ'(z) = 0.006648  ← very small
# z =   0: σ'(z) = 0.250000  ← maximum gradient
# z =   5: σ'(z) = 0.006648  ← very small
```

**When to use sigmoid**: output layer for binary classification only. It squashes to [0,1], making it directly interpretable as a probability.

---

### 4. Tanh — Zero-Centered Sigmoid

\[
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} = 2\sigma(2z) - 1
\]

```python
def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)

def tanh_derivative(z: np.ndarray) -> np.ndarray:
    """σ'(z) = 1 - tanh²(z)"""
    return 1 - np.tanh(z)**2

z_test = np.array([-5.0, -2.0, 0.0, 2.0, 5.0])
print("z:       ", z_test)
print("tanh(z): ", tanh(z_test).round(4))         # [-1.0, -0.964, 0.0, 0.964, 1.0]
print("tanh'(z):", tanh_derivative(z_test).round(4)) # [0.0, 0.071, 1.0, 0.071, 0.0]
```

**Advantages over sigmoid**: zero-centered (output range [-1, 1] vs [0, 1] for sigmoid). Zero-centered outputs mean gradients in the next layer can be both positive and negative, accelerating convergence. Still suffers gradient saturation for |z| > 3.

**When to use tanh**: RNNs and LSTMs (the original papers used tanh for hidden states); sometimes hidden layers in small networks.

---

### 5. Softmax — Multi-Class Output

\[
\text{softmax}(z_k) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}
\]

Covered in Module 00 Lesson 01. Key properties: maps logits to probabilities (all positive, sum to 1). Always used for multi-class classification output layers, never in hidden layers.

```python
def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    exp_z = np.exp(z - z.max())
    return exp_z / exp_z.sum()

logits = np.array([3.2, 1.5, 0.8, -0.2])   # raw scores for 4 classes
probs  = softmax(logits)
print("Logits:       ", logits)
print("Probabilities:", probs.round(4))      # sums to 1.0
print("Sum:          ", probs.sum())
```

### 6. SwiGLU (Modern LLMs)

Modern large language models (LLaMA, Mistral, Gemma) use SwiGLU in their feed-forward layers:

\[
\text{SwiGLU}(z, W_1, W_2) = \text{Swish}(z W_1) \odot (z W_2)
\]

where \(\text{Swish}(x) = x \cdot \sigma(x)\) and \(\odot\) is element-wise multiplication. Empirically 1-3% better than ReLU on large models at the cost of slightly more parameters.

---

## Choosing the Right Activation Function

| Layer Type | Task | Recommended Activation |
|-----------|------|------------------------|
| Hidden layers | Most tasks | **ReLU** (default) |
| Hidden layers | If dead neuron issues | **Leaky ReLU** or ELU |
| Hidden layers (LLMs) | Modern large models | **SwiGLU** |
| Output layer | Binary classification | **Sigmoid** |
| Output layer | Multi-class classification | **Softmax** |
| Output layer | Regression (any value) | **None** (linear) |
| Output layer | Bounded regression (0-1) | **Sigmoid** |
| RNN/LSTM internal gates | Sequence modeling | **Sigmoid** (gates), **Tanh** (cell) |

!!! warning "Most Common Mistake: Wrong Output Activation"
    Using ReLU on the output layer for binary classification will make predictions of "0" common but never produce values > 0 for negative logits — a fundamentally broken model. Always match output activation to your loss function and interpretation.

---

## Complete Forward Pass — Numerical Trace

Let us trace a complete forward pass through a 2-hidden-layer network for binary classification:

**Architecture**: 3 inputs → 4 hidden → 3 hidden → 1 output (sigmoid)

```python
import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-z))
def relu(z):    return np.maximum(0, z)

# Fixed (pretrained) weights — normally these come from training
np.random.seed(42)
W1 = np.array([[0.5, -0.1, 0.3, 0.8],
               [0.2,  0.7, -0.4, 0.1],
               [-0.3, 0.4,  0.6, -0.5]])   # shape (3, 4)
b1 = np.array([0.1, -0.2, 0.3, 0.1])       # shape (4,)

W2 = np.array([[0.6, 0.2, -0.3],
               [-0.1, 0.8, 0.4],
               [0.3, -0.5, 0.7],
               [0.2,  0.1, -0.6]])          # shape (4, 3)
b2 = np.array([0.2, 0.1, -0.1])            # shape (3,)

W3 = np.array([[0.7], [-0.4], [0.5]])       # shape (3, 1)
b3 = np.array([0.1])                        # shape (1,)

# Input: one example with 3 features
x = np.array([1.0, -0.5, 0.8])             # shape (3,)

print("=== Forward Pass Trace ===\n")

# Layer 1
z1 = W1.T @ x + b1   # (3,4).T = (4,3) @ (3,) = (4,) + (4,) = (4,)
                      # Wait: W1 is (3,4), so W1.T is (4,3)
                      # Actually: z1 = x @ W1 + b1 = (3,) @ (3,4) → (4,)
z1 = x @ W1 + b1
a1 = relu(z1)

print(f"Input x:      {x}")
print(f"z1 = x@W1+b1: {z1.round(3)}")      # pre-activation
print(f"a1 = relu(z1):{a1.round(3)}")      # post-activation

# Layer 2
z2 = a1 @ W2 + b2   # (4,) @ (4,3) → (3,)
a2 = relu(z2)

print(f"\nz2 = a1@W2+b2:{z2.round(3)}")
print(f"a2 = relu(z2): {a2.round(3)}")

# Layer 3 (output)
z3 = a2 @ W3 + b3   # (3,) @ (3,1) → (1,)
a3 = sigmoid(z3)

print(f"\nz3 = a2@W3+b3:{z3.round(3)}")
print(f"a3 = sigmoid(z3): {a3.round(4)}")
print(f"\nPrediction: {a3[0]:.4f} → {'Positive class' if a3[0] > 0.5 else 'Negative class'}")

# Shape summary
print("\n=== Shape Summary ===")
for name, tensor in [("x", x), ("z1", z1), ("a1", a1), ("z2", z2), ("a2", a2), ("z3", z3), ("a3", a3)]:
    print(f"{name:3s}: {tensor.shape}")
```

---

## Why Non-Linearity Enables Complex Boundaries

The activation function at each layer enables the network to carve non-linear decision boundaries:

```python
import numpy as np

def visualize_decision_boundary_intuition():
    """
    Show conceptually why linear layers cannot separate XOR,
    but a network with a hidden layer can.
    """
    # XOR problem: not linearly separable
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])   # XOR: 1 iff inputs differ

    # No single hyperplane can separate these
    print("XOR cannot be learned by a single neuron (linear decision boundary).")
    print("With a hidden layer (non-linear activation), it CAN be learned.")

    # A network that solves XOR (these are the ground truth weights):
    W1 = np.array([[1, 1], [1, 1]])   # (2, 2) — two hidden neurons
    b1 = np.array([0.0, -1.0])
    W2 = np.array([[1], [-2]])         # (2, 1) — output neuron
    b2 = np.array([0.0])

    def step(z): return (z > 0).astype(float)   # simplified activation

    for x, label in zip(X, y):
        h = step(x @ W1 + b1)
        pred = step(h @ W2 + b2)[0]
        print(f"  Input {x}: prediction={pred:.0f}, label={label}  {'✓' if pred == label else '✗'}")

visualize_decision_boundary_intuition()
```

---

## Edge Cases and Misconceptions

**"ReLU has no gradient at z=0."** Technically correct — ReLU is not differentiable at exactly 0. In practice, this is handled by choosing the subgradient (typically 0 or 1 at z=0). This occurs so rarely during training (continuous random weights rarely land exactly at 0) that it has no practical impact.

**"Sigmoid is deprecated."** Sigmoid is deprecated *in hidden layers*. It remains the correct choice for binary classification output layers. Using ReLU at the output for binary classification is a common mistake.

**"Leaky ReLU always outperforms ReLU."** Not empirically true. ReLU works well for the vast majority of architectures. The benefit of Leaky ReLU matters when many neurons are dying during training — which you can detect by monitoring activation distributions.

**"More complex activations are always better."** SwiGLU and GELU outperform ReLU on large-scale LLMs but the difference is marginal (1-2%) and they are computationally more expensive. For small models, ReLU is the right default.

---

## Key Takeaways

- Without activation functions, stacking linear layers is equivalent to a single linear layer — depth adds no expressivity
- **ReLU** is the default for hidden layers: cheap, no vanishing gradient for positive values, sparsity
- **Dead neurons** occur when z ≤ 0 for all inputs — use Leaky ReLU or lower learning rate if this happens
- **Sigmoid** saturates for |z| > 5, killing gradients — use only at the output layer for binary classification
- **Softmax** is the output activation for multi-class classification — converts logits to a probability distribution
- Match output activation to your task: sigmoid (binary), softmax (multi-class), linear (regression)

---

## Further Reading

- [CS231n Notes: Neural Network Architecture](https://cs231n.github.io/neural-networks-1/) — Stanford course notes with detailed activation function comparison
- [3Blue1Brown: Neural Networks Part 1](https://www.youtube.com/watch?v=aircAruvnKk) — visual walkthrough of neurons and layers
- [Maas et al. (2013): Rectifier Nonlinearities Improve Neural Network Acoustic Models](https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf) — original Leaky ReLU paper

---

**Next:** [Loss Functions and Measuring Performance](03-loss-functions.md)
