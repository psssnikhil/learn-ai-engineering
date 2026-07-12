---
title: Introduction to Neural Networks and Deep Learning
description: >-
  Build a precise mental model of neural networks — from a single artificial
  neuron through layered architectures to why depth enables hierarchical feature
  learning, with worked numerical examples and connections to the Transformers you
  have already studied
duration: 60 min
difficulty: beginner
has_code: true
module: module-05
---
# Introduction to Neural Networks and Deep Learning

## Prerequisites

- [Module 00: GenAI Foundations](../../module-00-genai-foundations-from-nlp-to-transformers/lessons/01-prerequisites.md) — vectors, matrix multiplication, dot products
- [Module 00 Lesson 02: Math Foundations](../../module-00-genai-foundations-from-nlp-to-transformers/lessons/02-math-foundations.md) — loss functions, gradient descent

## What You'll Learn

| Objective | Why It Matters |
|-----------|---------------|
| Define a neuron mathematically, not just metaphorically | Enables you to read neural network code and papers |
| Explain what depth adds that a single layer cannot | The core justification for "deep" learning |
| Trace a forward pass through a 3-layer network by hand | The foundation for understanding backpropagation |
| Connect neural networks to Transformers you have already studied | Module 00 layers are neural network layers |
| Identify what neural networks cannot do | Prevents wasted effort on ill-suited problems |

---

## What a Neural Network Is — Precisely

A neural network is a **parameterized function**: it takes an input, applies a sequence of mathematical operations controlled by numbers called weights and biases, and produces an output. Training finds the weights that make this function most useful for a task.

This is the entire definition. Every architectural variation — CNNs, RNNs, Transformers, diffusion models — is a specific choice of operations and connectivity.

The biological metaphor (neurons, synapses, firing) is a historical artifact that inspired the naming but should not be over-interpreted. Artificial neurons compute differently from biological ones, and the comparison can mislead more than it clarifies.

---

## The Artificial Neuron

An artificial neuron computes a weighted sum of its inputs, adds a bias, and applies a non-linear function called an activation function:

\[
y = f\!\left(\sum_{i=1}^{n} w_i x_i + b\right) = f(\mathbf{w}^\top \mathbf{x} + b)
\]

Where:
- \(\mathbf{x} = [x_1, x_2, \ldots, x_n]\) — input features
- \(\mathbf{w} = [w_1, w_2, \ldots, w_n]\) — learned weights (one per input)
- \(b\) — learned bias (a free scalar offset)
- \(f\) — activation function (e.g., ReLU, sigmoid, tanh)
- \(y\) — the neuron's output

### Worked Numerical Example: Spam Detection

Three features; one neuron; sigmoid activation:

```python
import numpy as np

def sigmoid(z: float) -> float:
    """Maps any real number to (0, 1) — suitable for probabilities."""
    return 1 / (1 + np.exp(-z))

# Features for an email:
# x1 = number of exclamation marks (normalized 0–1)
# x2 = contains "urgent"? (binary: 0 or 1)
# x3 = sender is in contacts? (binary: 0 or 1, high = not spam signal)

x = np.array([0.80, 1.0, 0.0])   # lots of !, "urgent" present, unknown sender

# Learned weights (what we find AFTER training)
w = np.array([ 0.70,   # many ! → likely spam (positive weight)
               0.50,   # "urgent" → suspicious (positive weight)
              -0.90])  # known sender → NOT spam (negative weight)
b = 0.10               # bias

# Step 1: weighted sum
z = np.dot(w, x) + b
print(f"z = {0.70}×{0.80} + {0.50}×{1.0} + (-{0.90})×{0.0} + {0.10}")
print(f"z = {0.70*0.80:.2f} + {0.50:.2f} + {0.00:.2f} + {0.10:.2f}")
print(f"z = {z:.4f}")

# Step 2: activation function
y = sigmoid(z)
print(f"y = sigmoid({z:.4f}) = {y:.4f}")
print(f"Interpretation: {y*100:.1f}% probability of spam")
# z ≈ 1.16, y ≈ 0.76 → 76% probability of spam
```

The weights encode what the network has learned: "exclamation marks and urgent language are spam signals; known senders are not." These weights come from training on labeled examples — we do not design them.

---

## From One Neuron to a Layer

A **layer** is a set of neurons that all receive the same inputs and compute in parallel. With n_in inputs and n_out neurons, the layer computes:

\[
\mathbf{h} = f(X \mathbf{W} + \mathbf{b})
\]

Where:
- \(X\): input matrix of shape `(batch_size, n_in)`
- \(\mathbf{W}\): weight matrix of shape `(n_in, n_out)` — one column per neuron
- \(\mathbf{b}\): bias vector of shape `(n_out,)`
- \(f\): applied element-wise

```python
import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    """Rectified Linear Unit: pass positive values, zero out negatives."""
    return np.maximum(0, x)

# A layer with 4 inputs and 3 neurons
np.random.seed(42)
n_in, n_out = 4, 3
batch_size  = 2

X = np.array([                         # 2 examples, 4 features each
    [0.80, 1.0, 0.0, 0.5],
    [0.20, 0.0, 1.0, 0.3],
])   # shape (2, 4)

W = np.random.randn(n_in, n_out) * 0.1  # shape (4, 3)
b = np.zeros(n_out)                      # shape (3,)

# The entire layer in one line:
Z = X @ W + b     # (2,4) @ (4,3) + (3,) → (2, 3)  (broadcasting adds b)
H = relu(Z)       # (2, 3)  element-wise

print(f"X.shape: {X.shape}")   # (2, 4)
print(f"W.shape: {W.shape}")   # (4, 3)
print(f"Z.shape: {Z.shape}")   # (2, 3)
print(f"H.shape: {H.shape}")   # (2, 3)

# Each row of H is one example's representation after this layer.
# Three numbers encode what the 3 neurons "detected" in the input.
```

This is the exact computation performed by a Transformer's feed-forward layer (`X @ W1 + b1`, apply activation, `@ W2 + b2`). You have already studied this code — it is in Lesson 07 of Module 00.

---

## From a Layer to a Deep Network

Stacking layers creates a **deep network**. The output of layer L becomes the input to layer L+1. Each layer transforms its input into a new representation:

```
Input → [Layer 1] → [Layer 2] → [Layer 3] → Output
(raw)    (simple)   (composite)  (abstract)
```

```python
import numpy as np

def relu(x): return np.maximum(0, x)
def sigmoid(x): return 1 / (1 + np.exp(-x))

class SimpleNeuralNetwork:
    """
    A 3-layer network:
    - Layer 1: n_in → hidden1
    - Layer 2: hidden1 → hidden2
    - Layer 3: hidden2 → n_out (final prediction)
    """

    def __init__(self, n_in: int, hidden1: int, hidden2: int, n_out: int,
                 seed: int = 42):
        np.random.seed(seed)
        # Small random initialization is critical — prevents symmetry breaking issues
        scale = 0.01
        self.params = {
            "W1": np.random.randn(n_in, hidden1)   * scale,
            "b1": np.zeros(hidden1),
            "W2": np.random.randn(hidden1, hidden2) * scale,
            "b2": np.zeros(hidden2),
            "W3": np.random.randn(hidden2, n_out)  * scale,
            "b3": np.zeros(n_out),
        }
        self.cache = {}   # stored for backpropagation (Lesson 05)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        X: (batch_size, n_in)
        Returns: (batch_size, n_out) — raw logits
        """
        p = self.params

        # Layer 1: linear + ReLU
        Z1 = X @ p["W1"] + p["b1"]     # (batch, hidden1)
        A1 = relu(Z1)                   # (batch, hidden1)

        # Layer 2: linear + ReLU
        Z2 = A1 @ p["W2"] + p["b2"]    # (batch, hidden2)
        A2 = relu(Z2)                   # (batch, hidden2)

        # Layer 3: linear only (no activation for raw logits)
        Z3 = A2 @ p["W3"] + p["b3"]    # (batch, n_out)

        # Cache intermediates — needed for backpropagation
        self.cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2, "Z3": Z3}

        return Z3   # raw logits — apply softmax externally for probabilities

    def count_parameters(self) -> int:
        """Total number of learnable parameters."""
        return sum(p.size for p in self.params.values())

# Build and test
net = SimpleNeuralNetwork(n_in=784, hidden1=256, hidden2=128, n_out=10)
print(f"Total parameters: {net.count_parameters():,}")
# 784×256 + 256 + 256×128 + 128 + 128×10 + 10 = 233,994

# Forward pass
X_batch = np.random.randn(32, 784)   # 32 images, each 784 pixels (28×28)
logits = net.forward(X_batch)
print(f"Logits shape: {logits.shape}")   # (32, 10) — one score per class per image
```

---

## Why Depth Matters: What Layers Learn

Why not use a single large layer? The answer is expressivity: deep networks can represent functions that shallow networks cannot represent efficiently.

Intuition for a digit classifier:

```
Layer 1 (simple patterns): detects edges at various angles and positions
  Neuron activates for: /  \  |  _  ~  (edge detectors)

Layer 2 (compositions): combines edges into curves and junctions
  Neuron activates for: ⌒  ⌣  |  ⌐  (curve detectors)

Layer 3 (parts): combines curves into digit parts
  Neuron activates for: circle top, long vertical, curved bottom

Layer 4 (whole): combines parts into complete digits
  Neuron activates for: "8" = two circles stacked
                        "7" = horizontal line + diagonal
```

This hierarchical composition — simple → composite → abstract — is what "deep" learning provides. A single-layer network can approximate the same function in principle (Universal Approximation Theorem), but would need exponentially more neurons to do so.

!!! note "Why Does This Connect to Transformers?"
    Transformers are 96-layer deep networks. Each layer computes multi-head attention (which relationships between tokens matter?) followed by a feed-forward network (what transformation to apply to each token's representation?). The layers build progressively more abstract representations: lower layers capture syntax and local patterns; higher layers capture semantics and long-range reasoning.

---

## Neural Networks vs. Traditional Programming

Understanding this comparison clarifies when neural networks are appropriate:

```
Traditional programming:
  Developer writes rules explicitly
  if text.contains("urgent") and text.contains("!!!"):
      label = "spam"

  Good: deterministic, interpretable, no data needed
  Bad:  cannot handle patterns too complex to enumerate

Machine learning (neural networks):
  Developer provides examples with labels
  Network infers the rules during training

  Good:  handles patterns too complex for explicit rules (images, language)
  Bad:   requires data, often opaque, can fail unexpectedly on edge cases
```

### When Neural Networks Are the Right Tool

- High-dimensional input (images = millions of pixels; text = thousands of tokens)
- Pattern exists but is too complex to describe with rules
- Large labeled dataset available (or self-supervised pre-training possible)
- Approximate answers are acceptable (probabilistic, not exact)

### When They Are Not

- Very small dataset (< a few thousand examples for most tasks)
- Exact, interpretable answers required (safety-critical without validation)
- Simple, explicit rules already exist and work
- Strict physical or logical constraints must be satisfied

---

## The Universal Approximation Theorem

A foundational result: a neural network with at least one hidden layer and a non-linear activation function can approximate *any* continuous function to arbitrary accuracy, given enough neurons.

```python
import numpy as np
import matplotlib.pyplot as plt

# Demonstrate: a 2-layer ReLU network can approximate a non-linear function
def target_function(x: np.ndarray) -> np.ndarray:
    """A non-linear function: sin(x) + 0.3*x"""
    return np.sin(x) + 0.3 * x

# Simple demonstration: approximate with a linear combination of ReLUs
# ReLU(x - c) is a "ramp" starting at x=c
# Linear combinations of ramps can approximate smooth functions
x = np.linspace(-5, 5, 300)
y_true = target_function(x)

# A few ReLU basis functions (normally, these would be LEARNED)
breakpoints = np.linspace(-5, 5, 10)
approximation = np.zeros_like(x)
for bp in breakpoints:
    weight = np.random.randn() * 0.5   # random weight for illustration
    approximation += weight * np.maximum(0, x - bp)

# With training (learned weights), the approximation would fit y_true closely
print("A trained version of this network would approximate the target function closely.")
print("This is the Universal Approximation Theorem in action.")
```

The theorem does not say how many neurons are needed (could be exponential) or how to train them (the hard problem). It establishes a theoretical capability floor.

---

## The Three-Part Recipe for Deep Learning Success

Neural networks only succeed when all three components are present:

| Component | What It Provides | What Happens Without It |
|-----------|-----------------|------------------------|
| **Data** | The training signal — what patterns to learn | Networks memorize noise or fail to converge |
| **Compute** | Training time; modern ML requires GPUs/TPUs | Training takes weeks instead of hours |
| **Algorithm** | Gradient descent + backpropagation | No way to adjust weights toward better predictions |

These are independent bottlenecks. More data cannot substitute for an inadequate architecture. Faster compute cannot substitute for a poor loss function. A good algorithm cannot learn from insufficient data.

---

## Edge Cases and Misconceptions

**"Deeper is always better."** More depth requires more data, more compute, and more careful training (gradient vanishing/exploding, careful initialization). A 3-layer network with good data beats a 96-layer network trained on 100 examples.

**"Neural networks are black boxes — you cannot understand them."** Partially true. The weights are not interpretable directly, but: (1) we can visualize what activates each neuron, (2) attention weights show what input positions the model focuses on, (3) probing classifiers can measure what information is encoded at each layer. Interpretability is an active research area.

**"More neurons always improve performance."** Past a certain point, more neurons without more data leads to overfitting: the network memorizes the training examples instead of learning generalizable patterns. We will cover regularization in Lesson 06.

**"Neural networks require GPU."** GPUs dramatically accelerate training (100x–1000x), but small networks can train on CPU. The networks in this lesson run in seconds on any modern laptop.

---

## Production Connection

| Concept | Where It Appears |
|---------|-----------------|
| **Forward pass** | Called millions of times during training (batch × learning step); also the core of inference |
| **Hidden layer dimensions** | Affect memory and latency; key parameter in model deployment |
| **Activation functions** | ReLU's derivative (0 or 1) makes gradient computation cheap — important for training large models |
| **Parameter count** | Determines model download size, inference memory, and training compute |

---

## Key Takeaways

- A neural network is a parameterized function; training finds weights that minimize a loss function
- A neuron computes: weighted sum of inputs → add bias → apply activation function
- A layer is N neurons sharing the same inputs, computed in parallel via matrix multiplication
- Depth enables hierarchical feature learning: simple patterns → compositions → abstractions
- Three ingredients: data, compute, algorithm — all are necessary
- Neural networks are appropriate when the pattern is too complex for explicit rules and data is available

---

## Further Reading

- [3Blue1Brown: But What Is a Neural Network?](https://www.youtube.com/watch?v=aircAruvnKk) — the best visual introduction to the content of this lesson (19 min)
- [Michael Nielsen: Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) — a free online book with worked examples; Chapters 1-2 align with this and the next lesson
- [Jay Alammar: A Visual and Interactive Guide to the Basics of Neural Networks](https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/) — excellent interactive diagrams for forward propagation

---

**Next:** [Neurons, Activation Functions, and Forward Propagation](02-neurons-activation-functions.md)
