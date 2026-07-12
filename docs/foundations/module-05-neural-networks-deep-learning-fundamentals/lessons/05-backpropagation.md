---
title: Backpropagation — Computing Gradients Through Any Network
description: >-
  Understand backpropagation as the chain rule applied systematically to a
  computational graph, trace gradients numerically through a multi-layer
  network, and understand vanishing/exploding gradients and their solutions
duration: 90 min
difficulty: intermediate
has_code: true
module: module-05
---
# Backpropagation — Computing Gradients Through Any Network

## Prerequisites

- [Lesson 02: Neurons & Activation Functions](02-neurons-activation-functions.md)
- [Lesson 03: Loss Functions](03-loss-functions.md)
- [Lesson 04: Gradient Descent](04-gradient-descent.md)
- [Module 00 Lesson 02: Math Foundations](../../module-00-genai-foundations-from-nlp-to-transformers/lessons/02-math-foundations.md) — chain rule, partial derivatives

## What You'll Learn

| Objective | Why It Matters |
|-----------|---------------|
| Understand the chain rule applied to a computational graph | Backprop is just the chain rule — nothing more |
| Trace gradients numerically through a 3-layer network | Makes backpropagation concrete and debuggable |
| Implement forward and backward passes from scratch | The only way to truly understand what PyTorch/JAX compute |
| Explain vanishing and exploding gradients | The central training challenge in deep networks |
| Understand residual connections as a gradient engineering solution | Why ResNet, Transformer, and GPT all use them |

---

## Backpropagation is the Chain Rule

Before writing any code, establish the core idea precisely:

Backpropagation is the **chain rule for derivatives applied systematically to a computational graph**, traversed in reverse order from loss to inputs.

Nothing more. No magic.

The chain rule: if \(L = f(g(x))\), then:
\[
\frac{dL}{dx} = \frac{dL}{df} \cdot \frac{df}{dg} \cdot \frac{dg}{dx}
\]

In a neural network, each layer is a function. Composing layers is function composition. The chain rule decomposes the gradient of the loss w.r.t. any parameter into a product of local gradients along the path from that parameter to the loss.

---

## Computational Graph Perspective

Every computation can be represented as a directed acyclic graph (DAG) where:
- **Nodes** are values (scalars, vectors, matrices)
- **Edges** are operations (addition, multiplication, activation functions)

```
x → [multiply w] → z → [ReLU] → a → [multiply w2] → y_hat → [MSE with y] → Loss
```

The **forward pass** computes values left to right. The **backward pass** computes gradients right to left, using the chain rule at each node.

```python
import numpy as np

class Value:
    """
    A minimal autograd scalar value — inspired by micrograd (Andrej Karpathy).
    Stores value, gradient, and how to backpropagate through the operation.
    """

    def __init__(self, data: float, _children=(), _op='', label=''):
        self.data  = float(data)
        self.grad  = 0.0             # dL/d(this value), starts at 0
        self._backward = lambda: None  # function to backpropagate
        self._prev = set(_children)
        self._op   = _op
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1
            # Chain rule: dL/da += dL/dout * 1
            self.grad  += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'ReLU')

        def _backward():
            # dReLU/dx = 1 if x > 0 else 0
            self.grad += (self.data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """
        Topological sort then backpropagate from loss (this node, grad=1.0).
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0   # dL/dL = 1
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"


def trace_simple_network():
    """
    Trace forward and backward through: y = relu(w1*x + b1)*w2 + b2
    Loss = (y - target)^2
    """
    # Parameters
    w1 = Value(0.5,  label='w1')
    b1 = Value(0.1,  label='b1')
    w2 = Value(0.8,  label='w2')
    b2 = Value(0.2,  label='b2')

    # Input and target
    x      = Value(2.0, label='x')
    target = Value(3.0, label='target')

    print("=== Forward Pass ===")
    z1   = w1 * x + b1          # pre-activation
    a1   = z1.relu()            # ReLU
    y    = a1 * w2 + b2         # output
    diff = y + Value(-target.data)  # y - target (subtract)
    loss = diff * diff          # (y - target)^2

    print(f"z1  = w1*x + b1 = {w1.data}*{x.data} + {b1.data} = {z1.data:.4f}")
    print(f"a1  = ReLU(z1)  = {a1.data:.4f}")
    print(f"y   = a1*w2 + b2= {y.data:.4f}")
    print(f"loss= (y-target)^2 = ({y.data:.4f} - {target.data})^2 = {loss.data:.4f}")

    print("\n=== Backward Pass ===")
    loss.backward()

    print(f"∂L/∂w1 = {w1.grad:.4f}  (gradient for first-layer weight)")
    print(f"∂L/∂b1 = {b1.grad:.4f}  (gradient for first-layer bias)")
    print(f"∂L/∂w2 = {w2.grad:.4f}  (gradient for second-layer weight)")
    print(f"∂L/∂b2 = {b2.grad:.4f}  (gradient for output bias)")

trace_simple_network()
```

---

## Full Numerical Trace: 3-Layer Network

A complete trace using pure NumPy — the exact computation PyTorch performs under the hood (for simple networks):

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)

def full_backprop_trace():
    """
    3-layer network: 3 inputs → 4 hidden (ReLU) → 3 hidden (ReLU) → 1 output (sigmoid)
    Loss: binary cross-entropy
    Trace all gradients numerically.
    """
    # Fixed weights (pretrained for illustration)
    W1 = np.array([[0.5, -0.1, 0.3, 0.8],
                   [0.2,  0.7, -0.4, 0.1],
                   [-0.3, 0.4,  0.6, -0.5]])   # (3, 4)
    b1 = np.array([0.1, -0.2, 0.3, 0.1])       # (4,)

    W2 = np.array([[0.6, 0.2, -0.3],
                   [-0.1, 0.8, 0.4],
                   [0.3, -0.5, 0.7],
                   [0.2,  0.1, -0.6]])          # (4, 3)
    b2 = np.array([0.2, 0.1, -0.1])            # (3,)

    W3 = np.array([[0.7], [-0.4], [0.5]])       # (3, 1)
    b3 = np.array([0.1])                        # (1,)

    x = np.array([1.0, -0.5, 0.8])             # (3,)
    y = np.array([1.0])                         # positive class

    lr = 0.01

    print("=== FORWARD PASS ===")
    # Layer 1
    z1 = x @ W1 + b1       # (3,)@(3,4) → (4,)
    a1 = relu(z1)           # (4,)
    print(f"z1 shape: {z1.shape}  {z1.round(3)}")
    print(f"a1 shape: {a1.shape}  {a1.round(3)}")

    # Layer 2
    z2 = a1 @ W2 + b2      # (4,)@(4,3) → (3,)
    a2 = relu(z2)           # (3,)
    print(f"z2 shape: {z2.shape}  {z2.round(3)}")
    print(f"a2 shape: {a2.shape}  {a2.round(3)}")

    # Output layer
    z3 = a2 @ W3 + b3      # (3,)@(3,1) → (1,)
    a3 = sigmoid(z3)        # (1,)  — probability
    print(f"z3 shape: {z3.shape}  {z3.round(4)}")
    print(f"a3 (output): {a3.round(4)}")

    # BCE loss
    a3_clipped = np.clip(a3, 1e-7, 1 - 1e-7)
    loss = -np.mean(y * np.log(a3_clipped) + (1 - y) * np.log(1 - a3_clipped))
    print(f"\nBCE Loss = {loss:.6f}")

    print("\n=== BACKWARD PASS ===")
    # d(BCE)/d(a3): gradient of loss w.r.t. sigmoid output
    # For BCE + sigmoid: dL/dz3 = a3 - y  (the beautiful simplification)
    dL_dz3 = a3 - y         # (1,) — prediction error at output

    # Gradients for output layer
    dL_dW3 = a2[:, None] @ dL_dz3[None, :]   # (3,1) = outer product of (3,)×(1,)
    dL_db3 = dL_dz3                            # (1,)

    print(f"dL/dz3 = a3 - y = {dL_dz3.round(4)}  (shape: {dL_dz3.shape})")
    print(f"dL/dW3 shape: {dL_dW3.shape}  values: {dL_dW3.flatten().round(4)}")
    print(f"dL/db3: {dL_db3.round(4)}")

    # Backprop through layer 2
    dL_da2 = dL_dz3 @ W3.T  # (1,)@(1,3) → (3,)
    dL_dz2 = dL_da2 * relu_grad(z2)   # element-wise with ReLU derivative

    dL_dW2 = a1[:, None] @ dL_dz2[None, :]   # (4,3)
    dL_db2 = dL_dz2                           # (3,)

    print(f"\ndL/da2 = dL/dz3 @ W3.T: {dL_da2.round(4)}  (shape: {dL_da2.shape})")
    print(f"relu'(z2): {relu_grad(z2).round(0).astype(int)}  (which neurons are active)")
    print(f"dL/dz2 = dL/da2 * relu'(z2): {dL_dz2.round(4)}")
    print(f"dL/dW2 shape: {dL_dW2.shape}")

    # Backprop through layer 1
    dL_da1 = dL_dz2 @ W2.T   # (3,)@(3,4) → (4,)
    dL_dz1 = dL_da1 * relu_grad(z1)

    dL_dW1 = x[:, None] @ dL_dz1[None, :]   # (3,4)
    dL_db1 = dL_dz1                          # (4,)

    print(f"\ndL/da1 = dL/dz2 @ W2.T: {dL_da1.round(4)}")
    print(f"relu'(z1): {relu_grad(z1).round(0).astype(int)}")
    print(f"dL/dz1 = dL/da1 * relu'(z1): {dL_dz1.round(4)}")
    print(f"dL/dW1 shape: {dL_dW1.shape}")

    print("\n=== GRADIENT DESCENT UPDATE ===")
    W1_new = W1 - lr * dL_dW1
    b1_new = b1 - lr * dL_db1
    W2_new = W2 - lr * dL_dW2
    b2_new = b2 - lr * dL_db2
    W3_new = W3 - lr * dL_dW3
    b3_new = b3 - lr * dL_db3

    # Verify loss decreased
    z1_n = x @ W1_new + b1_new
    a1_n = relu(z1_n)
    z2_n = a1_n @ W2_new + b2_new
    a2_n = relu(z2_n)
    z3_n = a2_n @ W3_new + b3_new
    a3_n = sigmoid(z3_n)
    a3_n_clipped = np.clip(a3_n, 1e-7, 1 - 1e-7)
    new_loss = -np.mean(y * np.log(a3_n_clipped) + (1 - y) * np.log(1 - a3_n_clipped))

    print(f"Loss before update: {loss:.6f}")
    print(f"Loss after update:  {new_loss:.6f}")
    print(f"Loss decreased: {new_loss < loss}")

full_backprop_trace()
```

---

## The Vanishing Gradient Problem

In deep networks, gradients are products of many local gradients multiplied together via the chain rule. If any local gradient < 1, the product shrinks exponentially with depth:

```python
import numpy as np

def vanishing_gradient_demo():
    """
    Show gradient magnitude decay through a deep sigmoid network.
    10 layers, sigmoid activations — the gradient vanishes to near-zero
    by the time it reaches the early layers.
    """
    def sigmoid(z): return 1 / (1 + np.exp(-z))
    def sigmoid_grad(z):
        s = sigmoid(z)
        return s * (1 - s)  # max value = 0.25 at z=0

    np.random.seed(42)
    n_layers = 10

    # Simulate gradient flowing backward through 10 sigmoid layers
    # At each layer, gradient is multiplied by the local derivative
    grad_magnitude = 1.0   # start at output layer

    for layer_idx in range(n_layers - 1, -1, -1):
        z = np.random.randn() * 2   # typical pre-activation value
        local_grad = sigmoid_grad(z)   # max is 0.25
        grad_magnitude *= local_grad
        print(f"Layer {layer_idx+1:2d}: local gradient = {local_grad:.4f}  |  cumulative gradient = {grad_magnitude:.2e}")

    print(f"\nAfter {n_layers} sigmoid layers: gradient = {grad_magnitude:.2e}")
    print(f"This is {grad_magnitude:.2e} = {(0.25**n_layers):.2e} (if all at maximum 0.25)")

vanishing_gradient_demo()
```

**Why this breaks training:**
- Gradients in early layers are orders of magnitude smaller than in late layers
- Early layers learn almost nothing — they barely move
- The network effectively only learns at the output layers
- This was the fundamental barrier to training deep networks before 2015

---

## The Exploding Gradient Problem

The opposite: if weights are too large or gradients > 1, the product grows exponentially:

```python
def exploding_gradient_demo():
    """
    If weights are large (>1), gradients can grow exponentially through layers.
    This causes NaN losses and training failure.
    """
    np.random.seed(42)
    n_layers = 10

    grad_magnitude = 1.0

    for layer_idx in range(n_layers - 1, -1, -1):
        local_grad = np.random.uniform(1.5, 2.0)   # weights > 1
        grad_magnitude *= local_grad
        print(f"Layer {layer_idx+1:2d}: local gradient = {local_grad:.4f}  |  cumulative = {grad_magnitude:.2e}")

    print(f"\nExploding: gradient = {grad_magnitude:.2e}")
    print("Solution: Gradient Clipping — cap gradients at a maximum norm")

exploding_gradient_demo()
```

**Solutions to exploding gradients:**

```python
def gradient_clipping(gradients: np.ndarray, max_norm: float = 1.0) -> np.ndarray:
    """
    Gradient clipping: if gradient norm exceeds max_norm, scale it down.
    Used in all modern LLM training (GPT, LLaMA, etc).
    """
    grad_norm = np.linalg.norm(gradients)
    if grad_norm > max_norm:
        scale = max_norm / (grad_norm + 1e-6)
        print(f"Clipping gradient: norm {grad_norm:.4f} > {max_norm} → scaled by {scale:.4f}")
        return gradients * scale
    return gradients
```

---

## Residual Connections: Gradient Engineering

The Transformer architecture (and ResNet before it) solves vanishing gradients with **residual connections**:

\[
\text{output} = \mathcal{F}(x) + x
\]

The gradient of a residual block w.r.t. the input:

\[
\frac{\partial \text{output}}{\partial x} = \frac{\partial \mathcal{F}(x)}{\partial x} + 1
\]

The `+ 1` means there is **always at least a gradient of magnitude 1** flowing through the skip connection, regardless of what \(\mathcal{F}(x)\) does. Gradients cannot vanish through the skip path.

```python
import numpy as np

def residual_gradient_demo():
    """
    Compare gradient flow: no residual vs. with residual connections.
    """
    np.random.seed(42)
    n_layers = 20

    def block_without_residual(x, local_grad):
        return x * local_grad

    def block_with_residual(x, local_grad):
        # F(x) contributes local_grad, skip contributes 1.0
        # Gradient = local_grad + 1 (via chain rule)
        return x * (local_grad + 1.0)

    grad_no_res  = 1.0
    grad_with_res = 1.0

    print(f"{'Layer':6} {'No Residual':15} {'With Residual':15}")
    print("-" * 40)

    for layer in range(n_layers, 0, -1):
        local_grad = abs(np.random.randn() * 0.5)  # random small gradient
        grad_no_res  = block_without_residual(grad_no_res, local_grad)
        grad_with_res = block_with_residual(grad_with_res, local_grad)

        if layer % 5 == 0 or layer == 1:
            print(f"{layer:6d} {grad_no_res:15.6e} {grad_with_res:15.6e}")

    print(f"\nNo residual: gradient collapsed to {grad_no_res:.2e}")
    print(f"With residual: gradient maintained at {grad_with_res:.2e}")

residual_gradient_demo()
```

This is why every modern architecture (Transformer blocks, ResNet, DenseNet) uses residual connections. It is gradient engineering embedded in architecture.

---

## Layer Normalization: Stabilizing Activations

Layer Norm ensures that the input to each block has zero mean and unit variance, keeping gradients in a numerically stable range:

```python
import numpy as np

def layer_norm_gradient_demo():
    """
    LayerNorm prevents gradients from shrinking or exploding within a layer.
    The normalized activations stay near unit scale.
    """
    def layer_norm(x, eps=1e-5):
        mean = x.mean()
        var  = x.var()
        return (x - mean) / np.sqrt(var + eps)

    # Without LayerNorm: activations can have very large or small values
    x_unnorm = np.array([0.001, 0.002, 0.001, 500.0, 1000.0, 0.003])
    print("Without LayerNorm:")
    print(f"  Activations: {x_unnorm}")
    print(f"  Mean: {x_unnorm.mean():.2f}  Std: {x_unnorm.std():.2f}")
    print(f"  Sigmoid grad at 500: {1/(1+np.exp(-500)) * (1 - 1/(1+np.exp(-500))):.2e}")

    x_norm = layer_norm(x_unnorm)
    print("\nAfter LayerNorm:")
    print(f"  Activations: {x_norm.round(4)}")
    print(f"  Mean: {x_norm.mean():.6f}  Std: {x_norm.std():.6f}")
    print("  Activations are now in a numerically stable range")

layer_norm_gradient_demo()
```

---

## Summary of Modern Solutions to Gradient Problems

| Problem | Symptom | Solution |
|---------|---------|---------|
| Vanishing gradients | Early layers don't learn; training stalls | Residual connections, ReLU activation, Layer Norm |
| Exploding gradients | Loss becomes NaN; unstable training | Gradient clipping (`max_norm=1.0`), weight initialization |
| Slow convergence | Training needs too many epochs | Adam optimizer, learning rate scheduling |
| Dead neurons | Many neurons output 0 forever | Leaky ReLU, lower learning rate, careful initialization |

---

## Edge Cases and Misconceptions

**"Backpropagation computes exact gradients."** True only for the mathematical function defined by the network weights and operations. The *weight* updates are noisy when using mini-batch SGD, because the loss landscape (and its gradients) is estimated from a subset of data.

**"Deeper networks always have vanishing gradients."** Not anymore. Residual connections, Layer Norm, and ReLU activations have made very deep networks (GPT-4 has 128 layers) trainable. Vanishing gradients are primarily a concern with sigmoid/tanh activations in non-residual architectures.

**"Autograd (PyTorch) is different from backprop."** Autograd *implements* backpropagation. It traces the computation graph during the forward pass and then applies the chain rule during `.backward()`. The operations are the same; the interface is automatic.

**"You need to implement backprop yourself."** Not for production work — PyTorch/JAX handle it. But understanding it is essential for debugging, designing custom loss functions, and truly understanding what your model is doing.

---

## Key Takeaways

- Backpropagation is the chain rule applied to a computational graph, traversed from loss back to parameters
- At each operation node, a **local gradient** is computed; the overall gradient is the product of local gradients along each path from that parameter to the loss
- **Vanishing gradients** occur when many sigmoid/tanh layers multiply sub-unit local gradients; the product shrinks to zero, preventing early layers from learning
- **Residual connections** solve vanishing gradients by adding a skip connection with gradient 1, ensuring gradient can always flow backward
- **Gradient clipping** solves exploding gradients by capping gradient norm before applying weight updates
- Modern training = backpropagation + Adam optimizer + residual connections + Layer Norm + careful initialization

---

## Further Reading

- [Andrej Karpathy: micrograd](https://github.com/karpathy/micrograd) — 100-line autograd engine that implements everything in this lesson
- [Andrej Karpathy: Neural Networks Zero to Hero](https://karpathy.ai/zero-to-hero.html) — video walkthrough building from backprop to GPT
- [He et al. (2016): Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — original ResNet paper introducing residual connections
- [Bengio et al. (1994): Learning Long-Term Dependencies with Gradient Descent is Difficult](http://www.iro.umontreal.ca/~lisa/pointeurs/ieeetrnn94.pdf) — the paper that formalized the vanishing gradient problem

---

**You've completed Module 05 Lessons 1-5.** You now have the mathematical and computational foundations to understand how modern neural networks — including Transformers — learn from data. The next step: apply these concepts to the full Transformer architecture in [Module 00: From NLP to Transformers](../../module-00-genai-foundations-from-nlp-to-transformers/lessons/07-transformer-architecture.md).
