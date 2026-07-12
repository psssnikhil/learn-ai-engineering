---
title: Backpropagation - How Networks Learn
description: >-
  Master the algorithm that makes deep learning possible - backpropagation and
  the chain rule
duration: 40 min
difficulty: intermediate
has_code: true
youtube: 'https://www.youtube.com/watch?v=Ilg3gGewQ5U'
objectives:
  - Understand backpropagation algorithm
  - Apply the chain rule to compute gradients
  - Trace gradients backward through layers
  - Implement backpropagation from scratch
---
# Backpropagation: How Networks Learn

## The Most Clever Algorithm in AI

**Backpropagation** (short for "backward propagation of errors") is the algorithm that makes training deep networks possible.

**The problem it solves**: How do we calculate gradients for millions of weights efficiently?

**Answer**: Use the chain rule from calculus, but apply it backward through the network!

---

## Forward vs Backward

```
FORWARD (Prediction):
Input → Hidden → Output → Loss
  x   →   h    →   ŷ    →  L

BACKWARD (Learning):
∂L/∂W ← Gradients flow backward ← Loss
```

**Forward pass**: Calculate predictions  
**Backward pass**: Calculate how to improve (gradients)

---

## The Chain Rule: The Key to Everything

Remember calculus? The chain rule says:

```
If y = f(u) and u = g(x)
Then: dy/dx = (dy/du) × (du/dx)
```

**Example**:
```
y = (3x + 2)²

Let u = 3x + 2
Then y = u²

dy/dx = (dy/du) × (du/dx)
      = 2u × 3
      = 2(3x + 2) × 3
      = 6(3x + 2)
```

Backprop applies this repeatedly through the network!

---

## Simple 2-Layer Network Example

```
Input (x) → Hidden (h) → Output (y) → Loss (L)
```

**Forward equations**:
```
h = σ(W₁x + b₁)    # Hidden layer
y = σ(W₂h + b₂)    # Output layer
L = (y - target)²  # Loss
```

**Goal**: Find ∂L/∂W₁ and ∂L/∂W₂ (how loss changes with weights)

---

## Backpropagation Step-by-Step

### Step 1: Output Layer Gradient

```
∂L/∂y = 2(y - target)    # Derivative of loss
```

### Step 2: Backprop Through Output Activation

```
∂L/∂z₂ = ∂L/∂y × ∂y/∂z₂
       = ∂L/∂y × σ'(z₂)   # Chain rule!
```

Where z₂ = W₂h + b₂ (before activation)

### Step 3: Weight Gradient (Output Layer)

```
∂L/∂W₂ = ∂L/∂z₂ × ∂z₂/∂W₂
       = ∂L/∂z₂ × h       # Chain rule again!
```

### Step 4: Backprop to Hidden Layer

```
∂L/∂h = ∂L/∂z₂ × ∂z₂/∂h
      = ∂L/∂z₂ × W₂      # Gradient flows backward!
```

### Step 5: Hidden Layer Gradient

```
∂L/∂z₁ = ∂L/∂h × ∂h/∂z₁
       = ∂L/∂h × σ'(z₁)
```

### Step 6: Weight Gradient (Hidden Layer)

```
∂L/∂W₁ = ∂L/∂z₁ × ∂z₁/∂W₁
       = ∂L/∂z₁ × x
```

**Done!** Now we can update all weights using gradient descent.

---

## Numerical Example

Let's trace through an actual example:

**Forward pass**:
```
x = 2.0
W₁ = 0.5, b₁ = 0.1
W₂ = 0.8, b₂ = 0.2
target = 1.0

# Hidden layer
z₁ = W₁×x + b₁ = 0.5×2.0 + 0.1 = 1.1
h = σ(z₁) = 0.750

# Output layer
z₂ = W₂×h + b₂ = 0.8×0.750 + 0.2 = 0.8
y = σ(z₂) = 0.689

# Loss
L = (y - target)² = (0.689 - 1.0)² = 0.097
```

**Backward pass**:
```
# Output layer
∂L/∂y = 2(y - target) = 2(0.689 - 1.0) = -0.622

∂y/∂z₂ = σ'(z₂) = y(1-y) = 0.689×0.311 = 0.214

∂L/∂z₂ = ∂L/∂y × ∂y/∂z₂ = -0.622 × 0.214 = -0.133

∂L/∂W₂ = ∂L/∂z₂ × h = -0.133 × 0.750 = -0.100

# Hidden layer
∂L/∂h = ∂L/∂z₂ × W₂ = -0.133 × 0.8 = -0.106

∂h/∂z₁ = σ'(z₁) = h(1-h) = 0.750×0.250 = 0.188

∂L/∂z₁ = ∂L/∂h × ∂h/∂z₁ = -0.106 × 0.188 = -0.020

∂L/∂W₁ = ∂L/∂z₁ × x = -0.020 × 2.0 = -0.040
```

**Update weights** (learning rate = 0.1):
```
W₂ = W₂ - 0.1×∂L/∂W₂ = 0.8 - 0.1×(-0.100) = 0.810
W₁ = W₁ - 0.1×∂L/∂W₁ = 0.5 - 0.1×(-0.040) = 0.504
```

Loss decreased! The network learned! 🎉

---

## Code Implementation

```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Derivative of sigmoid: σ'(x) = σ(x)(1 - σ(x))"""
    s = sigmoid(x)
    return s * (1 - s)

class TwoLayerNetwork:
    def __init__(self):
        # Initialize weights small random values
        self.W1 = np.random.randn() * 0.1
        self.b1 = 0.0
        self.W2 = np.random.randn() * 0.1
        self.b2 = 0.0
        
    def forward(self, x):
        """Forward propagation"""
        # Store intermediate values for backprop
        self.x = x
        
        # Hidden layer
        self.z1 = self.W1 * x + self.b1
        self.h = sigmoid(self.z1)
        
        # Output layer
        self.z2 = self.W2 * self.h + self.b2
        self.y = sigmoid(self.z2)
        
        return self.y
    
    def backward(self, target, learning_rate=0.1):
        """Backpropagation"""
        # Output layer gradients
        dL_dy = 2 * (self.y - target)
        dy_dz2 = sigmoid_derivative(self.z2)
        dL_dz2 = dL_dy * dy_dz2
        
        dL_dW2 = dL_dz2 * self.h
        dL_db2 = dL_dz2
        
        # Hidden layer gradients
        dL_dh = dL_dz2 * self.W2
        dh_dz1 = sigmoid_derivative(self.z1)
        dL_dz1 = dL_dh * dh_dz1
        
        dL_dW1 = dL_dz1 * self.x
        dL_db1 = dL_dz1
        
        # Update weights (gradient descent)
        self.W2 -= learning_rate * dL_dW2
        self.b2 -= learning_rate * dL_db2
        self.W1 -= learning_rate * dL_dW1
        self.b1 -= learning_rate * dL_db1
    
    def train(self, x, target, epochs=100):
        """Training loop"""
        for epoch in range(epochs):
            # Forward pass
            prediction = self.forward(x)
            
            # Calculate loss
            loss = (prediction - target)**2
            
            # Backward pass
            self.backward(target)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.6f}, Prediction = {prediction:.6f}")

# Train the network
network = TwoLayerNetwork()
network.train(x=2.0, target=1.0, epochs=100)
```

---

## Computational Graph Visualization

```
Forward (blue arrows →):
x=2.0 → ×W₁=0.5 → +b₁ → σ → h=0.75 → ×W₂=0.8 → +b₂ → σ → y=0.69 → L=0.097
                                                                      ↑target=1.0

Backward (red arrows ←):
Gradients flow backward through each operation
```

Each operation stores:
1. **Forward**: Its output
2. **Backward**: How to compute gradients

---

## Why Backpropagation is Brilliant

### Before Backprop (Manual Gradients)

For each weight:
1. Perturb weight slightly: W + ε
2. Run forward pass
3. Calculate finite difference: (L(W+ε) - L(W)) / ε
4. Repeat for ALL weights

**Problem**: N forward passes for N weights! 🐌

---

### With Backprop

1. One forward pass
2. One backward pass
3. Get ALL gradients at once!

**Complexity**: O(2E) where E = number of edges  
**Before**: O(NE) where N = number of weights

**Speedup**: 100x to 1000x faster! 🚀

---

## Matrix Form (Real Implementation)

For a batch of samples:

```python
def backprop_batch(X, y, W1, b1, W2, b2):
    """
    Backprop for a batch of samples
    X: (batch_size, input_dim)
    y: (batch_size, output_dim)
    """
    batch_size = X.shape[0]
    
    # Forward
    Z1 = X @ W1 + b1                    # (batch, hidden)
    A1 = np.maximum(0, Z1)              # ReLU
    Z2 = A1 @ W2 + b2                   # (batch, output)
    A2 = 1 / (1 + np.exp(-Z2))          # Sigmoid
    
    # Backward
    dZ2 = A2 - y                        # (batch, output)
    dW2 = (1/batch_size) * (A1.T @ dZ2)  # (hidden, output)
    db2 = (1/batch_size) * np.sum(dZ2, axis=0)
    
    dA1 = dZ2 @ W2.T                    # (batch, hidden)
    dZ1 = dA1 * (Z1 > 0)                # ReLU derivative
    dW1 = (1/batch_size) * (X.T @ dZ1)   # (input, hidden)
    db1 = (1/batch_size) * np.sum(dZ1, axis=0)
    
    return dW1, db1, dW2, db2
```

---

## Vanishing and Exploding Gradients

### Vanishing Gradients

Deep networks can suffer from **vanishing gradients**:

```
Layer 10 → 9 → 8 → 7 → 6 → 5 → 4 → 3 → 2 → 1
Gradient: 0.0001 ← Gets smaller and smaller
```

**Why?** Chain rule multiplies many derivatives < 1

**Solution**:
- Use ReLU (doesn't vanish for positive values)
- Batch normalization
- Residual connections (skip connections)

---

### Exploding Gradients

Opposite problem - gradients get huge:

```
Gradient: 10 → 100 → 1000 → 10000 → NaN!
```

**Solution**:
- Gradient clipping
- Better weight initialization
- Lower learning rate

---

## 📹 Recommended Videos

- [3Blue1Brown: Backpropagation](https://www.youtube.com/watch?v=Ilg3gGewQ5U) - Best visual explanation
- [Stanford CS231n: Backprop](https://www.youtube.com/watch?v=d14TUNcbn1k) - Detailed lecture
- [Backprop Calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) - 3Blue1Brown math deep dive

---

## 🎯 Key Takeaways

1. **Backpropagation** efficiently computes gradients using the chain rule
2. **Forward pass** calculates predictions, stores intermediate values
3. **Backward pass** calculates gradients by flowing backward
4. One forward + one backward = gradients for ALL weights
5. Matrix operations make it efficient for batches
6. Watch out for vanishing/exploding gradients

---

## 📹 Recommended Videos

- [Backpropagation Calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) — 3Blue1Brown's deep dive into the math
- [What is Backpropagation Really Doing?](https://www.youtube.com/watch?v=Ilg3gGewQ5U) — 3Blue1Brown intuition video
- [Backpropagation Explained](https://www.youtube.com/watch?v=SmZmBKc7Lrs) — deeplizard clear walkthrough

---

## 📚 Additional Resources

- [Calculus on Computational Graphs: Backpropagation](https://colah.github.io/posts/2015-08-Backprop/) — Chris Olah's excellent visual guide
- [CS231n: Backpropagation](https://cs231n.github.io/optimization-2/) — Stanford course notes on computing gradients
- [Yes You Should Understand Backprop](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b) — Andrej Karpathy's blog post

---

## 🚀 Next Lesson

**Lesson 6**: Overfitting, Regularization & Dropout
- Why networks memorize instead of learn
- How to prevent overfitting
- Regularization techniques
- Implementing dropout

**Keep going!** 💪
