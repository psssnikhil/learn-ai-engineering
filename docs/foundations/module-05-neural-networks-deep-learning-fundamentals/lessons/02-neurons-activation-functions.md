---
title: 'Neurons, Activation Functions & Forward Propagation'
description: >-
  Learn how artificial neurons compute, understand activation functions, and
  implement forward propagation
duration: 30 min
difficulty: beginner
has_code: true
module: module-05
youtube: 'https://www.youtube.com/watch?v=aircAruvnKk'
objectives:
  - Understand how neurons perform mathematical operations
  - Explain different activation functions and when to use them
  - Implement forward propagation
  - Calculate neuron outputs step-by-step
---
# Neurons, Activation Functions & Forward Propagation

![Neural Network Math](https://images.unsplash.com/photo-1635070041078-e363dbe005cb?w=800)

## The Math Behind Neurons

In Lesson 1, we learned what neural networks are conceptually. Now let's see **exactly how they work mathematically**.

Don't worry - the math is simpler than you think!

---

## The Neuron Formula

A neuron does two things:
1. **Weighted sum**: Multiplies inputs by weights and adds them up
2. **Activation**: Applies a function to decide the output

### Step 1: Weighted Sum (Linear Combination)

```
z = (w₁ × x₁) + (w₂ × x₂) + (w₃ × x₃) + ... + b
```

Or more compactly:
```
z = Σ(wᵢ × xᵢ) + b
```

**Where:**
- `x₁, x₂, x₃...` = Input values
- `w₁, w₂, w₃...` = Weights (what the network learns!)
- `b` = Bias (allows shifting the decision boundary)
- `z` = The weighted sum (before activation)

### Step 2: Activation Function

```
output = f(z)
```

Where `f()` is the activation function (we'll explore these next!)

---

## Real Example: Predicting House Prices

Let's build a simple neuron to predict if a house is expensive:

**Inputs:**
- x₁ = Square footage (normalized: 0-1)
- x₂ = Number of bedrooms (normalized: 0-1)
- x₃ = Distance to city center (normalized: 0-1)

**Weights (learned from data):**
- w₁ = 0.7 (square footage is very important)
- w₂ = 0.3 (bedrooms matter somewhat)
- w₃ = -0.5 (further from city = cheaper)
- b = 0.1 (bias term)

**Example house:**
- 2000 sq ft → x₁ = 0.6
- 3 bedrooms → x₂ = 0.5
- Close to city → x₃ = 0.2

**Calculate:**
```
z = (0.7 × 0.6) + (0.3 × 0.5) + (-0.5 × 0.2) + 0.1
z = 0.42 + 0.15 - 0.1 + 0.1
z = 0.57

Apply activation (sigmoid):
output = 1 / (1 + e^(-0.57)) ≈ 0.64

Result: 64% confidence this house is expensive
```

---

## Activation Functions: The Non-Linearity Secret

**Why do we need activation functions?**

Without them, no matter how many layers you stack, your network is just doing linear transformations. It would be equivalent to a single layer!

**Activation functions add non-linearity**, allowing networks to learn complex patterns.

### 1. Sigmoid (σ) - The Classic

```
σ(z) = 1 / (1 + e^(-z))
```

**Output range**: 0 to 1

**Shape**: S-curve

```
  1.0 |         ┌────
      |       ╱
  0.5 |     ╱
      |   ╱
  0.0 | ─┘
      └────────────
     -5    0    5
```

**Pros:**
- ✅ Output is probability-like (0-1)
- ✅ Smooth gradient
- ✅ Great for binary classification

**Cons:**
- ❌ Vanishing gradients (values too close to 0 or 1)
- ❌ Not zero-centered
- ❌ Slow to compute (exponential)

**When to use**: Output layer for binary classification

---

### 2. ReLU (Rectified Linear Unit) - The Modern Default

```
ReLU(z) = max(0, z)
```

**Output range**: 0 to ∞

**Shape**: Hockey stick

```
  10 |           ╱
     |         ╱
   5 |       ╱
     |     ╱
   0 |───────────
     └────────────
    -5    0    5
```

**Pros:**
- ✅ Dead simple to compute
- ✅ No vanishing gradient for positive values
- ✅ Networks train much faster
- ✅ Induces sparsity (some neurons output 0)

**Cons:**
- ❌ "Dying ReLU" problem (neurons can get stuck at 0)
- ❌ Not differentiable at 0

**When to use**: Hidden layers (most common choice!)

---

### 3. Leaky ReLU - ReLU's Better Brother

```
Leaky ReLU(z) = max(0.01z, z)
```

**Output range**: -∞ to ∞

**Shape**: Almost like ReLU but with a small slope for negatives

```
  10 |           ╱
     |         ╱
   5 |       ╱
     |     ╱
   0 |   ╱
     | ╱
  -1 |╱
     └────────────
    -5    0    5
```

**Pros:**
- ✅ Fixes dying ReLU problem
- ✅ Still simple and fast
- ✅ Works well in practice

**When to use**: Hidden layers (alternative to ReLU)

---

### 4. Tanh (Hyperbolic Tangent)

```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
```

**Output range**: -1 to 1

**Shape**: S-curve, zero-centered

```
  1.0 |        ┌────
      |      ╱
  0.0 |    ╱
      |  ╱
 -1.0 | ┘
      └────────────
     -5    0    5
```

**Pros:**
- ✅ Zero-centered (better than sigmoid)
- ✅ Stronger gradients than sigmoid

**Cons:**
- ❌ Still has vanishing gradient problem
- ❌ Slower than ReLU

**When to use**: RNNs (next module!), sometimes hidden layers

---

### 5. Softmax - For Multiple Classes

```
Softmax(zᵢ) = e^(zᵢ) / Σⱼ e^(zⱼ)
```

**Output**: Probability distribution (sums to 1)

**Example:**
```
z = [2.0, 1.0, 0.1]

Softmax:
[0.659, 0.242, 0.099]  ← Probabilities
```

**When to use**: Output layer for multi-class classification
- Cat, Dog, or Bird? → Use Softmax
- Gives probability for each class

---

## Choosing the Right Activation

| Layer Type | Task | Best Choice |
|------------|------|-------------|
| Hidden layers | Any | **ReLU** or Leaky ReLU |
| Output | Binary classification | **Sigmoid** |
| Output | Multi-class | **Softmax** |
| Output | Regression | **None** (linear) |
| RNN/LSTM | Sequence modeling | **Tanh** |

**Rule of thumb**: Start with ReLU for hidden layers. It works 90% of the time!

---

## Forward Propagation: Moving Through the Network

**Forward propagation** is the process of passing input through the network to get output.

### Simple Network Example

```
Input → Hidden Layer → Output
  x₁        h₁          y
  x₂   →    h₂     →
  x₃        h₃
```

**Step-by-step calculation:**

**Given:**
- Inputs: x = [1.0, 2.0, 3.0]
- Weights (input→hidden): W₁ = [[0.5, 0.2, -0.1], [0.3, 0.8, 0.4], [0.1, -0.2, 0.7]]
- Bias (hidden): b₁ = [0.1, 0.2, 0.3]
- Weights (hidden→output): W₂ = [[0.6], [0.3], [-0.2]]
- Bias (output): b₂ = [0.5]

**Forward pass:**

1. **Hidden layer calculation:**
```python
z₁ = W₁ × x + b₁
z₁ = [[0.5, 0.2, -0.1],    [[1.0],     [[0.1],
      [0.3, 0.8, 0.4],  ×   [2.0],  +   [0.2],
      [0.1, -0.2, 0.7]]     [3.0]]      [0.3]]

z₁ = [0.6, 2.1, 1.8]

h = ReLU(z₁) = [0.6, 2.1, 1.8]  # All positive, so unchanged
```

2. **Output layer calculation:**
```python
z₂ = W₂ × h + b₂
z₂ = [[0.6, 0.3, -0.2]] × [[0.6], [2.1], [1.8]] + [0.5]
z₂ = 0.36 + 0.63 - 0.36 + 0.5 = 1.13

y = Sigmoid(1.13) = 0.756
```

**Result**: Network outputs 0.756 (75.6% confidence)

---

## Coding Time! Building a Neuron

Let's implement this in Python:

```python
import numpy as np

def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))

def relu(z):
    """ReLU activation function"""
    return np.maximum(0, z)

def neuron(inputs, weights, bias, activation='relu'):
    """
    A single neuron computation
    
    Args:
        inputs: array of input values
        weights: array of weights
        bias: bias value
        activation: 'relu', 'sigmoid', or 'tanh'
    
    Returns:
        output of the neuron
    """
    # Step 1: Weighted sum
    z = np.dot(inputs, weights) + bias
    
    # Step 2: Apply activation
    if activation == 'relu':
        return relu(z)
    elif activation == 'sigmoid':
        return sigmoid(z)
    elif activation == 'tanh':
        return np.tanh(z)
    else:
        return z  # Linear activation

# Example usage
inputs = np.array([1.0, 2.0, 3.0])
weights = np.array([0.5, 0.3, -0.2])
bias = 0.1

output = neuron(inputs, weights, bias, activation='sigmoid')
print(f"Neuron output: {output:.4f}")
# Output: Neuron output: 0.6457
```

---

## The Power of Matrix Operations

For efficiency, we use **vectorized operations** (matrices):

```python
import numpy as np

def forward_propagation(X, W1, b1, W2, b2):
    """
    Forward pass through a 2-layer network
    
    Args:
        X: Input (shape: n_samples × n_features)
        W1: Hidden weights (shape: n_features × n_hidden)
        b1: Hidden bias (shape: n_hidden)
        W2: Output weights (shape: n_hidden × n_output)
        b2: Output bias (shape: n_output)
    
    Returns:
        predictions: Output of network
    """
    # Hidden layer
    Z1 = np.dot(X, W1) + b1
    A1 = np.maximum(0, Z1)  # ReLU
    
    # Output layer
    Z2 = np.dot(A1, W2) + b2
    A2 = 1 / (1 + np.exp(-Z2))  # Sigmoid
    
    return A2

# Example with 3 samples
X = np.array([[1.0, 2.0, 3.0],
              [0.5, 1.5, 2.5],
              [2.0, 1.0, 0.5]])

W1 = np.random.randn(3, 4)  # 3 inputs → 4 hidden neurons
b1 = np.zeros(4)
W2 = np.random.randn(4, 1)  # 4 hidden → 1 output
b2 = np.zeros(1)

predictions = forward_propagation(X, W1, b1, W2, b2)
print("Predictions:", predictions.flatten())
```

**Why matrices?**
- Process multiple samples at once (batch processing)
- GPUs are optimized for matrix operations
- 100x+ faster than loops!

---

## Visualizing Forward Propagation

```
Input (x)          Hidden (h)        Output (y)

  1.0              0.6
  2.0   ×W₁+b₁→    2.1    ×W₂+b₂→   0.756
  3.0              1.8
  
  ↓                ↓                 ↓
Multiply by     Apply ReLU       Apply Sigmoid
weights         activation       activation
```

**The flow:**
1. Inputs → multiply by weights → add bias → **activation** → Hidden layer
2. Hidden → multiply by weights → add bias → **activation** → Output

This happens **forward** through the network, hence "forward propagation"!

---

## 📹 Watch Next

- [3Blue1Brown: Neural Networks Part 1](https://www.youtube.com/watch?v=aircAruvnKk) - Beautiful visualizations
- [Activation Functions Explained](https://www.youtube.com/watch?v=m0pIlLfpXWE) - Visual intuition
- [But what is backpropagation?](https://www.youtube.com/watch?v=Ilg3gGewQ5U) - Preview of next lesson

---

## 🎯 Key Takeaways

1. Neurons perform **weighted sum + activation function**
2. **Activation functions** add non-linearity (networks can learn complex patterns)
3. **ReLU** is the default choice for hidden layers
4. **Sigmoid** for binary classification output, **Softmax** for multi-class
5. **Forward propagation** passes data through the network layer by layer
6. Matrix operations make computations **fast and efficient**

---

## ✅ Practice Exercise

Try this yourself:

**Build a neuron that predicts if a student will pass an exam:**

Inputs:
- x₁ = Hours studied (0-10, normalized to 0-1)
- x₂ = Previous grade (0-100, normalized to 0-1)
- x₃ = Attendance (0-100%, normalized to 0-1)

Use these weights:
- w = [0.5, 0.6, 0.3]
- b = -0.2

Calculate the output for a student who:
- Studied 8 hours (x₁ = 0.8)
- Had previous grade 85 (x₂ = 0.85)  
- Attended 90% of classes (x₃ = 0.9)

Use sigmoid activation. Will they pass? (>0.5 = pass)

---

## 📹 Recommended Videos

- [Activation Functions Explained](https://www.youtube.com/watch?v=-7scQpJT7uo) — StatQuest with Josh Starmer
- [ReLU, Sigmoid, Tanh — Which Activation Function?](https://www.youtube.com/watch?v=68BZ5f7P94Q) — Deeplearning.AI
- [The Neuron](https://www.youtube.com/watch?v=aircAruvnKk&t=60) — 3Blue1Brown's breakdown of neuron math

---

## 📚 Additional Resources

- [Understanding Activation Functions](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/) — Machine Learning Mastery
- [CS231n: Neural Network Basics](https://cs231n.github.io/neural-networks-1/) — Stanford's classic deep learning course notes
- [Activation Functions Comparison](https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity) — PyTorch official docs

---

## 🚀 Next Lesson

In **Lesson 3**, we'll learn:
- How to measure if the network is doing well (Loss Functions)
- Different types of loss for different problems
- How loss guides the learning process

**Ready to learn how networks learn?** Let's go! 💪
