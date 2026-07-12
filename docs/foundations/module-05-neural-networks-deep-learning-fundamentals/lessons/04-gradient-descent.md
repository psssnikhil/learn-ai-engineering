---
title: Gradient Descent - The Learning Algorithm
description: >-
  Discover how neural networks actually learn by following gradients downhill to
  minimize loss
duration: 35 min
difficulty: intermediate
has_code: true
module: module-05
youtube: 'https://www.youtube.com/watch?v=IHZwWFHWa-w'
objectives:
  - Understand how gradient descent works
  - Explain the role of learning rate
  - Calculate gradients for simple functions
  - Recognize different variants of gradient descent
---
# Gradient Descent: The Learning Algorithm

![Gradient Descent](https://images.unsplash.com/photo-1509228468518-180dd4864904?w=800)

## The Most Important Algorithm in Deep Learning

**Gradient descent** is how neural networks learn. It's the magic that makes everything work!

Think of it like hiking down a mountain in fog:
- You can't see the bottom
- But you can feel which direction is downhill
- You take small steps in that direction
- Eventually, you reach the valley

Neural networks do exactly this to minimize loss!

---

## The Intuition: Rolling Downhill

Imagine a ball rolling down a curved surface:

```
        Loss
         ↑
    🔴   |
     ╲  |
      ╲ |
       ╲|
        •────────→ Weight
      Valley
```

- **Ball** = Current weights
- **Surface** = Loss landscape
- **Gravity** = Gradient (direction to go)
- **Goal** = Reach the lowest point (minimum loss)

The ball naturally rolls downhill. Gradient descent does the same thing mathematically!

---

## The Math (Simplified)

### What is a Gradient?

The **gradient** is the direction and rate of steepest increase.

For a function `f(x)`:
```
Gradient = ∂f/∂x (derivative)
```

**Example**: `f(x) = x²`

```
f(x) = x²
Gradient = 2x

At x=3: gradient = 6 (steep upward slope)
At x=0: gradient = 0 (flat, we're at minimum!)
```

### The Update Rule

```
new_weight = old_weight - learning_rate × gradient
```

**Why minus?**
- Gradient points **uphill** (increasing loss)
- We want to go **downhill** (decreasing loss)
- So we subtract!

---

## Step-by-Step Example

Let's minimize `f(x) = (x-3)²` (minimum at x=3)

**Starting point**: x = 0  
**Learning rate**: α = 0.1  
**Gradient**: f'(x) = 2(x-3)

**Iteration 1:**
```
x₀ = 0
gradient = 2(0-3) = -6
x₁ = 0 - 0.1×(-6) = 0 + 0.6 = 0.6
```

**Iteration 2:**
```
x₁ = 0.6
gradient = 2(0.6-3) = -4.8
x₂ = 0.6 - 0.1×(-4.8) = 1.08
```

**Iteration 3:**
```
x₂ = 1.08
gradient = 2(1.08-3) = -3.84
x₃ = 1.08 + 0.384 = 1.464
```

Continue... converges to x ≈ 3! 🎯

---

## Learning Rate: The Most Important Hyperparameter

The **learning rate (α)** controls how big each step is.

### Too Small

```
Learning rate = 0.001

Loss
  ↑
10|•
  | •
 5|  •
  |   •
 0|____•___________→ Steps
      (Very slow! Takes forever)
```

**Problem**: Training takes too long, might never converge

---

### Too Large

```
Learning rate = 10

Loss
  ↑
15|    •
  |  •   •
10|•       •
  |  •   •    (Bouncing around!)
 5|    •
  └─────────────→ Steps
```

**Problem**: Overshoots the minimum, loss oscillates or explodes!

---

### Just Right

```
Learning rate = 0.1

Loss
  ↑
10|•
  | ╲
 5|  ╲
  |   ╲__
 0|______╲_____→ Steps
      (Smooth convergence)
```

**Sweet spot**: Fast enough, but stable

**Typical values**: 0.001, 0.01, 0.1
- Usually start with 0.001 and tune from there

---

## Gradient Descent in Neural Networks

For a neural network with weights W and bias b:

```python
# Forward pass
predictions = model(X, W, b)

# Calculate loss
loss = loss_function(y_true, predictions)

# Backward pass (compute gradients)
∂L/∂W, ∂L/∂b = compute_gradients(loss)

# Update weights
W = W - learning_rate × ∂L/∂W
b = b - learning_rate × ∂L/∂b
```

This happens **thousands of times** during training!

---

## Code Implementation

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(x_start, learning_rate, num_iterations):
    """
    Minimize f(x) = (x-3)² using gradient descent
    """
    x = x_start
    history = [x]
    
    for i in range(num_iterations):
        # Calculate gradient: f'(x) = 2(x-3)
        gradient = 2 * (x - 3)
        
        # Update x
        x = x - learning_rate * gradient
        history.append(x)
        
        if i % 10 == 0:
            loss = (x - 3)**2
            print(f"Iteration {i}: x = {x:.4f}, loss = {loss:.4f}")
    
    return x, history

# Run gradient descent
final_x, history = gradient_descent(
    x_start=0, 
    learning_rate=0.1, 
    num_iterations=50
)

print(f"
Final x: {final_x:.4f}")
print(f"Target: 3.0000")

# Visualize
plt.figure(figsize=(10, 6))
plt.plot(history, label='x value')
plt.axhline(y=3, color='r', linestyle='--', label='Target (minimum)')
plt.xlabel('Iteration')
plt.ylabel('x')
plt.title('Gradient Descent Convergence')
plt.legend()
plt.grid(True)
plt.show()
```

---

## Variants of Gradient Descent

### 1. Batch Gradient Descent (BGD)

Uses **all** training data for each update.

```python
# Calculate gradient on entire dataset
for epoch in range(num_epochs):
    gradient = compute_gradient(X_train, y_train, weights)
    weights = weights - learning_rate * gradient
```

**Pros:**
- ✅ Stable, smooth convergence
- ✅ Guaranteed to reach minimum (convex functions)

**Cons:**
- ❌ Slow for large datasets
- ❌ Can't fit huge datasets in memory

---

### 2. Stochastic Gradient Descent (SGD)

Uses **one** training example at a time.

```python
# Update after each sample
for epoch in range(num_epochs):
    for x, y in shuffle(training_data):
        gradient = compute_gradient(x, y, weights)
        weights = weights - learning_rate * gradient
```

**Pros:**
- ✅ Fast updates
- ✅ Can escape local minima (noisy updates)
- ✅ Works with huge datasets

**Cons:**
- ❌ Noisy, erratic path
- ❌ May not converge exactly

---

### 3. Mini-Batch Gradient Descent (Most Common!)

Uses **small batches** of data (typically 32, 64, 128, 256).

```python
batch_size = 32

for epoch in range(num_epochs):
    for batch in get_batches(X_train, y_train, batch_size):
        X_batch, y_batch = batch
        gradient = compute_gradient(X_batch, y_batch, weights)
        weights = weights - learning_rate * gradient
```

**Pros:**
- ✅ Balanced: faster than BGD, more stable than SGD
- ✅ Efficient use of GPU/CPU vectorization
- ✅ Best of both worlds!

**Cons:**
- None really - this is the standard!

**Best practice**: Use mini-batch with batch size 32-256

---

## Advanced Optimizers (Beyond Basic GD)

### 1. Momentum

Adds "momentum" to smooth out updates:

```python
velocity = 0.9 * velocity + learning_rate * gradient
weights = weights - velocity
```

Think: **ball rolling with inertia**

**Pros:**
- Accelerates in consistent directions
- Dampens oscillations
- Faster convergence

---

### 2. Adam (Adaptive Moment Estimation)

The **most popular** optimizer (2024):

```python
# Combines momentum + adaptive learning rates
m = β₁ * m + (1-β₁) * gradient        # 1st moment
v = β₂ * v + (1-β₂) * gradient²       # 2nd moment
weights = weights - α * m / (√v + ε)
```

**Why Adam is popular:**
- ✅ Works well out of the box
- ✅ Adapts learning rate per parameter
- ✅ Fast convergence
- ✅ Default choice for most problems

**Typical hyperparameters:**
- α (learning rate) = 0.001
- β₁ = 0.9
- β₂ = 0.999

---

### 3. RMSprop

```python
# Divides learning rate by running average of gradient magnitudes
cache = 0.9 * cache + 0.1 * gradient²
weights = weights - learning_rate * gradient / (√cache + ε)
```

**Good for**: Recurrent neural networks (RNNs)

---

## Comparison of Optimizers

```
Path to minimum:

SGD:        ~~~~~~~~~~~~~~~~•  (zigzag, slow)
Momentum:   ~~~~~~~•           (smoother, faster)
Adam:       ~~~~•              (fastest, smoothest)
```

**Recommendation**: Start with **Adam**, learning rate 0.001

---

## Complete Training Loop Example

```python
import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Initialize weights randomly
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
        self.learning_rate = learning_rate
        self.loss_history = []
    
    def forward(self, X):
        """Forward propagation"""
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.maximum(0, self.Z1)  # ReLU
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = 1 / (1 + np.exp(-self.Z2))  # Sigmoid
        return self.A2
    
    def backward(self, X, y):
        """Backward propagation (compute gradients)"""
        m = X.shape[0]
        
        # Output layer gradients
        dZ2 = self.A2 - y
        dW2 = (1/m) * np.dot(self.A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.Z1 > 0)  # ReLU derivative
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        return dW1, db1, dW2, db2
    
    def update_weights(self, dW1, db1, dW2, db2):
        """Gradient descent update"""
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def train(self, X, y, epochs=1000):
        """Full training loop"""
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X)
            
            # Calculate loss
            loss = -np.mean(y * np.log(predictions + 1e-8) + 
                           (1 - y) * np.log(1 - predictions + 1e-8))
            self.loss_history.append(loss)
            
            # Backward pass
            dW1, db1, dW2, db2 = self.backward(X, y)
            
            # Update weights (gradient descent!)
            self.update_weights(dW1, db1, dW2, db2)
            
            if epoch % 100 == 0:
                accuracy = np.mean((predictions > 0.5) == y)
                print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

# Example usage
X = np.random.randn(100, 3)  # 100 samples, 3 features
y = (X[:, 0] + X[:, 1] > 0).astype(float).reshape(-1, 1)  # Simple rule

nn = NeuralNetwork(input_size=3, hidden_size=5, output_size=1, learning_rate=0.1)
nn.train(X, y, epochs=1000)
```

---

## Visualizing the Training Process

```python
import matplotlib.pyplot as plt

def plot_training(loss_history):
    """Visualize loss decreasing over time"""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.yscale('log')  # Log scale to see details
    plt.show()

plot_training(nn.loss_history)
```

---

## Common Issues and Solutions

### Issue 1: Loss Exploding (NaN)
**Cause**: Learning rate too high  
**Solution**: Reduce learning rate by 10x

### Issue 2: Loss Not Decreasing
**Cause**: Learning rate too low, or stuck in bad local minimum  
**Solution**: Increase learning rate, or restart with different initialization

### Issue 3: Loss Oscillating
**Cause**: Learning rate too high  
**Solution**: Reduce learning rate, use momentum or Adam

### Issue 4: Slow Training
**Cause**: Learning rate too low, or dataset too large  
**Solution**: Increase learning rate, use mini-batches, try Adam

---

## 📹 Recommended Videos

- [3Blue1Brown: Gradient Descent](https://www.youtube.com/watch?v=IHZwWFHWa-w) - Beautiful visualization
- [StatQuest: Gradient Descent](https://www.youtube.com/watch?v=sDv4f4s2SB8) - Clear explanation
- [Optimizers Explained](https://www.youtube.com/watch?v=mdKjMPmcWjY) - Adam, RMSprop, etc.

---

---

## 🎯 Key Takeaways

```
✅ Gradient descent minimizes loss by following the gradient downhill
✅ Update rule: weight = weight - learning_rate × gradient  
✅ Learning rate is the most important hyperparameter
✅ Mini-batch GD is the standard (batch size 32-256)
✅ Adam optimizer is the modern default choice
✅ Training = repeating gradient descent thousands of times
```

---

## 📊 Quick Reference Guide

### Gradient Descent Variants Comparison

| Type | Batch Size | Memory | Speed | Noise | Best For |
|------|-----------|--------|-------|-------|----------|
| **Batch GD** | All data | High | Slow | None | Small datasets (<10K) |
| **Stochastic GD** | 1 | Low | Fast | High | Online learning |
| **Mini-Batch GD** | 32-256 | Medium | **Optimal** | Low | **Standard choice** ✅ |

### Optimizer Recommendations

| Optimizer | Learning Rate | When to Use | Pros | Cons |
|-----------|--------------|-------------|------|------|
| **SGD** | 0.01-0.1 | Simple problems | Simple, well-understood | Slow convergence |
| **SGD + Momentum** | 0.01-0.1 | General use | Faster than SGD | Still slower than Adam |
| **Adam** | 0.001 | **Default choice** ✅ | Fast, adaptive | Slightly more memory |
| **AdamW** | 0.001 | Fine-tuning | Better regularization | - |

### Typical Hyperparameters

| Parameter | Range | Recommended Start | Notes |
|-----------|-------|-------------------|-------|
| **Learning Rate** | 0.0001 - 0.1 | **0.001** | Most critical parameter |
| **Batch Size** | 16 - 512 | **32** | Power of 2 for GPU |
| **Momentum (β₁)** | 0.8 - 0.99 | **0.9** | For SGD with momentum |
| **Adam β₂** | 0.9 - 0.9999 | **0.999** | Second moment |
| **Gradient Clip** | 1.0 - 10.0 | **5.0** | For RNNs |

### Common Issues Troubleshooting

| Problem | Symptoms | Solution |
|---------|----------|----------|
| 🔴 **Loss exploding** | NaN, Inf values | ↓ Reduce learning rate by 10x |
| 🟡 **Loss not decreasing** | Flat line | ↑ Increase learning rate |
| 🟡 **Loss oscillating** | Zigzag pattern | ↓ Reduce learning rate or add momentum |
| 🔵 **Very slow training** | Tiny improvements | ↑ Increase learning rate or change optimizer |
| 🟣 **Overfitting** | Train ↓ Val ↑ | Add regularization, dropout |

---

## 💡 Pro Tips

1. **Start with Adam**: It works out of the box for 90% of problems
2. **Learning rate is king**: Spend time tuning this first
3. **Plot your losses**: Always visualize training curves
4. **Use learning rate schedules**: Decay over time for better convergence
5. **Batch size matters**: Larger batches = more stable but slower
6. **Gradient clipping**: Essential for RNNs to prevent explosions

---

## 🚀 Next Lesson

**Lesson 5: Backpropagation** - How Networks Calculate Gradients

You'll learn:
- 🧮 Chain rule in practice
- 🔄 Backward pass implementation
- 🎯 Computing gradients efficiently
- 📊 Computational graphs

**Get ready for the math!** This is where it all comes together! 💪

---

## 📚 Additional Resources

- 📺 [3Blue1Brown: Gradient Descent](https://www.youtube.com/watch?v=IHZwWFHWa-w) - Best visualization
- 📺 [StatQuest: Gradient Descent](https://www.youtube.com/watch?v=sDv4f4s2SB8) - Clear explanation
- 📄 [Adam Paper](https://arxiv.org/abs/1412.6980) - Original algorithm
- 💻 [Interactive Demo](https://playground.tensorflow.org) - Visual learning

---

*⏱️ Estimated time: 35 minutes | 📊 Difficulty: Intermediate | ✅ Completion: Add to your progress*
