---
title: Training Best Practices & Optimization
description: >-
  Master techniques to train neural networks effectively - learning rates,
  initialization, normalization, and debugging
duration: 40 min
difficulty: intermediate
has_code: true
module: module-05
youtube: 'https://www.youtube.com/watch?v=pZEHXsizR7I'
objectives:
  - Choose appropriate learning rate schedules
  - Implement weight initialization strategies
  - Use batch normalization
  - Debug training issues
---
# Training Best Practices & Optimization

## The Art of Training Neural Networks

Building a network is easy. **Training it well** is hard!

This lesson covers battle-tested techniques to:
- ✅ Train faster
- ✅ Achieve better accuracy
- ✅ Debug problems
- ✅ Avoid common pitfalls

---

## 1. Weight Initialization

**Wrong initialization → Network won't train!**

### ❌ All Zeros

```python
W = np.zeros((100, 100))
```

**Problem**: All neurons learn the same thing (symmetry problem).

---

### ❌ Too Large

```python
W = np.random.randn(100, 100) * 10  # Too big!
```

**Problem**: Activations explode or saturate.

---

### ✅ Xavier/Glorot Initialization

**For Tanh/Sigmoid**:

```python
def xavier_init(n_in, n_out):
    """Xavier initialization"""
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, (n_in, n_out))

W = xavier_init(input_size, hidden_size)
```

**Why**: Keeps variance consistent across layers.

---

### ✅ He Initialization

**For ReLU**:

```python
def he_init(n_in, n_out):
    """He initialization (for ReLU)"""
    std = np.sqrt(2 / n_in)
    return np.random.randn(n_in, n_out) * std

W = he_init(input_size, hidden_size)
```

**Rule of thumb**:
- ReLU → He initialization
- Tanh/Sigmoid → Xavier initialization

---

## 2. Batch Normalization

**Problem**: Internal covariate shift - distribution of layer inputs changes during training.

**Solution**: Normalize activations!

### How It Works:

```
For each mini-batch:
1. Calculate mean μ and variance σ²
2. Normalize: x̂ = (x - μ) / √(σ² + ε)
3. Scale and shift: y = γx̂ + β
```

Where γ and β are learnable parameters.

### Implementation:

```python
class BatchNorm:
    """Batch Normalization layer"""
    
    def __init__(self, num_features, epsilon=1e-5, momentum=0.9):
        self.epsilon = epsilon
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = np.ones((1, num_features))
        self.beta = np.zeros((1, num_features))
        
        # Running statistics (for inference)
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))
    
    def forward(self, X, training=True):
        """
        Forward pass
        
        Args:
            X: Input (batch_size, num_features)
            training: Whether in training mode
        
        Returns:
            Normalized output
        """
        if training:
            # Calculate batch statistics
            batch_mean = np.mean(X, axis=0, keepdims=True)
            batch_var = np.var(X, axis=0, keepdims=True)
            
            # Normalize
            X_norm = (X - batch_mean) / np.sqrt(batch_var + self.epsilon)
            
            # Update running statistics
            self.running_mean = (self.momentum * self.running_mean + 
                               (1 - self.momentum) * batch_mean)
            self.running_var = (self.momentum * self.running_var + 
                              (1 - self.momentum) * batch_var)
            
            # Cache for backprop
            self.cache = (X, X_norm, batch_mean, batch_var)
        else:
            # Use running statistics at inference
            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        
        # Scale and shift
        out = self.gamma * X_norm + self.beta
        
        return out


# Usage in network
class NetworkWithBatchNorm:
    def __init__(self):
        self.W1 = he_init(784, 256)
        self.bn1 = BatchNorm(256)
        self.W2 = he_init(256, 10)
    
    def forward(self, X, training=True):
        # Layer 1
        Z1 = X @ self.W1
        Z1_norm = self.bn1.forward(Z1, training=training)
        A1 = np.maximum(0, Z1_norm)  # ReLU
        
        # Layer 2
        Z2 = A1 @ self.W2
        
        return Z2
```

### Benefits:
- ✅ Faster training (higher learning rates)
- ✅ Less sensitive to initialization
- ✅ Acts as regularization
- ✅ Improved gradient flow

---

## 3. Learning Rate Schedules

**Fixed learning rate** often suboptimal.

### Step Decay

Reduce LR by factor every N epochs:

```python
def step_decay(initial_lr, epoch, drop_every=10, drop_rate=0.5):
    """
    Example: 0.1 → 0.05 → 0.025 → 0.0125
    """
    return initial_lr * (drop_rate ** (epoch // drop_every))

# Usage
for epoch in range(100):
    lr = step_decay(initial_lr=0.1, epoch=epoch)
    # Train with this learning rate
```

---

### Exponential Decay

Smooth exponential decrease:

```python
def exp_decay(initial_lr, epoch, decay_rate=0.95):
    """
    lr = initial_lr * decay_rate^epoch
    """
    return initial_lr * (decay_rate ** epoch)
```

---

### Cosine Annealing

Smooth cosine curve:

```python
def cosine_annealing(initial_lr, epoch, total_epochs):
    """
    Follows cosine curve from initial_lr to 0
    """
    return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))
```

---

### Warm Restarts

Periodically reset to high LR:

```python
def cosine_with_restarts(initial_lr, epoch, restart_period=50):
    """
    Cosine annealing with periodic restarts
    """
    epoch_in_cycle = epoch % restart_period
    return cosine_annealing(initial_lr, epoch_in_cycle, restart_period)
```

**Why**: Helps escape local minima!

---

### 🔥 One Cycle Policy (Popular!)

Used by fast.ai, Jeremy Howard:

```
Phase 1 (50%): Increase LR from low → high
Phase 2 (50%): Decrease LR from high → very low
```

```python
class OneCycleLR:
    def __init__(self, max_lr, total_steps):
        self.max_lr = max_lr
        self.total_steps = total_steps
    
    def get_lr(self, step):
        if step < self.total_steps / 2:
            # Increasing phase
            return (self.max_lr / 10) + (step / (self.total_steps / 2)) * (self.max_lr * 0.9)
        else:
            # Decreasing phase
            progress = (step - self.total_steps / 2) / (self.total_steps / 2)
            return self.max_lr * (1 - 0.9 * progress)

# Usage
scheduler = OneCycleLR(max_lr=0.1, total_steps=1000)

for step in range(1000):
    lr = scheduler.get_lr(step)
    # Train with this LR
```

---

## 4. Advanced Optimizers

### Adam (Default Choice 2024)

Combines momentum + adaptive learning rates:

```python
class AdamOptimizer:
    """Adam optimizer"""
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        
        self.m = {}  # First moment
        self.v = {}  # Second moment
        self.t = 0   # Time step
    
    def update(self, params, grads):
        """
        Update parameters
        
        Args:
            params: Dictionary of parameters
            grads: Dictionary of gradients
        """
        self.t += 1
        
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            
            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            
            # Update biased second moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            
            # Bias correction
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


# Usage
optimizer = AdamOptimizer(learning_rate=0.001)

for epoch in range(100):
    # Forward + backward pass
    grads = compute_gradients()
    
    # Update with Adam
    optimizer.update(params, grads)
```

---

### AdamW (Adam with Weight Decay)

**Better regularization**:

```python
# In update step, add weight decay:
params[key] -= learning_rate * weight_decay * params[key]
```

**Recommended**: Use AdamW instead of Adam + L2 regularization!

---

## 5. Gradient Clipping

Prevent exploding gradients (especially in RNNs):

```python
def clip_gradients(gradients, max_norm=5.0):
    """
    Clip gradients by global norm
    
    Args:
        gradients: Dictionary of gradients
        max_norm: Maximum allowed norm
    
    Returns:
        Clipped gradients
    """
    # Calculate global norm
    total_norm = 0
    for grad in gradients.values():
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)
    
    # Clip if necessary
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for key in gradients:
            gradients[key] *= clip_coef
    
    return gradients


# Usage
grads = compute_gradients()
grads = clip_gradients(grads, max_norm=5.0)
optimizer.update(params, grads)
```

---

## 6. Debugging Training

### Problem: Loss Not Decreasing

**Possible causes**:

1. **Learning rate too high**
   - Solution: Reduce by 10x

2. **Learning rate too low**
   - Solution: Increase by 10x

3. **Bad initialization**
   - Solution: Use He/Xavier init

4. **Wrong loss function**
   - Solution: Check your task

5. **Bug in code**
   - Solution: Start simple, add complexity gradually

---

### Problem: Loss Exploding (NaN)

**Causes**:
- Learning rate too high
- Gradient explosion
- Numerical instability

**Solutions**:
```python
# 1. Lower learning rate
lr = lr / 10

# 2. Gradient clipping
grads = clip_gradients(grads, max_norm=1.0)

# 3. Check for NaN
if np.isnan(loss):
    print("NaN detected! Stopping...")
    break
```

---

### Problem: Overfitting

**Signs**:
- Train accuracy >> Val accuracy
- Train loss << Val loss

**Solutions**:
1. More data
2. Data augmentation
3. Dropout
4. L2 regularization
5. Early stopping
6. Smaller network

---

### Problem: Underfitting

**Signs**:
- Both train and val accuracy low
- Loss not decreasing

**Solutions**:
1. Larger network
2. Train longer
3. Better features
4. Reduce regularization

---

## 7. Complete Training Pipeline

```python
import numpy as np

class Trainer:
    """Complete training pipeline"""
    
    def __init__(self, model, optimizer, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(self, X_train, y_train, batch_size=32):
        """Train for one epoch"""
        num_samples = X_train.shape[0]
        indices = np.random.permutation(num_samples)
        
        epoch_loss = 0
        epoch_correct = 0
        
        for start_idx in range(0, num_samples, batch_size):
            # Get batch
            batch_indices = indices[start_idx:start_idx + batch_size]
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            # Forward pass
            predictions = self.model.forward(X_batch, training=True)
            loss = self.model.compute_loss(y_batch, predictions)
            
            # Backward pass
            grads = self.model.backward(X_batch, y_batch)
            
            # Clip gradients
            grads = clip_gradients(grads, max_norm=5.0)
            
            # Update
            self.optimizer.update(self.model.params, grads)
            
            # Track metrics
            epoch_loss += loss * len(X_batch)
            epoch_correct += np.sum(np.argmax(predictions, axis=1) == y_batch)
        
        epoch_loss /= num_samples
        epoch_acc = epoch_correct / num_samples
        
        return epoch_loss, epoch_acc
    
    def validate(self, X_val, y_val):
        """Validation"""
        predictions = self.model.forward(X_val, training=False)
        loss = self.model.compute_loss(y_val, predictions)
        acc = np.mean(np.argmax(predictions, axis=1) == y_val)
        return loss, acc
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100):
        """Full training loop"""
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc = self.train_epoch(X_train, y_train)
            
            # Validate
            val_loss, val_acc = self.validate(X_val, y_val)
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step(epoch)
            
            # Track history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | "
                      f"Train: {train_loss:.4f} ({train_acc:.4f}) | "
                      f"Val: {val_loss:.4f} ({val_acc:.4f})")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"
Early stopping at epoch {epoch}")
                break
        
        print("
✅ Training complete!")
```

---

## 🎯 Best Practices Checklist

- ✅ Use **He initialization** for ReLU
- ✅ Use **Batch Normalization** for deep networks
- ✅ Start with **Adam optimizer** (lr=0.001)
- ✅ Use **learning rate scheduling**
- ✅ Apply **gradient clipping** for RNNs
- ✅ Monitor **train vs val metrics**
- ✅ Use **early stopping**
- ✅ **Save best model** during training
- ✅ **Standardize inputs** (mean=0, std=1)
- ✅ Start simple, add complexity gradually

---

## 📹 Recommended Resources

- [CS231n: Training Neural Networks](https://www.youtube.com/watch?v=wEoyxE0GP2M)
- [fast.ai: Practical Deep Learning](https://www.youtube.com/watch?v=0oyCUWLL_fU)
- [Papers: Adam, BatchNorm, ResNet](https://paperswithcode.com/)

---

## 🎯 Key Takeaways

1. **Initialization matters**: Use He for ReLU, Xavier for Tanh
2. **Batch Norm** speeds training and improves accuracy
3. **Learning rate schedules** crucial for best performance
4. **Adam** is the default optimizer (2024)
5. **Gradient clipping** prevents explosions
6. **Monitor metrics** to debug issues
7. **Early stopping** prevents overfitting
8. Start simple, iterate, measure!

---

## 🎉 Module 01 Complete!

**Congratulations!** You've mastered:
- Neural network fundamentals
- Forward & backward propagation
- Activation functions & loss functions
- Gradient descent & backpropagation
- Regularization techniques
- CNNs for images
- RNNs/LSTMs for sequences
- Training best practices

---

## 🚀 Next Module

**Module 02: Large Language Models (LLMs)**
- Transformer architecture
- Attention mechanisms
- Pre-training & fine-tuning
- GPT, BERT, and beyond
- Building your own LLM applications

**Keep learning!** 💪

---

## 📹 Recommended Videos

- [How to Train Neural Networks](https://www.youtube.com/watch?v=pZEHXsizR7I) — Practical tips from fast.ai
- [Learning Rate Schedules Explained](https://www.youtube.com/watch?v=DE150MslZE0) — Cosine annealing, warm restarts
- [Batch Normalization Explained](https://www.youtube.com/watch?v=yXOMHOpbon8) — Visual intuition for batch norm

---

## 📚 Additional Resources

- [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) — Andrej Karpathy's practical guide
- [PyTorch Training Best Practices](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html) — Official tuning guide
- [Weight Initialization Strategies](https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/) — Machine Learning Mastery
