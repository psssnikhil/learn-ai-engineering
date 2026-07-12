---
title: 'Overfitting, Regularization & Dropout'
description: >-
  Master the bias-variance tradeoff, L1/L2 regularization theory, dropout
  mechanics, and early stopping — with implementation and production patterns
duration: 55 min
difficulty: intermediate
has_code: true
module: module-05
youtube: 'https://www.youtube.com/watch?v=EehRcPo1M-Q'
objectives:
  - Understand the bias-variance tradeoff mathematically
  - Derive and implement L1, L2, and elastic-net regularization
  - Implement inverted dropout from scratch with correct inference scaling
  - Apply early stopping with model checkpointing
  - Combine techniques using PyTorch best practices
---

# Overfitting, Regularization & Dropout

## Prerequisites

- [Lesson 04: Gradient Descent](./04-gradient-descent.md) — SGD, mini-batching, learning rates
- [Lesson 05: Backpropagation](./05-backpropagation.md) — gradient computation and weight updates
- Basic understanding of loss functions and model evaluation

## What You'll Learn

```
The fundamental tension:
  Underfitting ←————————————→ Overfitting
  (high bias)                  (high variance)
  Model too simple             Model too complex

Goal: Find the sweet spot → generalizes to new data
```

---

## 1. The Bias-Variance Tradeoff

Any model's expected test error can be decomposed into three terms:

```
Expected MSE = Bias² + Variance + Irreducible Noise

Bias²:     Error from wrong assumptions (underfitting)
Variance:  Error from sensitivity to training data (overfitting)
Noise:     Irreducible — comes from the data itself
```

```python
import numpy as np
import matplotlib.pyplot as plt


def generate_dataset(n: int = 30, noise: float = 0.3) -> tuple[np.ndarray, np.ndarray]:
    """
    True function: y = sin(x) + noise
    Goal: learn this from n samples.
    """
    x = np.linspace(0, 2 * np.pi, n)
    y = np.sin(x) + np.random.normal(0, noise, n)
    return x, y


def fit_polynomial(
    x_train: np.ndarray,
    y_train: np.ndarray,
    degree:  int,
) -> np.ndarray:
    """
    Fit polynomial of given degree.
    Returns coefficients from numpy.polyfit.
    """
    coeffs = np.polyfit(x_train, y_train, degree)
    return coeffs


def bias_variance_experiment(
    n_experiments: int = 100,
    n_train:       int = 20,
    degrees:       list[int] = [1, 3, 8, 15],
) -> None:
    """
    Run many experiments to empirically measure bias and variance.
    - High degree → low bias, high variance (overfitting)
    - Low degree  → high bias, low variance (underfitting)
    """
    x_test = np.linspace(0, 2 * np.pi, 100)
    y_true = np.sin(x_test)   # ground truth

    results = {}
    for degree in degrees:
        predictions = []
        for _ in range(n_experiments):
            x_train, y_train = generate_dataset(n_train)
            coeffs = fit_polynomial(x_train, y_train, degree)
            y_pred = np.polyval(coeffs, x_test)
            predictions.append(y_pred)

        predictions = np.array(predictions)   # (n_experiments, 100)

        # Bias: how far is the mean prediction from truth?
        mean_pred = predictions.mean(axis=0)
        bias_sq   = np.mean((mean_pred - y_true) ** 2)

        # Variance: how much do predictions vary across experiments?
        variance  = np.mean(predictions.var(axis=0))

        results[degree] = {"bias_sq": bias_sq, "variance": variance}
        print(f"Degree {degree:2d}: Bias²={bias_sq:.4f}, Variance={variance:.4f}, "
              f"Total={bias_sq + variance:.4f}")

    return results


# Example output:
# Degree  1: Bias²=0.4521, Variance=0.0012, Total=0.4533  ← underfitting
# Degree  3: Bias²=0.0034, Variance=0.0089, Total=0.0123  ← sweet spot
# Degree  8: Bias²=0.0012, Variance=0.1870, Total=0.1882  ← overfitting
# Degree 15: Bias²=0.0008, Variance=8.9021, Total=8.9029  ← severe overfitting
```

---

## 2. Identifying Overfitting

**Diagnostic**: plot training and validation loss curves.

```python
import numpy as np


def plot_learning_curves(
    train_losses: list[float],
    val_losses:   list[float],
) -> dict:
    """
    Analyze learning curves and identify overfitting regime.
    Returns: diagnostic dict.
    """
    train = np.array(train_losses)
    val   = np.array(val_losses)

    # Find the epoch where val loss is minimized
    best_epoch = int(np.argmin(val))
    best_val   = val[best_epoch]

    # Generalization gap at the end
    final_gap = val[-1] - train[-1]

    # Detect overfitting: val loss starts rising while train keeps falling
    overfit_epoch = None
    for i in range(1, len(val)):
        if val[i] > val[i - 1] and train[i] < train[i - 1]:
            overfit_epoch = i
            break

    return {
        "best_epoch":       best_epoch,
        "best_val_loss":    round(float(best_val), 4),
        "final_train_loss": round(float(train[-1]), 4),
        "final_val_loss":   round(float(val[-1]), 4),
        "generalization_gap": round(float(final_gap), 4),
        "overfit_starts_at":  overfit_epoch,
        "diagnosis": (
            "underfitting" if val[-1] > 0.3 and abs(final_gap) < 0.05 else
            "overfitting"  if final_gap > 0.1 else
            "good_fit"
        ),
    }
```

---

## 3. L2 Regularization (Weight Decay)

L2 regularization adds a penalty proportional to the squared magnitude of weights:

```
Loss_L2 = Loss_data + λ/2 × Σ_i w_i²
```

**Gradient effect**:
```
∂Loss_L2/∂w = ∂Loss_data/∂w + λ × w

Weight update:
  w ← w - α × (∂Loss_data/∂w + λ × w)
  w ← w × (1 - α × λ)  - α × ∂Loss_data/∂w   ← "weight decay"

Every step, weights are shrunk by factor (1 - α × λ).
With α=0.01, λ=0.01: shrink by 0.0001 per step — small but cumulative!
```

```python
import numpy as np


class L2RegularizedNetwork:
    """
    Two-layer network with L2 regularization.
    Loss = cross-entropy + λ/2 × (||W1||² + ||W2||²)
    """

    def __init__(
        self,
        input_size:  int,
        hidden_size: int,
        output_size: int,
        l2_lambda:   float = 0.01,
        lr:          float = 0.01,
    ):
        # He initialization for ReLU layers
        scale1 = np.sqrt(2.0 / input_size)
        scale2 = np.sqrt(2.0 / hidden_size)

        self.W1 = np.random.randn(input_size, hidden_size) * scale1   # (d_in, d_h)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * scale2  # (d_h, d_out)
        self.b2 = np.zeros(output_size)

        self.l2_lambda = l2_lambda
        self.lr = lr

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        """
        X: (B, d_in)
        Returns: (predictions, cache for backward pass)
        """
        Z1 = X @ self.W1 + self.b1         # (B, d_h)
        A1 = np.maximum(0, Z1)             # ReLU
        Z2 = A1 @ self.W2 + self.b2        # (B, d_out)
        # Softmax for classification
        exp_Z2 = np.exp(Z2 - Z2.max(axis=1, keepdims=True))
        A2 = exp_Z2 / exp_Z2.sum(axis=1, keepdims=True)  # (B, d_out)

        cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
        return A2, cache

    def loss(self, A2: np.ndarray, y: np.ndarray) -> float:
        """
        Cross-entropy loss + L2 penalty.
        y: (B,) integer class labels
        """
        B = len(y)
        # Cross-entropy
        log_probs = -np.log(A2[np.arange(B), y] + 1e-9)
        data_loss = log_probs.mean()

        # L2 penalty (λ/2 × Σw² for all weight matrices)
        l2_loss = (self.l2_lambda / 2) * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))

        return data_loss + l2_loss

    def backward(self, cache: dict, y: np.ndarray) -> None:
        """
        Backprop with L2 gradient added to weight gradients.
        """
        B = len(y)
        A2, A1, Z1, X = cache["A2"], cache["A1"], cache["Z1"], cache["X"]

        # Output gradient (softmax + cross-entropy combined)
        dZ2 = A2.copy()
        dZ2[np.arange(B), y] -= 1
        dZ2 /= B   # (B, d_out)

        # Gradients for W2, b2
        dW2 = A1.T @ dZ2 + self.l2_lambda * self.W2   # ← L2 gradient term!
        db2 = dZ2.sum(axis=0)

        # Backprop through A1 and Z1
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (Z1 > 0)   # ReLU gradient

        # Gradients for W1, b1
        dW1 = X.T @ dZ1 + self.l2_lambda * self.W1   # ← L2 gradient term!
        db1 = dZ1.sum(axis=0)

        # SGD update
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
```

**In PyTorch**: `weight_decay` parameter in the optimizer applies L2 regularization:
```python
import torch.optim as optim
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
# weight_decay = L2 λ coefficient
```

---

## 4. L1 Regularization (Lasso)

```
Loss_L1 = Loss_data + λ × Σ_i |w_i|

Gradient: ∂|w|/∂w = sign(w)  (1 if w > 0, -1 if w < 0, undefined at 0)
```

**Key difference from L2**: L1 drives some weights to **exactly zero** (sparse solution). L2 drives weights toward (but not to) zero.

```python
def l1_gradient(W: np.ndarray, l1_lambda: float) -> np.ndarray:
    """
    L1 regularization gradient: λ × sign(W).
    Returns a sparse gradient that drives weights to exactly zero.
    """
    return l1_lambda * np.sign(W)


def elastic_net_gradient(
    W:           np.ndarray,
    l1_lambda:   float,
    l2_lambda:   float,
) -> np.ndarray:
    """
    Elastic Net = L1 + L2 (best of both worlds).
    L2 handles correlated features; L1 induces sparsity.
    """
    return l1_lambda * np.sign(W) + l2_lambda * W


# Comparison: weights after 100 steps of gradient descent
def regularization_comparison(
    initial_weight: float = 2.0,
    lr:             float = 0.1,
    n_steps:        int = 50,
    lambda_val:     float = 0.05,
) -> dict[str, list[float]]:
    """
    Show how L1, L2, and ElasticNet differently shape weight trajectories.
    """
    w_l1 = w_l2 = w_enet = initial_weight
    history = {"L1": [w_l1], "L2": [w_l2], "ElasticNet": [w_enet]}

    for _ in range(n_steps):
        # Assume data gradient is zero (isolate regularization effect)
        w_l1   -= lr * (lambda_val * np.sign(w_l1))
        w_l2   -= lr * (lambda_val * w_l2)
        w_enet -= lr * (0.5 * lambda_val * np.sign(w_enet) + 0.5 * lambda_val * w_enet)

        history["L1"].append(w_l1)
        history["L2"].append(w_l2)
        history["ElasticNet"].append(w_enet)

    return history
# L1:      weights reach exactly 0 (sparse solution)
# L2:      weights approach 0 exponentially, never quite reach it
# ElasticNet: faster than L2 decay, reaches 0 like L1 but smoother
```

---

## 5. Dropout

Dropout (Srivastava et al. 2014) randomly sets activations to zero during training with probability `p`. This forces the network to learn redundant representations and prevents co-adaptation of neurons.

### Intuition: Ensemble of Models

At every training step, a different subnetwork is active. With `p=0.5`, a layer with 1024 neurons creates 2^1024 possible subnetworks. At inference, we use the full network as an approximation of the ensemble average.

### Inverted Dropout (Correct Implementation)

```python
import numpy as np


def inverted_dropout(
    A:          np.ndarray,   # (B, d) activations
    keep_prob:  float = 0.8,
    training:   bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Inverted dropout: scale activations DURING training (not during inference).

    Why inverted? Without scaling, expected activation during training ≠ inference.
    Inverted dropout fixes this: scale by 1/keep_prob during training,
    no adjustment needed at inference.

    Non-inverted (wrong approach):
      Train:     A * mask          (no scaling → mean activation decreases)
      Inference: A * keep_prob     (manual scaling needed at test time)

    Inverted (correct):
      Train:     A * mask / keep_prob   (scale up survivors → mean unchanged)
      Inference: A                       (no adjustment needed!)

    Returns: (dropped_activations, mask)
    """
    if not training:
        return A, None

    # mask[i] = 1 (keep) with probability keep_prob, 0 (drop) otherwise
    mask = (np.random.rand(*A.shape) < keep_prob).astype(float)

    # Scale surviving activations so expected value matches full network
    A_dropped = A * mask / keep_prob

    return A_dropped, mask


class NeuralNetWithDropout:
    """
    Two-layer network with inverted dropout.
    Dropout is applied after each hidden layer's activation.
    """

    def __init__(
        self,
        input_size:  int,
        hidden_size: int,
        output_size: int,
        keep_prob:   float = 0.8,
        lr:          float = 0.01,
    ):
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(output_size)
        self.keep_prob = keep_prob
        self.lr = lr

    def forward(self, X: np.ndarray, training: bool = True) -> tuple[np.ndarray, dict]:
        Z1 = X @ self.W1 + self.b1         # (B, d_h)
        A1_raw = np.maximum(0, Z1)         # ReLU

        A1, mask1 = inverted_dropout(A1_raw, self.keep_prob, training)

        Z2 = A1 @ self.W2 + self.b2
        exp_Z2 = np.exp(Z2 - Z2.max(axis=1, keepdims=True))
        A2 = exp_Z2 / exp_Z2.sum(axis=1, keepdims=True)

        return A2, {"X": X, "Z1": Z1, "A1": A1, "A1_raw": A1_raw,
                    "mask1": mask1, "Z2": Z2, "A2": A2}

    def backward(self, cache: dict, y: np.ndarray) -> None:
        B = len(y)
        X, Z1, A1, mask1 = cache["X"], cache["Z1"], cache["A1"], cache["mask1"]
        A2 = cache["A2"]

        dZ2 = A2.copy()
        dZ2[np.arange(B), y] -= 1
        dZ2 /= B

        dW2 = A1.T @ dZ2
        db2 = dZ2.sum(axis=0)

        dA1 = dZ2 @ self.W2.T

        # Apply dropout mask to gradient
        if mask1 is not None:
            dA1 = dA1 * mask1 / self.keep_prob   # same scaling as forward pass

        dZ1 = dA1 * (Z1 > 0)   # ReLU gradient

        dW1 = X.T @ dZ1
        db1 = dZ1.sum(axis=0)

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
```

**Dropout in PyTorch**:
```python
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout = nn.Dropout(p=0.5)  # drop 50% of neurons
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)    # drops during train, identity during eval
        return self.fc2(x)

# Important: call model.eval() for inference, model.train() for training!
model.eval()   # dropout becomes identity (no dropping)
model.train()  # dropout active
```

---

## 6. Early Stopping with Checkpointing

```python
import copy


class EarlyStopping:
    """
    Monitor validation loss, stop when it stops improving.
    Saves the best model state (checkpointing).

    patience:     how many epochs to wait after last improvement
    min_delta:    minimum change to qualify as improvement
    restore_best: if True, load best weights when stopping
    """

    def __init__(
        self,
        patience:     int   = 10,
        min_delta:    float = 1e-4,
        restore_best: bool  = True,
    ):
        self.patience     = patience
        self.min_delta    = min_delta
        self.restore_best = restore_best

        self.counter      = 0
        self.best_loss    = float("inf")
        self.best_state   = None   # checkpointed model weights
        self.stopped_epoch = None

    def step(self, val_loss: float, model_state: dict) -> bool:
        """
        Returns True if training should stop.
        model_state: can be model.state_dict() for PyTorch, or a copy of numpy weights.
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement!
            self.best_loss  = val_loss
            self.counter    = 0
            self.best_state = copy.deepcopy(model_state)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True   # signal to stop

        return False

    def load_best(self, model) -> None:
        """Restore the best checkpoint into model."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def training_loop_with_early_stopping(
    model,
    optimizer,
    train_loader,
    val_loader,
    max_epochs:  int = 200,
    patience:    int = 15,
) -> dict:
    """Full PyTorch training loop with early stopping."""
    import torch
    import torch.nn.functional as F

    early_stop = EarlyStopping(patience=patience, restore_best=True)
    train_losses, val_losses = [], []

    for epoch in range(max_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x)
                val_loss += F.cross_entropy(logits, y).item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: train={train_losses[-1]:.4f}, val={val_loss:.4f}")

        if early_stop.step(val_loss, model.state_dict()):
            print(f"\nEarly stopping at epoch {epoch} (best val: {early_stop.best_loss:.4f})")
            break

    # Restore best weights
    early_stop.load_best(model)
    return {"train_losses": train_losses, "val_losses": val_losses}
```

---

## 7. Batch Normalization

Batch normalization normalizes layer inputs to zero mean and unit variance, then allows the network to learn a scale (γ) and shift (β):

```
μ_B = (1/m) Σ x_i          (batch mean)
σ²_B = (1/m) Σ (x_i - μ_B)² (batch variance)
x̂_i = (x_i - μ_B) / √(σ²_B + ε)   (normalize)
y_i = γ × x̂_i + β                   (scale and shift)
```

```python
import numpy as np


class BatchNorm1D:
    """
    Batch normalization for 2D inputs (B, d).
    Tracks running statistics for inference.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        self.gamma   = np.ones(num_features)    # learnable scale
        self.beta    = np.zeros(num_features)   # learnable shift
        self.eps     = eps
        self.momentum = momentum

        # Running statistics for inference (updated during training)
        self.running_mean = np.zeros(num_features)
        self.running_var  = np.ones(num_features)

        self.cache = {}

    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        x: (B, d) — batch of feature vectors
        """
        if training:
            mu  = x.mean(axis=0)        # (d,)
            var = x.var(axis=0)         # (d,)

            x_hat = (x - mu) / np.sqrt(var + self.eps)   # (B, d) normalized
            out   = self.gamma * x_hat + self.beta

            # Save for backward pass
            self.cache = {"x": x, "mu": mu, "var": var, "x_hat": x_hat}

            # Update running stats (used at inference)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
        else:
            # Use running statistics at inference — no batch statistics
            x_hat = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out   = self.gamma * x_hat + self.beta

        return out
```

**Key insight**: BatchNorm acts as a mild regularizer because the normalization adds noise (batch statistics are noisy approximations of population statistics). This is why dropout is sometimes unnecessary when using BatchNorm.

---

## Regularization Strategy Guide

```
Start with:
  1. L2 weight decay (λ = 1e-4)  — always safe
  2. Early stopping (patience = 10-20 epochs)

Add if still overfitting:
  3. Dropout (p = 0.3–0.5) — for fully-connected layers
  4. Batch normalization — for deep networks
  5. Data augmentation — for vision/audio tasks

Consider L1 or ElasticNet:
  - When you need feature selection (NLP, tabular data)
  - Sparse model interpretability required
```

| Technique | Hyperparameter | Search range | Effect |
|-----------|---------------|-------------|--------|
| L2 | λ | 1e-5 to 1e-1 | Reduces large weights |
| L1 | λ | 1e-5 to 1e-1 | Sparsifies weights |
| Dropout | keep_prob | 0.5 to 0.9 | Ensemble effect |
| Early stopping | patience | 5 to 30 | Optimal epoch |
| Batch Norm | momentum | 0.01 to 0.1 | Normalizes activations |

---

## Edge Cases & Misconceptions

!!! warning "Misconception: More dropout = more regularization = always better"
    High dropout (keep_prob < 0.5) dramatically slows training because fewer neurons receive gradient updates. For deep networks, use p=0.1–0.2 per layer rather than p=0.5 on all layers. In modern architectures like Transformers, very low dropout (p=0.1) is standard.

!!! note "Dropout and BatchNorm don't mix well"
    Using dropout after BatchNorm layers can cause training-inference inconsistency because the variance seen by BatchNorm changes between training and inference. If using BatchNorm, prefer not using dropout in the same sub-layer, or use dropout after the final normalization.

!!! warning "Misconception: Early stopping always finds the best model"
    Early stopping with insufficient patience can stop before the model reaches optimal loss — especially on difficult tasks where validation loss can temporarily plateau before improving. Monitor the validation curve shape, not just patience count.

---

## Production Connection

**Large language model training**: L2 weight decay (λ=0.1) is used in most large-scale training runs (GPT-3, LLaMA). Dropout in Transformer models is typically very low (p=0.1) or absent in modern architectures. Early stopping is less common at LLM scale — instead, models are trained for a fixed compute budget and evaluated on benchmarks.

**Practical tips**:
- In PyTorch: `torch.optim.AdamW` implements decoupled weight decay (recommended over Adam + L2).
- Monitor the ratio `val_loss / train_loss` — values consistently above 1.2 indicate overfitting.
- Use Weights & Biases or MLflow to track regularization experiments across runs.

---

## Key Takeaways

1. **Bias-variance tradeoff**: all model error decomposes into Bias² + Variance + Noise — regularization reduces variance at the cost of slight bias.
2. **L2** shrinks all weights exponentially ("weight decay"); **L1** drives some weights to exactly zero (sparse).
3. **Inverted dropout** scales activations by 1/keep_prob during training — no scaling adjustment needed at inference. Always call `model.eval()` for inference.
4. **Early stopping** with checkpointing: save the model at the best validation loss, not at the last epoch.
5. **BatchNorm** is a mild regularizer as a side effect of normalization noise during training.
6. Combine techniques: start with L2 + early stopping, add dropout if still overfitting.

---

## Further Reading

- [Dropout paper](https://jmlr.org/papers/v15/srivastava14a.html) — Srivastava et al. 2014: original dropout formulation
- [BatchNorm paper](https://arxiv.org/abs/1502.03167) — Ioffe & Szegedy 2015
- [Adam vs AdamW](https://arxiv.org/abs/1711.05101) — Decoupled Weight Decay Regularization
- [Regularization in Deep Learning](https://www.deeplearningbook.org/contents/regularization.html) — Goodfellow et al., Chapter 7

---

## 🚀 Next Lesson

**[Lesson 7: Building a Neural Network from Scratch](./07-building-nn-from-scratch.md)** — implement a complete two-layer network in pure NumPy with forward pass, backpropagation, and training loop.
