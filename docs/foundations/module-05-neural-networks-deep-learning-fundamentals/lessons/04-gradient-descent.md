---
title: Gradient Descent — How Neural Networks Learn
description: >-
  Understand gradient descent from first principles, trace a full numerical
  example from loss to weight update, understand learning rate effects,
  mini-batches, and the variants used in modern deep learning
duration: 90 min
difficulty: beginner
has_code: true
module: module-05
---
# Gradient Descent — How Neural Networks Learn

## Prerequisites

- [Lesson 02: Neurons & Activation Functions](02-neurons-activation-functions.md)
- [Lesson 03: Loss Functions](03-loss-functions.md)
- [Module 00 Lesson 02: Math Foundations](../../module-00-genai-foundations-from-nlp-to-transformers/lessons/02-math-foundations.md) — gradients, partial derivatives

## What You'll Learn

| Objective | Why It Matters |
|-----------|---------------|
| Understand the gradient as a direction, not a value | Prerequisite for all of modern ML |
| Trace a complete gradient descent update numerically | Required before understanding backpropagation |
| Understand learning rate effects precisely | The most impactful hyperparameter in practice |
| Distinguish batch, mini-batch, and stochastic GD | SGD-based training is used in all modern models |
| Understand Adam vs. SGD | Adam is the default for transformers; SGD for CNNs |

---

## The Core Problem

A neural network has millions of parameters (weights \(W\), biases \(b\)). Training means finding values for all parameters such that the loss \(\mathcal{L}\) is minimized. The space of parameters is high-dimensional — GPT-2 small has 117 million parameters.

We cannot enumerate all possible weights. We need an algorithm that efficiently navigates this high-dimensional space toward the minimum.

**Gradient descent** is that algorithm.

---

## Intuition: The Hillwalker Analogy

Imagine you are on a foggy mountain. You want to reach the lowest valley. You cannot see far, but you can feel the slope beneath your feet.

- The **slope** at your position is the **gradient** of the loss with respect to your current parameters
- You take a **step downhill** — in the direction opposite to the gradient
- You repeat until the slope is flat — you've reached a local minimum

Mathematically, the gradient \(\nabla_\theta \mathcal{L}\) is a vector that points in the direction of steepest *increase* of the loss. We step in the opposite direction:

\[
\theta \leftarrow \theta - \alpha \nabla_\theta \mathcal{L}
\]

where \(\alpha\) is the **learning rate** — the step size.

---

## Gradient of the Loss: One-Dimensional Example

Let us start simple. Suppose the loss is a function of a single weight \(w\):

\[
\mathcal{L}(w) = (w - 3)^2 \quad \text{(minimum at } w=3 \text{)}
\]

The gradient (derivative) is:

\[
\frac{d\mathcal{L}}{dw} = 2(w - 3)
\]

```python
import numpy as np

def gradient_descent_1d():
    """
    Minimize L(w) = (w - 3)^2 using gradient descent.
    Trace each step to build intuition.
    """
    def loss(w):       return (w - 3) ** 2
    def grad_loss(w):  return 2 * (w - 3)

    w = 8.0          # initial weight — far from minimum at w=3
    lr = 0.1         # learning rate
    steps = 20

    print(f"{'Step':5} {'w':8} {'L(w)':8} {'Gradient':10} {'Update':10}")
    print("-" * 50)

    for step in range(steps):
        L    = loss(w)
        grad = grad_loss(w)
        update = -lr * grad

        print(f"{step:5d} {w:8.4f} {L:8.4f} {grad:10.4f} {update:10.4f}")

        w = w + update   # gradient descent step

        if abs(grad) < 0.001:
            print(f"\nConverged at step {step}! w = {w:.4f}")
            break

gradient_descent_1d()
```

Each step:
1. Compute loss at current \(w\)
2. Compute gradient — the slope at current \(w\)
3. Step opposite to gradient by \(\alpha \times \text{gradient}\)

---

## The Learning Rate: Critical Hyperparameter

```python
import numpy as np

def learning_rate_comparison():
    """
    Show the effect of different learning rates on convergence.
    L(w) = (w - 3)^2, starting at w=0.
    """
    def loss(w):      return (w - 3) ** 2
    def grad_loss(w): return 2 * (w - 3)

    learning_rates = {
        "Too small (lr=0.001)": 0.001,
        "Good (lr=0.1)":        0.1,
        "Too large (lr=0.95)":  0.95,
        "Diverges (lr=1.5)":    1.5,
    }

    for name, lr in learning_rates.items():
        w = 0.0
        trajectory = [w]
        for _ in range(50):
            grad = grad_loss(w)
            w = w - lr * grad
            trajectory.append(w)
            if abs(w) > 100:   # diverged
                break
        final_w = trajectory[-1]
        print(f"{name}: final w = {final_w:.4f}  L = {loss(final_w):.6f}")

learning_rate_comparison()
# Too small: slow convergence, still far from minimum after 50 steps
# Good: converges quickly to w ≈ 3.0
# Too large: oscillates but may converge
# Diverges: explodes to ±∞
```

!!! warning "Learning Rate is the Most Important Hyperparameter"
    If your model doesn't train, check the learning rate first. Common starting points: `1e-3` for Adam, `1e-1` for SGD with momentum. Transformers typically use `1e-4` to `3e-4` with warmup and decay schedules.

---

## Gradient Descent in Multiple Dimensions

With multiple parameters, the gradient is a vector of partial derivatives — one per parameter:

\[
\nabla_\theta \mathcal{L} = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial w_1} \\ \frac{\partial \mathcal{L}}{\partial w_2} \\ \vdots \end{bmatrix}
\]

```python
import numpy as np

def gradient_descent_2d():
    """
    Minimize L(w1, w2) = w1^2 + 4*w2^2 — an elliptical bowl.
    Minimum is at (w1, w2) = (0, 0).
    """
    def loss(w1, w2):          return w1**2 + 4 * w2**2
    def grad(w1, w2):          return np.array([2*w1, 8*w2])

    params = np.array([3.0, 2.0])   # start at (3, 2)
    lr = 0.1

    print(f"{'Step':5} {'w1':8} {'w2':8} {'Loss':10}")
    print("-" * 40)

    for step in range(15):
        L = loss(*params)
        g = grad(*params)
        print(f"{step:5d} {params[0]:8.4f} {params[1]:8.4f} {L:10.4f}")
        params = params - lr * g

    print(f"\nFinal: w1={params[0]:.6f}  w2={params[1]:.6f}  (should be near 0, 0)")

gradient_descent_2d()
```

Notice: convergence is slower for \(w_2\) — the gradient is larger there (coefficient 4 vs 1 for \(w_1\)) so overshooting is more likely. This asymmetry motivates **adaptive learning rates** like Adam.

---

## Batch, Mini-Batch, and Stochastic Gradient Descent

In practice, computing the gradient over the entire dataset before each update (full batch GD) is prohibitively expensive. The variants:

| Variant | Batch Size | Update Frequency | Noise | Used When |
|---------|-----------|-----------------|-------|-----------|
| Full Batch GD | All N examples | Once per epoch | Low | Small datasets only |
| Mini-Batch SGD | B examples (32-256) | B updates per epoch | Medium | Standard practice |
| Stochastic GD | 1 example | N updates per epoch | High | Online learning |

```python
import numpy as np

def compare_gradient_estimates():
    """
    Show how gradient estimates differ with different batch sizes.
    Task: linear regression, L(w) = (1/N) sum (wx_i - y_i)^2
    True gradient with all data vs. noisy estimate with 1 example.
    """
    np.random.seed(42)
    N  = 1000
    X  = np.random.randn(N)           # features
    y  = 2.5 * X + 0.3 + np.random.randn(N) * 0.5   # true w=2.5, noise

    w = 0.0   # initialize

    def batch_gradient(w, X_b, y_b):
        preds = w * X_b
        return 2 * np.mean((preds - y_b) * X_b)

    # True gradient (full data)
    true_grad = batch_gradient(w, X, y)
    print(f"True gradient (N={N}):            {true_grad:.6f}")

    # Mini-batch estimates (average over 5 random mini-batches of size 32)
    mb_grads = []
    for _ in range(5):
        idx = np.random.choice(N, 32, replace=False)
        mb_grads.append(batch_gradient(w, X[idx], y[idx]))
    print(f"Mini-batch gradient (B=32, mean): {np.mean(mb_grads):.6f}  std: {np.std(mb_grads):.6f}")

    # Stochastic estimates (single examples)
    sg_grads = [batch_gradient(w, X[i:i+1], y[i:i+1]) for i in range(5)]
    print(f"Stochastic gradient (B=1, mean):  {np.mean(sg_grads):.6f}  std: {np.std(sg_grads):.6f}")
    print("\nNote: mini-batch is unbiased but noisy; noise helps escape local minima")

compare_gradient_estimates()
```

**Why mini-batch works so well:**
1. Noisy gradients help escape sharp local minima and saddle points
2. Parallelizes perfectly on GPUs — process B examples simultaneously
3. Memory efficient — no need to load all N examples at once

---

## Numerical Complete Example: One Gradient Step in a Neural Network

A full end-to-end step: forward pass → compute loss → compute gradient → update weight.

```python
import numpy as np

def full_gradient_step():
    """
    One complete gradient descent step for a simple network.
    Input: 2 features. One hidden neuron (ReLU). One output (linear). Loss: MSE.
    We manually compute everything to see exactly what happens.
    """
    # Data: one example
    x = np.array([1.0, 0.5])   # input
    y = 2.0                     # target

    # Parameters (initialized)
    w1 = np.array([0.3, -0.5])  # weights of hidden neuron (shape 2)
    b1 = 0.1                    # bias of hidden neuron
    w2 = np.array([0.8])        # weight from hidden to output
    b2 = 0.2                    # output bias

    lr = 0.01

    print("=== Forward Pass ===")
    # Hidden neuron pre-activation
    z1 = np.dot(w1, x) + b1
    print(f"z1 = w1·x + b1 = {np.dot(w1, x):.4f} + {b1} = {z1:.4f}")

    # Hidden neuron activation (ReLU)
    a1 = max(0, z1)
    relu_grad = 1.0 if z1 > 0 else 0.0   # ReLU derivative
    print(f"a1 = ReLU({z1:.4f}) = {a1:.4f}  (ReLU'={relu_grad})")

    # Output (linear)
    y_hat = w2[0] * a1 + b2
    print(f"ŷ = w2*a1 + b2 = {w2[0]} * {a1:.4f} + {b2} = {y_hat:.4f}")

    # Loss (MSE for single example)
    loss = (y_hat - y) ** 2
    print(f"\nLoss = (ŷ - y)^2 = ({y_hat:.4f} - {y})^2 = {loss:.4f}")

    print("\n=== Backward Pass (chain rule) ===")
    # d(Loss)/d(ŷ)
    dL_dyhat = 2 * (y_hat - y)
    print(f"∂L/∂ŷ = 2(ŷ - y) = 2({y_hat:.4f} - {y}) = {dL_dyhat:.4f}")

    # d(Loss)/d(w2) via chain rule
    dL_dw2 = dL_dyhat * a1
    print(f"∂L/∂w2 = ∂L/∂ŷ · ∂ŷ/∂w2 = {dL_dyhat:.4f} * {a1:.4f} = {dL_dw2:.4f}")

    # d(Loss)/d(b2)
    dL_db2 = dL_dyhat * 1   # ∂ŷ/∂b2 = 1
    print(f"∂L/∂b2 = ∂L/∂ŷ · 1 = {dL_db2:.4f}")

    # d(Loss)/d(a1) = d(Loss)/d(ŷ) * w2
    dL_da1 = dL_dyhat * w2[0]
    print(f"∂L/∂a1 = ∂L/∂ŷ · w2 = {dL_dyhat:.4f} * {w2[0]} = {dL_da1:.4f}")

    # d(Loss)/d(z1) = d(Loss)/d(a1) * ReLU'(z1)
    dL_dz1 = dL_da1 * relu_grad
    print(f"∂L/∂z1 = ∂L/∂a1 · ReLU'({z1:.4f}) = {dL_da1:.4f} * {relu_grad} = {dL_dz1:.4f}")

    # d(Loss)/d(w1) = d(Loss)/d(z1) * x
    dL_dw1 = dL_dz1 * x
    print(f"∂L/∂w1 = ∂L/∂z1 · x = {dL_dz1:.4f} * {x} = {dL_dw1}")

    # d(Loss)/d(b1)
    dL_db1 = dL_dz1 * 1
    print(f"∂L/∂b1 = ∂L/∂z1 = {dL_db1:.4f}")

    print("\n=== Weight Update ===")
    w2_new = w2 - lr * np.array([dL_dw2])
    b2_new = b2 - lr * dL_db2
    w1_new = w1 - lr * dL_dw1
    b1_new = b1 - lr * dL_db1

    print(f"w2: {w2[0]:.4f} → {w2_new[0]:.4f}  (Δ = {w2_new[0]-w2[0]:.6f})")
    print(f"b2: {b2:.4f}  → {b2_new:.4f}  (Δ = {b2_new-b2:.6f})")
    print(f"w1: {w1} → {w1_new}  (Δ = {w1_new-w1})")
    print(f"b1: {b1:.4f}  → {b1_new:.4f}  (Δ = {b1_new-b1:.6f})")

    # Verify: new prediction is slightly closer to y=2.0
    z1_new = np.dot(w1_new, x) + b1_new
    a1_new = max(0, z1_new)
    yhat_new = w2_new[0] * a1_new + b2_new
    print(f"\nNew prediction: {yhat_new:.4f}  (was {y_hat:.4f}, target is {y})")
    print(f"Loss improved: {(yhat_new - y)**2:.4f} < {loss:.4f}")

full_gradient_step()
```

---

## Adam: Adaptive Moment Estimation

Vanilla gradient descent struggles with:
- Different parameters having different gradient scales (some gradients always large, some always small)
- Saddle points where gradients are near-zero

**Adam** addresses this with two moment estimates:

\[
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(first moment: exponential moving average of gradient)}
\]
\[
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(second moment: exponential moving average of gradient squared)}
\]
\[
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad \text{(bias correction)}
\]
\[
\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}
\]

Typical defaults: \(\beta_1 = 0.9\), \(\beta_2 = 0.999\), \(\varepsilon = 10^{-8}\), \(\alpha = 10^{-3}\).

```python
import numpy as np

class AdamOptimizer:
    """Minimal Adam implementation to understand the mechanics."""

    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.t     = 0     # step counter
        self.m     = None  # first moment
        self.v     = None  # second moment

    def step(self, params: np.ndarray, grads: np.ndarray) -> np.ndarray:
        """One Adam update step. Returns new params."""
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1

        # Update biased moments
        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        self.v = self.beta2 * self.v + (1 - self.beta2) * grads**2

        # Bias-corrected moments
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        # Parameter update
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def adam_vs_sgd_comparison():
    """Compare Adam vs. SGD on a simple 2D problem."""
    def loss(params): return params[0]**2 + 100 * params[1]**2   # narrow valley
    def grad(params): return np.array([2*params[0], 200*params[1]])

    adam = AdamOptimizer(lr=0.1)
    start = np.array([3.0, 0.3])

    params_adam = start.copy()
    params_sgd  = start.copy()
    lr_sgd = 0.001

    print(f"{'Step':5} {'Adam Loss':12} {'SGD Loss':12}")
    for step in range(1, 21):
        g_adam = grad(params_adam)
        params_adam = adam.step(params_adam, g_adam)

        g_sgd = grad(params_sgd)
        params_sgd = params_sgd - lr_sgd * g_sgd

        if step % 5 == 0:
            print(f"{step:5d} {loss(params_adam):12.6f} {loss(params_sgd):12.6f}")

    print(f"\nAdam converges faster on this ill-conditioned problem (100x scale difference in w1 vs w2)")

adam_vs_sgd_comparison()
```

**Why Adam is default for Transformers**: transformer training has very different gradient scales across parameters (embedding layers vs. attention weights). Adam's per-parameter adaptive rates handle this naturally. SGD with momentum is preferred for CNNs where gradient scales are more uniform.

---

## Learning Rate Schedules

The learning rate should not be constant throughout training:

```python
import numpy as np

def learning_rate_schedules():
    """
    Common LR schedules used in practice.
    """
    total_steps = 1000
    warmup_steps = 100
    steps = np.arange(1, total_steps + 1)

    # 1. Constant
    constant_lr = np.full(total_steps, 3e-4)

    # 2. Linear warmup + cosine decay (standard for Transformers)
    def cosine_with_warmup(step, base_lr=3e-4, warmup=100, total=1000):
        if step < warmup:
            return base_lr * step / warmup    # linear warmup
        progress = (step - warmup) / (total - warmup)
        return base_lr * 0.5 * (1 + np.cos(np.pi * progress))

    cosine_lr = np.array([cosine_with_warmup(s) for s in steps])

    # 3. Step decay (common in CNNs)
    step_lr = np.array([1e-3 * (0.1 ** (s // 333)) for s in steps])

    print("Learning Rate Schedules (sample values):")
    for step in [1, 50, 100, 200, 500, 1000]:
        print(f"  Step {step:5d}: cosine={cosine_with_warmup(step):.6f}  constant={3e-4:.6f}")

    print("\nWhy warmup? In early training, gradients are very noisy.")
    print("A large LR causes divergence. Warmup lets the model stabilize first.")
    print("Cosine decay: LR anneals smoothly to ~0 as training ends.")

learning_rate_schedules()
```

---

## Edge Cases and Misconceptions

**"Gradient descent always finds the global minimum."** False. It finds a local minimum. For non-convex loss surfaces (all neural networks), there can be many local minima. In practice, deep networks often have many local minima of similar quality — finding the global minimum is not necessary.

**"Smaller learning rate is always safer."** A too-small learning rate can get stuck in saddle points (points where gradient ≈ 0 but which are not minima) or converge so slowly the model never reaches good performance. Learning rate scheduling, not just small LR, is the answer.

**"Adam is always better than SGD."** Not true for all tasks. For image classification with CNNs, SGD with momentum often generalizes better than Adam. Adam sometimes overfits because its adaptive learning rates allow it to memorize data faster. For LLMs and transformers, Adam variants are superior.

**"Gradients point toward the minimum."** Gradients point toward *steepest ascent*. We go *against* the gradient for gradient *descent*.

---

## Key Takeaways

- Gradient descent works by repeatedly computing the gradient of the loss and taking a step opposite to it
- The **learning rate** \(\alpha\) controls step size — too small: slow convergence; too large: oscillation or divergence
- **Mini-batch SGD** is the standard: update after processing B examples (typically 32-256), balancing accuracy of gradient estimate with computational efficiency
- A single gradient descent step = forward pass → loss → gradient computation → weight update
- **Adam** maintains exponential moving averages of gradients and their squares, providing per-parameter adaptive learning rates — default for Transformer training
- **Learning rate schedules** (warmup + cosine decay) are essential for modern LLM training
- Gradient descent does not guarantee finding the global minimum — but local minima in deep networks are often good enough

---

## Further Reading

- [3Blue1Brown: Gradient Descent, How Neural Networks Learn](https://www.youtube.com/watch?v=IHZwWFHWa-w) — best visual introduction
- [Kingma & Ba (2014): Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) — original Adam paper
- [Sebastian Ruder's Overview of Gradient Descent Optimization Algorithms](https://ruder.io/optimizing-gradient-descent/) — comprehensive comparison of variants

---

**Next:** [Backpropagation — Computing Gradients Through Any Network](05-backpropagation.md)
