---
title: Loss Functions — Measuring What the Model Gets Wrong
description: >-
  Understand the mathematical foundations of loss functions, why the choice of
  loss shapes what the model learns, trace numerical examples for MSE and
  cross-entropy, and connect loss design to real production decisions
duration: 75 min
difficulty: beginner
has_code: true
module: module-05
---
# Loss Functions — Measuring What the Model Gets Wrong

## Prerequisites

- [Lesson 01: Introduction to Neural Networks](01-introduction-to-neural-networks.md)
- [Lesson 02: Neurons & Activation Functions](02-neurons-activation-functions.md)
- [Module 00 Lesson 02: Math Foundations](../../module-00-genai-foundations-from-nlp-to-transformers/lessons/02-math-foundations.md) — probability, log

## What You'll Learn

| Objective | Why It Matters |
|-----------|---------------|
| What a loss function is and why the choice matters | Different losses produce fundamentally different models |
| Derive MSE from first principles | Understanding why regression uses MSE, not MAE |
| Derive cross-entropy from maximum likelihood | Why log probability is the natural loss for classification |
| Numerically compare loss landscapes | Intuition for optimization difficulty |
| Recognize incorrect loss/activation pairings | The most common training bug |

---

## What Is a Loss Function?

A loss function \(\mathcal{L}(\hat{y}, y)\) quantifies the discrepancy between the model's prediction \(\hat{y}\) and the ground-truth label \(y\). For a dataset of \(N\) examples:

\[
\mathcal{L}_\text{total} = \frac{1}{N} \sum_{i=1}^N \mathcal{L}(\hat{y}_i, y_i)
\]

The loss has one job: to define "wrong" precisely enough that gradient descent can move the weights toward "right."

!!! note "Loss vs. Metric"
    A **loss function** must be differentiable — it guides optimization. A **metric** (accuracy, F1, BLEU) measures what you actually care about but is often non-differentiable. Always think in both: minimize a differentiable proxy loss; evaluate on the metric you care about. They can diverge: a model can have 99% accuracy but high cross-entropy loss if it is overconfident.

```python
import numpy as np

def loss_vs_metric_demo():
    """
    Illustrate loss vs metric divergence.
    Two models with identical accuracy but very different loss.
    """
    # Binary classification: true labels
    y_true = np.array([1, 1, 0, 0])

    # Model A: moderately confident
    p_A = np.array([0.7, 0.6, 0.3, 0.4])   # probabilities of class 1

    # Model B: overconfident
    p_B = np.array([0.99, 0.55, 0.01, 0.45])

    def bce_loss(y, p):
        """Binary cross-entropy loss."""
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def accuracy(y, p):
        return (y == (p > 0.5)).mean()

    print(f"Model A — accuracy: {accuracy(y_true, p_A):.0%}  BCE loss: {bce_loss(y_true, p_A):.4f}")
    print(f"Model B — accuracy: {accuracy(y_true, p_B):.0%}  BCE loss: {bce_loss(y_true, p_B):.4f}")
    # Both 100% accuracy, but Model A has lower loss — less overconfident, better calibrated

loss_vs_metric_demo()
```

---

## Mean Squared Error (MSE)

\[
\text{MSE} = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2
\]

### Derivation from Gaussian Likelihood

MSE is not arbitrary. It follows naturally from assuming a Gaussian noise model. If we assume:

\[
y_i = f(x_i; \theta) + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \sigma^2)
\]

Then the probability of observing \(y_i\) given the model is:

\[
P(y_i | x_i, \theta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(y_i - \hat{y}_i)^2}{2\sigma^2}\right)
\]

Maximizing the log-likelihood (MLE) over the dataset:

\[
\log P = \text{const} - \frac{1}{2\sigma^2} \sum_{i=1}^N (y_i - \hat{y}_i)^2
\]

Maximizing log-likelihood is equivalent to minimizing \(\sum (y_i - \hat{y}_i)^2\) — which is MSE.

**Implication**: MSE is the right loss when your errors really are Gaussian. If your data has heavy-tailed errors or outliers, MSE is a poor fit.

```python
import numpy as np

def mse_numerical_example():
    """
    Step-by-step MSE computation with gradient.
    Predicting house prices (in $100,000 units).
    """
    y_true = np.array([3.5, 2.0, 4.8, 1.5, 5.2])   # true prices
    y_pred = np.array([3.2, 2.3, 4.6, 1.8, 4.9])   # model predictions

    # Per-example squared errors
    errors  = y_pred - y_true
    sq_errors = errors ** 2

    print("=== MSE Computation ===")
    for i, (yt, yp, e, se) in enumerate(zip(y_true, y_pred, errors, sq_errors)):
        print(f"  Example {i+1}: true={yt:.1f}  pred={yp:.1f}  error={e:+.1f}  sq_error={se:.4f}")

    mse = sq_errors.mean()
    print(f"\nMSE = {sq_errors.sum():.4f} / {len(y_true)} = {mse:.4f}")
    print(f"RMSE = sqrt(MSE) = {np.sqrt(mse):.4f}")
    # RMSE is in the same units as y — easier to interpret

    # Gradient of MSE w.r.t. predictions
    # d(MSE)/d(y_pred_i) = (2/N) * (y_pred_i - y_true_i)
    grad_mse = (2 / len(y_true)) * errors
    print(f"\nGradient ∂MSE/∂ŷ: {grad_mse.round(4)}")
    print("(Positive grad means prediction too high; gradient will push prediction down)")

mse_numerical_example()
```

### MSE Sensitivity to Outliers

```python
def mse_outlier_sensitivity():
    """
    MSE squares errors, so outliers are penalized much more heavily.
    This can dominate training.
    """
    y_true = np.array([2.0, 3.0, 2.5, 3.5, 2.0])

    y_pred_good    = np.array([2.1, 3.1, 2.4, 3.6, 2.1])   # small errors
    y_pred_outlier = np.array([2.1, 3.1, 2.4, 3.6, 8.0])   # one large outlier

    def mse(yt, yp): return ((yt - yp)**2).mean()
    def mae(yt, yp): return (np.abs(yt - yp)).mean()

    print("Without outlier:")
    print(f"  MSE = {mse(y_true, y_pred_good):.4f}   MAE = {mae(y_true, y_pred_good):.4f}")

    print("With outlier (last prediction = 8.0, true = 2.0):")
    print(f"  MSE = {mse(y_true, y_pred_outlier):.4f}   MAE = {mae(y_true, y_pred_outlier):.4f}")
    # MSE jumps dramatically (outlier penalty is 6^2 = 36), MAE jumps moderately (|8-2| = 6)

mse_outlier_sensitivity()
```

---

## Binary Cross-Entropy Loss

For binary classification, the model outputs a probability \(\hat{p} = \sigma(z) \in (0, 1)\) of the positive class.

\[
\mathcal{L}_\text{BCE} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]
\]

### Derivation from Bernoulli Likelihood

Each label \(y_i \in \{0, 1\}\) is modeled as a Bernoulli random variable:

\[
P(y_i | x_i) = \hat{p}_i^{y_i} (1 - \hat{p}_i)^{1 - y_i}
\]

Log-likelihood over the dataset:

\[
\log P = \sum_{i=1}^N \left[ y_i \log \hat{p}_i + (1 - y_i) \log(1 - \hat{p}_i) \right]
\]

Negating gives the cross-entropy loss. **Cross-entropy is the natural loss for any probability model.**

```python
import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def bce_loss_trace():
    """
    Numerical trace of BCE for a mini-batch of 4 examples.
    Spam classification: 1=spam, 0=not spam.
    """
    y_true = np.array([1, 0, 1, 0])
    z      = np.array([2.0, -1.5, 0.8, -3.0])   # raw logits from output layer

    p_hat = sigmoid(z)
    print("=== Binary Cross-Entropy Trace ===")
    print(f"{'Example':8} {'y_true':6} {'logit z':8} {'p_hat':8} {'loss':8}")
    print("-" * 50)

    per_example_loss = []
    for i, (y, z_val, p) in enumerate(zip(y_true, z, p_hat)):
        if y == 1:
            loss = -np.log(p)           # penalize for not predicting 1
        else:
            loss = -np.log(1 - p)       # penalize for not predicting 0
        per_example_loss.append(loss)
        print(f"{i+1:8d} {y:6d} {z_val:8.1f} {p:8.4f} {loss:8.4f}")

    total_loss = np.mean(per_example_loss)
    print(f"\nBCE Loss = {total_loss:.4f}")

bce_loss_trace()
```

### Gradient of BCE — Why It's Well-Behaved

The gradient of BCE w.r.t. the logit \(z\) (before sigmoid) is:

\[
\frac{\partial \mathcal{L}_\text{BCE}}{\partial z} = \hat{p} - y
\]

This is beautifully simple: just the prediction error. No sigmoid derivative involved.

```python
def bce_gradient_trace():
    """
    Show the gradient of BCE w.r.t. the logit z.
    When loss is BCE + sigmoid output, gradients don't vanish.
    """
    y_true = np.array([1.0, 0.0, 1.0, 0.0])
    z      = np.array([2.0, -1.5, 0.8, -3.0])

    p_hat = sigmoid(z)
    gradient = p_hat - y_true   # d(BCE)/dz = p_hat - y

    print("=== BCE Gradient w.r.t. Logit z ===")
    for y, p, g in zip(y_true, p_hat, gradient):
        direction = "← logit too HIGH, push down" if g > 0 else "← logit too LOW, push up"
        print(f"  y={y:.0f}  p={p:.4f}  ∂L/∂z={g:+.4f}  {direction}")

bce_gradient_trace()
```

!!! note "Why the Pairing Matters"
    When you pair **sigmoid output + BCE loss**, the `σ'(z)` term in the backward pass cancels with a denominator in the BCE derivative, leaving only `p̂ - y`. This is numerically clean. If you use **sigmoid output + MSE loss**, the gradient includes `σ'(z)` which saturates for |z| > 5. This is not impossible but is harder to train.

---

## Categorical Cross-Entropy (Multi-Class)

For K-class classification with one-hot encoded labels \(y \in \{0,1\}^K\):

\[
\mathcal{L}_\text{CE} = -\sum_{k=1}^K y_k \log(\hat{p}_k)
\]

Since only one \(y_k = 1\), this simplifies to:

\[
\mathcal{L}_\text{CE} = -\log(\hat{p}_\text{correct class})
\]

**Intuition**: maximize the probability assigned to the correct class. Every other class probability is irrelevant as long as the correct one is high.

```python
def softmax(z: np.ndarray) -> np.ndarray:
    exp_z = np.exp(z - z.max())
    return exp_z / exp_z.sum()

def categorical_cross_entropy_trace():
    """
    Numerical trace of categorical cross-entropy.
    3-class problem: classes are [cat, dog, bird].
    """
    # Network output logits for one example
    logits = np.array([2.0, 1.5, 0.3])

    # Ground truth: label index 0 (cat)
    true_class = 0

    # Convert logits to probabilities via softmax
    probs = softmax(logits)

    # One-hot encode ground truth
    y_one_hot = np.zeros(3)
    y_one_hot[true_class] = 1.0

    # Cross-entropy
    loss_full   = -np.sum(y_one_hot * np.log(probs + 1e-10))
    loss_simple = -np.log(probs[true_class])   # equivalent, just index correct class

    print("=== Categorical Cross-Entropy ===")
    print(f"Logits:           {logits}")
    print(f"Softmax probs:    {probs.round(4)}")
    print(f"True class:       {true_class} (cat)")
    print(f"Loss (full form): {loss_full:.4f}")
    print(f"Loss (shortcut):  {loss_simple:.4f}")
    print(f"  = -log({probs[true_class]:.4f}) = {loss_simple:.4f}")
    print(f"\nInterpretation: the model assigned {probs[true_class]:.1%} probability to cat")
    print(f"  A perfect model would assign 100% → loss = -log(1) = 0")
    print(f"  A random model for 3 classes would assign 33% → loss = {-np.log(1/3):.4f}")

categorical_cross_entropy_trace()
```

---

## Loss Landscape Intuition

The shape of the loss landscape determines how easy optimization is:

```python
import numpy as np

def loss_landscape_comparison():
    """
    Compare MSE and BCE loss landscapes for a single sigmoid neuron.
    Shows why BCE is convex (easy to optimize) for logistic regression,
    while MSE + sigmoid combination creates a non-convex landscape.
    """
    y_true = 1   # positive class

    z_range = np.linspace(-5, 5, 100)   # logit values
    p_hat   = sigmoid(z_range)          # predicted probabilities

    # BCE loss at each z
    bce = -np.log(p_hat)   # since y=1, only first term survives

    # MSE loss at each z
    mse = (1 - p_hat)**2   # (y_true - p_hat)^2

    # Find minimum of each
    bce_min_idx = np.argmin(bce)
    mse_min_idx = np.argmin(mse)

    print("BCE loss: monotonically decreasing → global minimum at z → +∞")
    print(f"  At z=-3: BCE={bce[z_range.searchsorted(-3)]:.4f}")
    print(f"  At z= 0: BCE={bce[z_range.searchsorted(0)]:.4f}")
    print(f"  At z=+3: BCE={bce[z_range.searchsorted(3)]:.4f}")
    print(f"\nMSE loss for sigmoid output: also monotone for this simple case")
    print("  But gradient very small near z=0 due to σ'(z) multiplication")
    print(f"  σ'(0) = 0.25 → gradient ×4 smaller than BCE")

loss_landscape_comparison()
```

---

## Language Model Loss: Cross-Entropy Over Vocabulary

In LLMs like GPT, the loss is cross-entropy over a vocabulary of 50,000+ tokens:

\[
\mathcal{L}_\text{LM} = -\frac{1}{T} \sum_{t=1}^T \log P(w_t | w_1, \ldots, w_{t-1})
\]

Each step is a 50,000-way classification problem: predict the correct next token.

```python
def language_model_loss_example():
    """
    Conceptual LM loss for a 5-token sequence.
    Shows how perplexity relates to cross-entropy.
    """
    vocab_size = 50257   # GPT-2 vocabulary size

    # Simulated next-token probabilities (what the model assigns to the correct token)
    # In practice these come from softmax over the vocabulary
    correct_token_probs = np.array([0.23, 0.15, 0.42, 0.08, 0.31])

    # Per-token losses
    token_losses = -np.log(correct_token_probs)

    # Average cross-entropy
    avg_ce = token_losses.mean()

    # Perplexity = exp(cross-entropy)
    perplexity = np.exp(avg_ce)

    print("=== Language Model Loss ===")
    for i, (p, l) in enumerate(zip(correct_token_probs, token_losses)):
        print(f"  Token {i+1}: P(correct) = {p:.2f}  CE = {l:.4f}")

    print(f"\nAverage CE loss = {avg_ce:.4f}")
    print(f"Perplexity = exp({avg_ce:.4f}) = {perplexity:.2f}")
    print(f"\nRandom model perplexity = {vocab_size:.0f}")
    print(f"Our model is {vocab_size / perplexity:.1f}x better than random")

language_model_loss_example()
```

---

## Choosing the Right Loss

| Task | Output | Loss Function | Activation |
|------|--------|--------------|------------|
| Regression | Single value | MSE or MAE | Linear |
| Binary classification | 0/1 probability | Binary cross-entropy | Sigmoid |
| Multi-class (mutually exclusive) | One of K classes | Categorical cross-entropy | Softmax |
| Multi-label (multiple labels possible) | K independent probabilities | Summed BCE per label | K sigmoids |
| Language modeling | Next token from vocabulary | Categorical cross-entropy | Softmax |
| Anomaly scoring | Reconstruction error | MSE on input space | Linear or Sigmoid |

!!! warning "Common Mistakes"
    **Mistake 1**: Using MSE for classification. MSE doesn't penalize confident wrong predictions heavily enough and its gradient with sigmoid output is numerically poor.

    **Mistake 2**: Using softmax for multi-label classification (when multiple labels can be true simultaneously). Softmax forces probabilities to sum to 1, so if class A is very probable, class B is penalized even if B is also true. Use independent sigmoids instead.

    **Mistake 3**: Forgetting to clip probabilities before taking log. `log(0) = -∞` will crash training. Always use `np.clip(p, 1e-7, 1 - 1e-7)`.

---

## Production Connection

In production language models, the training loss is monitored throughout. Loss curves tell you:

- **Loss decreasing smoothly**: healthy training
- **Loss plateau early**: learning rate too small or model capacity too low
- **Loss spiking**: learning rate too large or data quality issues
- **Training loss >> validation loss**: overfitting
- **Training loss ≈ validation loss, both high**: underfitting

```python
import numpy as np

def interpret_loss_curve():
    """
    Simulate and interpret a typical training loss curve.
    """
    steps = np.arange(1, 101)

    # Simulated loss curves
    train_loss = 2.3 * np.exp(-0.03 * steps) + 0.3 + 0.05 * np.random.randn(100)
    val_loss   = 2.3 * np.exp(-0.025 * steps) + 0.45 + 0.08 * np.random.randn(100)

    # Diagnose at step 50 and 100
    for step in [10, 50, 100]:
        tl = train_loss[step - 1]
        vl = val_loss[step - 1]
        gap = vl - tl

        print(f"Step {step:3d}: train={tl:.4f}  val={vl:.4f}  gap={gap:.4f}")

    print("\nInterpretation:")
    print("  Early steps (high loss): model starts near random baseline")
    print("  Mid training: loss decreases as model learns patterns")
    print("  Late training: if val loss rises while train falls → overfitting")

interpret_loss_curve()
```

---

## Key Takeaways

- A loss function is the mathematical definition of "wrong" — it determines what the model optimizes toward
- **MSE** derives from Gaussian noise assumptions; it squares errors, making it sensitive to outliers
- **Binary cross-entropy** derives from Bernoulli likelihood; it is the natural loss for binary classification
- **Categorical cross-entropy** simplifies to `-log(p_correct_class)` — maximize probability of the right label
- Language model training uses cross-entropy over the vocabulary at every token position
- Always pair loss with output activation correctly: sigmoid ↔ BCE, softmax ↔ CE, linear ↔ MSE
- Loss and metrics can diverge: always evaluate the metric you care about, not just training loss

---

## Further Reading

- [3Blue1Brown: How does a Neural Network Learn?](https://www.youtube.com/watch?v=IHZwWFHWa-w) — visual derivation of gradient descent with loss
- [CS229 Lecture Notes: Generalized Linear Models](https://cs229.stanford.edu/notes2023fall/main_notes.pdf) — MLE derivation of MSE and cross-entropy
- [PyTorch Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions) — production implementations with numerically stable forms

---

**Next:** [Gradient Descent — Finding the Minimum](04-gradient-descent.md)
