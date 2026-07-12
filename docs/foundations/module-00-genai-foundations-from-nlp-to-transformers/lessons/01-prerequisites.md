---
title: Prerequisites — What You Need Before Starting
description: >-
  Review the essential Python, math, and conceptual foundations needed to
  succeed in this AI engineering curriculum
duration: 60 min
difficulty: beginner
has_code: true
module: module-00
---
# Prerequisites — What You Need Before Starting

## Prerequisites

This is the first lesson. No prior AI or ML knowledge required. You do need:

- **Python**: ability to write functions, use classes, and work with `pip`
- **Math comfort**: not afraid of fractions, exponents, or summation notation (Σ)
- **Command line**: can navigate directories and run scripts

## What You'll Learn

| Objective | Why It Matters |
|-----------|---------------|
| Verify Python and NumPy readiness | Every AI library is built on top of these |
| Understand vectors and dot products | The single most-used operation in AI |
| Understand matrix multiplication | Neural networks ARE chains of matrix multiplies |
| Understand softmax and cross-entropy | These appear in every classification model |
| Set up your development environment | So you can run every code example in this course |

---

## Who This Curriculum Is For

This curriculum is designed for **software engineers** who want to understand and build with modern AI systems — not just call APIs blindly, but understand what is happening under the hood well enough to debug, optimize, and make good architectural decisions.

### What You Should Already Know

| Skill | Level Needed | Self-Check |
|-------|-------------|------------|
| **Python** | Intermediate | Can you write a class with `__init__` and methods? Can you use list comprehensions and decorators? |
| **Data structures** | Basic | Do you understand lists, dicts, sets, and when O(1) vs O(n) matters? |
| **Command line** | Basic | Can you navigate directories, run scripts, use git, and set environment variables? |
| **Math comfort** | Basic | Are you OK with fractions, exponents, and reading Σ notation? |

### What You Do NOT Need

- A degree in mathematics or computer science
- Prior machine learning experience
- GPU hardware (we use cloud APIs for most exercises)
- Knowledge of PyTorch or TensorFlow

!!! note "If You Are Rusty on Math"
    The key concepts you need are taught in this lesson. You do not need to know them already — you need to be willing to work through the examples slowly. Every formula in this curriculum is explained in plain English before the math is introduced.

---

## Python Essentials Review

### NumPy — The Language of AI

Almost every AI library — PyTorch, TensorFlow, JAX, scikit-learn — uses NumPy arrays (or GPU equivalents) as the fundamental data structure. If you understand NumPy shapes and operations, you can read nearly any AI code.

```python
import numpy as np

# ── Vectors (1D arrays) ──
v = np.array([1, 2, 3])
print(v.shape)   # (3,)  — 3 elements, one axis

# ── Matrices (2D arrays) ──
M = np.array([[1, 2, 3],
              [4, 5, 6]])
print(M.shape)   # (2, 3)  — 2 rows, 3 columns

# ── 3D tensors (batch of matrices) ──
# Shape (batch, seq_len, dim) is the standard for Transformer inputs
T = np.random.randn(32, 128, 512)
print(T.shape)   # (32, 128, 512)  — 32 examples, 128 tokens, 512-dim embeddings

# ── Matrix multiplication — THE core neural network operation ──
A = np.array([[1, 2],
              [3, 4]])   # shape (2, 2)
B = np.array([[5, 6],
              [7, 8]])   # shape (2, 2)
C = A @ B               # shape (2, 2) — @ is the matmul operator
print(C)
# [[19, 22],
#  [43, 50]]

# ── Shape rules: (m, k) @ (k, n) → (m, n) ──
# The inner dimensions must match!
X = np.random.randn(10, 4)   # 10 examples, 4 features
W = np.random.randn(4, 8)    # weight matrix: 4 inputs → 8 outputs
Y = X @ W                    # shape (10, 8) ✓
```

!!! warning "The Most Common NumPy Mistake"
    `A * B` is element-wise multiplication (requires same shape).
    `A @ B` is matrix multiplication.
    In AI code you almost always want `@`. Using `*` by accident produces silently wrong results.

### Key Python Patterns Used in AI Code

```python
from dataclasses import dataclass
from typing import Optional
import json

# ── Type hints (essential for reading modern AI library code) ──
def embed(text: str, model: str = "text-embedding-3-small") -> list[float]:
    """Returns a 1536-dimensional embedding vector."""
    ...

# ── List comprehensions (used heavily in data preprocessing) ──
documents = ["doc1", "doc2", "doc3"]
embeddings = [embed(doc) for doc in documents]

# ── Dataclasses (structured data in AI pipelines) ──
@dataclass
class Token:
    text: str
    id: int
    embedding: Optional[list[float]] = None

# ── Context managers (for resources like API clients) ──
with open("data.jsonl") as f:
    dataset = [json.loads(line) for line in f]

# ── Generator expressions (memory-efficient for large datasets) ──
total_tokens = sum(len(doc.split()) for doc in documents)
```

---

## Linear Algebra Essentials

### Vectors: Meaning as Coordinates

A vector is a list of numbers. In AI, **vectors represent meaning** — words, sentences, and images are all converted to vectors before any computation happens.

```python
import numpy as np

# A word embedding is just a vector of floats
# (real embeddings are 512–3072 dimensions, but 4D works for illustration)
word_king  = np.array([0.50, -0.20,  0.80, 0.10])
word_queen = np.array([0.48, -0.18,  0.78, 0.12])  # similar direction
word_table = np.array([-0.30,  0.90, -0.10, 0.60])  # very different direction

# Vectors that mean similar things point in similar directions in high-D space
```

The key intuition: two vectors are "similar" if they point in the same direction, regardless of their magnitude.

### Dot Product: Measuring Similarity

The dot product is the foundation of nearly every AI operation you will encounter — attention scores, cosine similarity, classification logits, and more.

**Definition:**

\[
a \cdot b = \sum_{i=1}^{n} a_i b_i = a_1 b_1 + a_2 b_2 + \cdots + a_n b_n
\]

This is just: multiply corresponding elements, then sum.

```python
# Worked numerical example — step by step

a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])

# Manual calculation:
# 1.0×4.0 + 2.0×5.0 + 3.0×6.0
# = 4.0 + 10.0 + 18.0
# = 32.0

dot_manual = sum(ai * bi for ai, bi in zip(a, b))  # 32.0
dot_numpy  = np.dot(a, b)                           # 32.0  (same result)

print(f"Manual: {dot_manual}, NumPy: {dot_numpy}")
```

**What does the value mean?**
- Positive → vectors point in similar directions (similar meaning)
- Zero → vectors are perpendicular (unrelated meaning)
- Negative → vectors point in opposite directions

### Cosine Similarity: Normalized Dot Product

The raw dot product depends on magnitude. Cosine similarity normalizes it to [-1, 1], making it a pure measure of direction:

\[
\cos(\theta) = \frac{a \cdot b}{\|a\| \|b\|}
\]

```python
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Returns 1.0 for identical direction, 0.0 for perpendicular, -1.0 for opposite."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Try it on our word vectors
print(cosine_similarity(word_king, word_queen))  # ≈ 0.999  (very similar)
print(cosine_similarity(word_king, word_table))  # ≈ -0.15  (unrelated)
```

### Matrix Multiplication: A Layer of a Neural Network

A neural network layer is literally just: `output = input @ weights + bias`.

Understanding matrix multiplication means understanding what a neural network layer computes.

```
Input shape:  (batch_size, input_dim)
Weight shape: (input_dim, output_dim)
Output shape: (batch_size, output_dim)

Rule: (m, k) @ (k, n) → (m, n)
The inner dimension k must match!
```

```python
# This IS a neural network layer (without the activation function)
batch_size  = 4      # 4 examples at once
input_dim   = 8      # 8 input features per example
output_dim  = 3      # 3 output values (e.g., 3 classes)

X = np.random.randn(batch_size, input_dim)    # shape: (4, 8)
W = np.random.randn(input_dim, output_dim)    # shape: (8, 3)
b = np.zeros(output_dim)                      # shape: (3,)

output = X @ W + b  # shape: (4, 3)
print(f"Input:  {X.shape}")   # (4, 8)
print(f"Output: {output.shape}")  # (4, 3)

# Each of the 4 examples has been transformed from 8 features to 3 values.
# The matrix W encodes WHAT transformation to apply — these are the "learned weights".
```

!!! note "Why This Matters for AI Engineering"
    Every time you call an embedding API, the model is performing hundreds of these matrix multiplications on your text. Understanding this helps you reason about latency (more parameters → more compute), memory (storing weight matrices), and context length limits (sequence dimension grows the intermediate tensors).

---

## Probability Essentials

### Softmax — Converting Scores to Probabilities

Any time a model outputs a probability distribution (next word prediction, classification, attention weights), it uses softmax. Understanding softmax means understanding the output of virtually every AI model.

**Definition:**

\[
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
\]

```python
def softmax(x: np.ndarray) -> np.ndarray:
    """
    Convert a vector of real-valued scores to a probability distribution.

    The subtraction of max(x) is NOT optional — it prevents numerical overflow
    when scores are large (e.g., 1000), since exp(1000) is infinity in float32.
    """
    shifted = x - np.max(x)   # numerically stable trick: max shifts to 0
    exp_x   = np.exp(shifted)
    return exp_x / exp_x.sum()

# Worked example: a language model's top-3 next-word predictions
scores = np.array([3.2, 1.5, 0.8])   # raw logits for words ["cat", "dog", "sat"]
probs  = softmax(scores)
print(probs.round(3))   # [0.813, 0.164, 0.023]
print(probs.sum())      # exactly 1.0

# Without the numerical stability trick:
big_scores = np.array([1000.0, 999.0, 998.0])
unstable = np.exp(big_scores) / np.exp(big_scores).sum()   # nan or inf!
stable   = softmax(big_scores)
print(stable.round(3))  # [0.576, 0.212, 0.212]  — correct
```

!!! warning "Numerical Stability Is Not Optional"
    The `max` subtraction in softmax prevents NaN values during training. If you ever see `nan` losses during model training, numerical instability is the first thing to check. This is why PyTorch's `nn.CrossEntropyLoss` takes raw logits, not probabilities — it handles the numerical stability internally.

### Cross-Entropy Loss — Measuring Prediction Quality

Cross-entropy quantifies how "surprised" a model is by the correct answer. A model that confidently predicts the right answer has low cross-entropy; a model that assigns tiny probability to the right answer has high cross-entropy.

**Definition:**

\[
\text{CE} = -\log p(\text{correct class})
\]

```python
def cross_entropy(predicted_probs: np.ndarray, true_label_index: int) -> float:
    """
    How wrong is our probability distribution?
    Lower is better. Perfect prediction (p=1.0) gives loss = 0.
    """
    # Only the probability of the TRUE class matters!
    return -np.log(predicted_probs[true_label_index] + 1e-10)  # 1e-10 for log(0) safety

# Example: predicting the next word in "The cat sat on the ___"
# Suppose the true answer is "mat" (index 0)
good_prediction = np.array([0.90, 0.07, 0.03])   # 90% confident → correct
bad_prediction  = np.array([0.10, 0.20, 0.70])   # only 10% confident → wrong answer

print(f"Good loss: {cross_entropy(good_prediction, 0):.3f}")   # 0.105
print(f"Bad loss:  {cross_entropy(bad_prediction, 0):.3f}")    # 2.303

# For reference: -log(1.0) = 0.0, -log(0.5) = 0.693, -log(0.1) = 2.303
# Training minimizes this number across ALL examples in the dataset.
```

**Why cross-entropy and not MSE for classification?**

Cross-entropy produces much larger gradients when the model is confidently wrong, which makes training faster and more stable. MSE treats "50% probability on the wrong class" and "1% probability on the wrong class" almost the same — cross-entropy strongly distinguishes them.

---

## Understanding Gradients (Preview)

You will see the concept of a **gradient** throughout this curriculum. Here is the minimal intuition you need now:

A gradient is a vector of partial derivatives that points in the direction that increases a function the most. For training:

```python
# Minimizing f(x) = (x - 3)²
# f'(x) = 2(x - 3) — the gradient tells us which direction INCREASES f
# We go in the OPPOSITE direction (subtract the gradient)

x = 10.0
learning_rate = 0.1

for step in range(15):
    loss     = (x - 3) ** 2           # current loss
    gradient = 2 * (x - 3)            # df/dx — direction of steepest increase
    x        = x - learning_rate * gradient  # step OPPOSITE to gradient

print(f"x converged to: {x:.4f}")     # ≈ 3.0000 (the minimum)
```

This simple update rule — compute gradient, step opposite to it — is how every neural network trains. The complexity in deep learning comes from computing the gradient for millions of parameters efficiently (that is what backpropagation does).

---

## Environment Setup

```bash
# Create and activate a virtual environment
python -m venv ai-env
source ai-env/bin/activate   # On Windows: ai-env\Scripts\activate

# Core packages used throughout this curriculum
pip install numpy pandas matplotlib scipy
pip install openai anthropic             # LLM APIs
pip install tiktoken                     # Token counting (OpenAI)
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only PyTorch

# For running notebooks
pip install jupyter ipykernel
python -m ipykernel install --user --name=ai-env

# Verify everything works
python -c "
import numpy as np
import torch
print(f'NumPy: {np.__version__}')
print(f'PyTorch: {torch.__version__}')
print('Environment ready!')
"
```

---

## Edge Cases and Misconceptions

**"Matrix multiplication is just element-wise multiply."** No. Element-wise multiplication requires identical shapes. Matrix multiplication requires the inner dimensions to match: `(m, k) @ (k, n) → (m, n)`. This is the most important shape rule in all of deep learning.

**"Softmax outputs the probability of being correct."** Softmax outputs a probability *distribution* over classes. Whether the highest-probability class is actually correct depends on how well the model was trained.

**"Cross-entropy only works for classification."** Language models use cross-entropy for next-token prediction, which is technically a classification task over the entire vocabulary. This is also why perplexity (a related metric) appears in LLM evaluation.

**"Dot product requires unit vectors."** The raw dot product depends on magnitude. Cosine similarity normalizes to [-1, 1]. In attention mechanisms, the dot product is scaled by √d_k before softmax to prevent extremely large values that saturate softmax.

---

## Production Connection

Every operation in this lesson appears directly in production AI systems:

| Concept | Where It Appears in Production |
|---------|-------------------------------|
| **Matrix multiply** | Every Transformer layer (attention, feed-forward) — billions of times per second on GPU |
| **Dot product** | Attention score computation, vector similarity search (RAG) |
| **Softmax** | Token probability output, attention weight normalization |
| **Cross-entropy** | Training objective for all language models |
| **Cosine similarity** | Embedding search in vector databases (Pinecone, Chroma, pgvector) |

---

## Self-Assessment Checklist

Before moving to the next lesson, verify you can:

- [ ] Create a NumPy array of a given shape and explain what the shape means
- [ ] Perform matrix multiplication and predict the output shape from the input shapes
- [ ] Compute a dot product manually on a 3-element vector
- [ ] Explain what softmax does and why the numerical stability trick matters
- [ ] Explain what cross-entropy loss measures (in plain English)
- [ ] Run gradient descent manually for 3 steps on a simple function

---

## Key Takeaways

- **Vectors represent meaning** in AI — words, sentences, and images all become lists of numbers
- **Dot product** is the primitive operation underlying attention, similarity search, and classification
- **Matrix multiplication** `(m,k) @ (k,n) → (m,n)` is literally what a neural network layer computes
- **Softmax** converts raw scores to probabilities; always use the numerically stable version
- **Cross-entropy** measures how surprised the model is by the correct answer — training minimizes this
- **Gradient** points in the direction of steepest increase; we step in the opposite direction to minimize loss

---

## Further Reading

- [3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) — the best visual introduction to vectors and matrices
- [Jay Alammar: A Visual and Interactive Guide to the Basics of Neural Networks](https://jalammar.github.io/visual-interactive-guide-basics-neural-networks/) — builds intuition from scratch with great visuals
- [Khan Academy: Linear Algebra](https://www.khanacademy.org/math/linear-algebra) — interactive exercises if you want to deepen the math
- [NumPy User Guide](https://numpy.org/doc/stable/user/index.html) — reference for all NumPy operations used throughout this curriculum

---

**Next:** [Mathematical Foundations for Deep Learning](02-math-foundations.md)
