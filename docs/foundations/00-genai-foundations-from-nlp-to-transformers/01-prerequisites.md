---
title: Prerequisites — What You Need Before Starting
description: >-
  Review the essential Python, math, and conceptual foundations needed to
  succeed in this AI engineering curriculum
duration: 45 min
difficulty: beginner
has_code: true
---
# Prerequisites — What You Need Before Starting

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Verify your Python readiness for AI/ML work | 45 min | Beginner |
| Review essential linear algebra concepts (vectors, matrices, dot products) | | |
| Review essential probability and statistics concepts | | |
| Set up your development environment | | |

---

## Who This Curriculum Is For

This curriculum is designed for **software developers and engineers** who want to understand and build with modern AI systems. You do not need a PhD in machine learning — but you do need some foundations.

### What You Should Already Know

| Skill | Level Needed | Self-Check |
|-------|-------------|------------|
| **Python** | Intermediate | Can you write classes, use list comprehensions, and work with pip? |
| **Data structures** | Basic | Do you understand lists, dicts, sets, and their time complexities? |
| **Command line** | Basic | Can you navigate directories, run scripts, and use git? |
| **Math comfort** | Basic | Are you OK with equations, not afraid of summation notation? |

### What You Do NOT Need

- A degree in mathematics or computer science
- Prior experience with machine learning
- GPU hardware (we use cloud APIs for most exercises)

---

## Python Essentials Review

If any of these feel unfamiliar, review them before continuing:

### NumPy — The Language of AI

```python
import numpy as np

# Vectors (1D arrays)
v = np.array([1, 2, 3])
print(v.shape)  # (3,)

# Matrices (2D arrays)
M = np.array([[1, 2], [3, 4], [5, 6]])
print(M.shape)  # (3, 2)

# Matrix multiplication — THE core operation in neural networks
A = np.array([[1, 2], [3, 4]])  # 2x2
B = np.array([[5, 6], [7, 8]])  # 2x2
C = A @ B  # Matrix multiply
print(C)
# [[19 22]
#  [43 50]]

# Element-wise operations
print(A * B)  # Element-wise multiply (NOT matrix multiply)
print(A + B)  # Element-wise addition
```

### Key Python Patterns Used in AI Code

```python
# List comprehensions (used everywhere in data processing)
embeddings = [get_embedding(text) for text in documents]

# Dictionary comprehensions
word_counts = {word: text.count(word) for word in vocabulary}

# Type hints (used in modern AI libraries)
def embed(text: str) -> list[float]:
    ...

# Context managers (for GPU memory, file handling)
with open("data.json") as f:
    data = json.load(f)

# Dataclasses (structured data)
from dataclasses import dataclass

@dataclass
class Token:
    text: str
    id: int
    embedding: list[float]
```

---

## Linear Algebra Essentials

You do not need to be an expert, but you must understand these concepts:

### Vectors

A vector is a list of numbers. In AI, vectors represent **meaning** — words, sentences, and images are all converted to vectors.

```python
import numpy as np

# A word embedding is a vector
word_king = np.array([0.5, -0.2, 0.8, 0.1])    # 4-dimensional vector
word_queen = np.array([0.48, -0.18, 0.78, 0.12])  # Similar direction = similar meaning
```

### Dot Product

The dot product measures how "similar" two vectors are. It is the single most important operation in AI.

```python
def dot_product(a, b):
    """Sum of element-wise products."""
    return sum(a_i * b_i for a_i, b_i in zip(a, b))

# Or with NumPy:
similarity = np.dot(word_king, word_queen)
print(f"Similarity: {similarity:.3f}")  # Higher = more similar
```

### Matrix Multiplication

Neural networks are chains of matrix multiplications. If you understand this, you understand 80% of how deep learning computes.

```
Input vector [1x4] × Weight matrix [4x3] = Output vector [1x3]

  [x1, x2, x3, x4]  ×  [[w11, w12, w13],     =  [y1, y2, y3]
                          [w21, w22, w23],
                          [w31, w32, w33],
                          [w41, w42, w43]]
```

```python
# This is literally what a neural network layer does:
input_vector = np.array([1, 2, 3, 4])      # Input (4 features)
weights = np.random.randn(4, 3)             # Weights (4 inputs → 3 outputs)
output = input_vector @ weights             # Matrix multiply
print(output.shape)  # (3,) — transformed to 3 dimensions
```

---

## Probability Essentials

### Softmax — Converting Numbers to Probabilities

Softmax is how neural networks output probabilities. It takes any list of numbers and converts them to a probability distribution (all positive, sums to 1).

```python
def softmax(x):
    """Convert a vector of scores to probabilities."""
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / exp_x.sum()

scores = np.array([2.0, 1.0, 0.1])
probs = softmax(scores)
print(probs)  # [0.659, 0.242, 0.099] — sums to 1.0
```

You will see softmax in:
- Attention mechanisms (choosing which words to focus on)
- Language model outputs (choosing the next word)
- Classification (choosing a category)

### Cross-Entropy Loss

This measures how wrong a probability prediction is. Lower = better.

```python
def cross_entropy(predicted_probs, true_label_index):
    """How wrong is our prediction?"""
    return -np.log(predicted_probs[true_label_index])

# Perfect prediction: very low loss
print(cross_entropy([0.9, 0.05, 0.05], 0))  # 0.105

# Bad prediction: high loss
print(cross_entropy([0.1, 0.1, 0.8], 0))    # 2.302
```

---

## Environment Setup

```bash
# Create a virtual environment
python -m venv ai-env
source ai-env/bin/activate  # On Windows: ai-env\Scripts\activate

# Install core packages
pip install numpy pandas matplotlib
pip install openai anthropic    # LLM APIs
pip install torch               # PyTorch (for understanding, not required for all lessons)
pip install jupyter             # For interactive exploration

# Verify installation
python -c "import numpy; import torch; print('Ready!')"
```

---

## Self-Assessment Checklist

Before moving to the next lesson, make sure you can:

- [ ] Write Python functions with type hints
- [ ] Create and manipulate NumPy arrays
- [ ] Explain what a dot product measures
- [ ] Explain what matrix multiplication does
- [ ] Explain what softmax does and why it is useful
- [ ] Run Python scripts and Jupyter notebooks

If you struggle with any of these, here are the best resources to review:

## Resources

- [3Blue1Brown: Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) -- The best visual introduction to vectors and matrices (free YouTube series)
- [YouTube: Python for Data Science - Full Course](https://www.youtube.com/watch?v=LHBE6Q9XlzI) -- Python + NumPy fundamentals
- [Khan Academy: Linear Algebra](https://www.khanacademy.org/math/linear-algebra) -- Interactive exercises for vectors and matrices
- [Khan Academy: Probability & Statistics](https://www.khanacademy.org/math/statistics-probability) -- Probability foundations

---

Next: Mathematical Foundations for Deep Learning
