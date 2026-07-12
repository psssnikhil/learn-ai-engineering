---
title: Recurrent Neural Networks (RNNs) & LSTMs
description: >-
  Understand how neural networks process sequences and remember information over
  time
duration: 35 min
difficulty: intermediate
has_code: true
module: module-05
youtube: 'https://www.youtube.com/watch?v=LHXXI4-IEns'
objectives:
  - Understand RNN architecture
  - Explain vanishing gradient in RNNs
  - Implement simple RNN
  - Understand LSTM gates
---
# Recurrent Neural Networks (RNNs) & LSTMs

## The Problem with Sequences

**CNNs** work great for images (fixed size, spatial structure).

But what about **sequences**?
- Text: "The cat sat on the..."
- Time series: Stock prices over time
- Audio: Speech recognition
- Video: Frame-by-frame analysis

**Need**: Networks with **memory** that can process variable-length sequences.

**Solution**: Recurrent Neural Networks (RNNs)!

---

## The RNN Idea: Memory Through Time

Traditional neural network:

```
Input → Network → Output
(Each input processed independently)
```

RNN:

```
Input₁ → RNN → Output₁
          ↓ (hidden state)
Input₂ → RNN → Output₂
          ↓
Input₃ → RNN → Output₃
```

**Key**: Hidden state carries information from previous steps!

---

## RNN Architecture

### Unrolled View:

```
    h₀        h₁        h₂        h₃
    ↓         ↓         ↓         ↓
[x₁] → [RNN] → [RNN] → [RNN] → [RNN]
        ↓         ↓         ↓         ↓
       y₁        y₂        y₃        y₄
```

### Mathematical Formulation:

```
h_t = tanh(W_h × h_{t-1} + W_x × x_t + b)
y_t = W_y × h_t + b_y
```

Where:
- `h_t`: Hidden state at time t
- `x_t`: Input at time t
- `y_t`: Output at time t
- `W_h, W_x, W_y`: Weight matrices
- `b, b_y`: Biases

---

## Simple RNN Implementation

```python
import numpy as np

class SimpleRNN:
    """
    Vanilla RNN for sequence processing
    """
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights
        self.W_h = np.random.randn(hidden_size, hidden_size) * 0.01
        self.W_x = np.random.randn(hidden_size, input_size) * 0.01
        self.W_y = np.random.randn(output_size, hidden_size) * 0.01
        
        self.b_h = np.zeros((hidden_size, 1))
        self.b_y = np.zeros((output_size, 1))
        
        self.hidden_size = hidden_size
    
    def forward(self, inputs):
        """
        Forward pass through sequence
        
        Args:
            inputs: List of input vectors [(input_size, 1), ...]
        
        Returns:
            outputs: List of output vectors
            hidden_states: List of hidden states
        """
        h = np.zeros((self.hidden_size, 1))  # Initial hidden state
        hidden_states = []
        outputs = []
        
        for x_t in inputs:
            # Update hidden state
            h = np.tanh(self.W_h @ h + self.W_x @ x_t + self.b_h)
            
            # Calculate output
            y_t = self.W_y @ h + self.b_y
            
            hidden_states.append(h)
            outputs.append(y_t)
        
        return outputs, hidden_states
    
    def predict_next(self, sequence):
        """Predict next element in sequence"""
        outputs, _ = self.forward(sequence)
        return outputs[-1]


# Example: Simple character-level RNN
vocab_size = 26  # a-z
hidden_size = 50
output_size = 26

rnn = SimpleRNN(vocab_size, hidden_size, output_size)

# Input sequence: "hello" (one-hot encoded)
sequence = [
    np.random.randn(vocab_size, 1),  # 'h'
    np.random.randn(vocab_size, 1),  # 'e'
    np.random.randn(vocab_size, 1),  # 'l'
    np.random.randn(vocab_size, 1),  # 'l'
    np.random.randn(vocab_size, 1),  # 'o'
]

outputs, hidden_states = rnn.forward(sequence)

print(f"Processed {len(outputs)} timesteps")
print(f"Output shape: {outputs[0].shape}")
print(f"Hidden state shape: {hidden_states[0].shape}")
```

---

## RNN Applications

### 1. Sentiment Analysis (Many-to-One)

```
"This movie is amazing!" → Positive ✅

[This] → [RNN] →
[movie] → [RNN] →
[is] → [RNN] →
[amazing] → [RNN] → [Positive/Negative]
```

---

### 2. Text Generation (One-to-Many)

```
Seed: "Once upon a"
      ↓
     RNN → "time"
      ↓
     RNN → "there"
      ↓
     RNN → "was"
      ↓
     RNN → "a"
      ↓
     RNN → "dragon"
```

---

### 3. Machine Translation (Many-to-Many)

```
Encoder:                Decoder:
"Hello" → [RNN] →       → [RNN] → "Hola"
"World" → [RNN] → [h]   → [RNN] → "Mundo"
                  ↑
            (Context vector)
```

---

## The Vanishing Gradient Problem

RNNs struggle with **long sequences**:

```
"The cat, which was sitting on the mat that was placed in the room where the family often gathered, was orange."

Need to remember "cat" → "was" (far away!)
```

**Problem**: Gradients vanish during backpropagation through time.

```
Gradient flow:
h₁₀₀ ← h₉₉ ← ... ← h₂ ← h₁

Each step: gradient × tanh'(x) (< 1)

After 100 steps: gradient ≈ 0 😢
```

**Consequence**: Can't learn long-term dependencies!

---

## LSTM: The Solution

**Long Short-Term Memory (LSTM)** networks solve vanishing gradients.

**Key idea**: Explicit memory cell + gates to control information flow.

### LSTM Architecture:

```
        ┌─────────────────┐
        │   Memory Cell   │ ← Long-term memory
        └─────────────────┘
          ↑     ↑     ↑
          │     │     │
     [Forget] [Input] [Output]
       Gate    Gate    Gate
```

---

## LSTM Gates

### 1. Forget Gate

**Decides what to forget from cell state**:

```
f_t = σ(W_f × [h_{t-1}, x_t] + b_f)

Output: 0 (forget) to 1 (keep)
```

---

### 2. Input Gate

**Decides what new information to store**:

```
i_t = σ(W_i × [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_C × [h_{t-1}, x_t] + b_C)

New candidates to add to memory
```

---

### 3. Update Cell State

```
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
      ↑                ↑
   Forget old      Add new
```

(⊙ means element-wise multiplication)

---

### 4. Output Gate

**Decides what to output**:

```
o_t = σ(W_o × [h_{t-1}, x_t] + b_o)
h_t = o_t ⊙ tanh(C_t)
```

---

## LSTM Implementation

```python
import numpy as np

class LSTMCell:
    """Single LSTM cell"""
    
    def __init__(self, input_size, hidden_size):
        # Combine all gates into single matrices for efficiency
        self.W_f = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.W_i = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.W_C = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        self.W_o = np.random.randn(hidden_size, input_size + hidden_size) * 0.01
        
        self.b_f = np.zeros((hidden_size, 1))
        self.b_i = np.zeros((hidden_size, 1))
        self.b_C = np.zeros((hidden_size, 1))
        self.b_o = np.zeros((hidden_size, 1))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x_t, h_prev, C_prev):
        """
        LSTM forward pass
        
        Args:
            x_t: Input at time t (input_size, 1)
            h_prev: Previous hidden state (hidden_size, 1)
            C_prev: Previous cell state (hidden_size, 1)
        
        Returns:
            h_t: New hidden state
            C_t: New cell state
        """
        # Concatenate input and previous hidden state
        concat = np.vstack([h_prev, x_t])
        
        # Forget gate
        f_t = self.sigmoid(self.W_f @ concat + self.b_f)
        
        # Input gate
        i_t = self.sigmoid(self.W_i @ concat + self.b_i)
        
        # Candidate cell state
        C_tilde = np.tanh(self.W_C @ concat + self.b_C)
        
        # Update cell state
        C_t = f_t * C_prev + i_t * C_tilde
        
        # Output gate
        o_t = self.sigmoid(self.W_o @ concat + self.b_o)
        
        # New hidden state
        h_t = o_t * np.tanh(C_t)
        
        return h_t, C_t


class LSTM:
    """Multi-layer LSTM"""
    
    def __init__(self, input_size, hidden_size, output_size):
        self.cell = LSTMCell(input_size, hidden_size)
        self.W_y = np.random.randn(output_size, hidden_size) * 0.01
        self.b_y = np.zeros((output_size, 1))
        self.hidden_size = hidden_size
    
    def forward(self, inputs):
        """Process sequence"""
        h = np.zeros((self.hidden_size, 1))
        C = np.zeros((self.hidden_size, 1))
        
        outputs = []
        
        for x_t in inputs:
            h, C = self.cell.forward(x_t, h, C)
            y_t = self.W_y @ h + self.b_y
            outputs.append(y_t)
        
        return outputs


# Example usage
input_size = 10
hidden_size = 20
output_size = 5

lstm = LSTM(input_size, hidden_size, output_size)

# Sequence of length 7
sequence = [np.random.randn(input_size, 1) for _ in range(7)]

outputs = lstm.forward(sequence)

print(f"Processed {len(outputs)} timesteps")
print(f"Each output shape: {outputs[0].shape}")
```

---

## RNN vs LSTM vs GRU

| Feature | RNN | LSTM | GRU |
|---------|-----|------|-----|
| **Parameters** | Low | High | Medium |
| **Long sequences** | ❌ | ✅ | ✅ |
| **Training speed** | Fast | Slow | Medium |
| **Memory** | No explicit | Cell state | Hidden state |
| **Gates** | 0 | 3 | 2 |
| **Use case** | Simple sequences | Complex, long sequences | Good middle ground |

---

## Modern Alternatives: Transformers

**2017**: Transformers replaced RNNs for many tasks.

**Key advantages**:
- ✅ Parallel processing (RNNs are sequential)
- ✅ Better at long-range dependencies
- ✅ Attention mechanism
- ✅ Faster training

**Transformers** power GPT, BERT, ChatGPT, etc.

But RNNs/LSTMs still useful for:
- Time series forecasting
- Real-time streaming data
- Low-resource environments

---

## 📹 Recommended Videos

- [The Illustrated LSTM](https://www.youtube.com/watch?v=8HyCNIVRbSU) - Best visual explanation
- [Stanford CS230: RNNs](https://www.youtube.com/watch?v=LHXXI4-IEns)
- [Andrej Karpathy: The Unreasonable Effectiveness of RNNs](https://www.youtube.com/watch?v=iX5V1WpxxkY)

---

## 🎯 Key Takeaways

1. **RNNs** process sequences with hidden state (memory)
2. **Vanishing gradients** prevent RNNs from learning long dependencies
3. **LSTMs** use gates to control information flow
4. **Three gates**: Forget, Input, Output
5. **Cell state** acts as long-term memory
6. **Transformers** have largely replaced RNNs in NLP
7. RNNs still useful for streaming/time-series data

---

## 📹 Recommended Videos

- [Illustrated Guide to RNNs](https://www.youtube.com/watch?v=LHXXI4-IEns) — The A.I. Hacker clear visual walkthrough
- [Understanding LSTM Networks](https://www.youtube.com/watch?v=8HyCNIVRbSU) — The A.I. Hacker LSTM deep dive
- [Recurrent Neural Networks (RNN)](https://www.youtube.com/watch?v=AsNTP8Kwu80) — StatQuest

---

## 📚 Additional Resources

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — Chris Olah's definitive LSTM explainer
- [The Unreasonable Effectiveness of RNNs](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) — Andrej Karpathy's classic blog post
- [Sequence Models](https://d2l.ai/chapter_recurrent-neural-networks/index.html) — Dive into Deep Learning (d2l.ai) chapter

---

## 🚀 Next Lesson

**Lesson 10**: Training Best Practices & Optimization
- Learning rate schedules
- Batch normalization
- Weight initialization strategies
- Advanced optimizers (Adam, AdamW)
- Debugging training

**Final lesson of Module 01!** 🎉
