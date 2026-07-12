---
title: Multi-Head Attention
description: >-
  Learn how multiple attention heads allow Transformers to focus on different
  aspects simultaneously
duration: 30 min
difficulty: intermediate
has_code: true
youtube: 'https://www.youtube.com/watch?v=mMa2PmYJlCo'
---
# Multi-Head Attention

## The Idea: Multiple Perspectives

**Single-head attention**: One way to look at relationships
**Multi-head attention**: Multiple parallel attention mechanisms!

```
Sentence: "The cat sat on the mat"

Head 1: Focuses on syntax (subject-verb-object)
Head 2: Focuses on semantics (meaning relationships)  
Head 3: Focuses on position (nearby words)
Head 4: Focuses on long-range dependencies
```

Like having multiple experts looking at the same data!

---

## Architecture

```
Input → [Head 1] → Concat →
      → [Head 2] →        → Linear → Output
      → [Head 3] →
      → [Head h] →

Each head: Independent Q, K, V matrices
```

## Implementation

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Separate Q, K, V for each head
        self.W_q = np.random.randn(num_heads, d_model, self.d_k) * 0.01
        self.W_k = np.random.randn(num_heads, d_model, self.d_k) * 0.01
        self.W_v = np.random.randn(num_heads, d_model, self.d_k) * 0.01
        
        # Output projection
        self.W_o = np.random.randn(d_model, d_model) * 0.01
    
    def forward(self, x):
        batch, seq_len, d_model = x.shape
        
        # Apply each head
        head_outputs = []
        for h in range(self.num_heads):
            Q = x @ self.W_q[h]  # (batch, seq_len, d_k)
            K = x @ self.W_k[h]
            V = x @ self.W_v[h]
            
            # Scaled dot-product attention
            scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.d_k)
            weights = self.softmax(scores)
            output = weights @ V
            
            head_outputs.append(output)
        
        # Concatenate heads
        concat = np.concatenate(head_outputs, axis=-1)  # (batch, seq_len, d_model)
        
        # Final linear projection
        output = concat @ self.W_o
        
        return output
```

---

## Why It Works

- **Different subspaces**: Each head learns different patterns
- **Ensemble effect**: Combines multiple views
- **Specialization**: Heads naturally specialize during training
- **Robustness**: If one head fails, others compensate

**Typical values**:
- BERT: 12 heads
- GPT-3: 96 heads!

---

## 📚 Additional Resources

- [Multi-Head Attention Explained](https://jalammar.github.io/illustrated-transformer/#multi-headed-attention) — Visual guide
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — Harvard NLP implementation
- [Distill.pub: Attention](https://distill.pub/2016/augmented-rnns/) — Interactive visualizations
