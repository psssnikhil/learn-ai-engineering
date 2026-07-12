---
title: Implementing Attention from Scratch
description: Build a complete attention mechanism in pure NumPy
duration: 60 min
difficulty: advanced
has_code: true
module: module-06
youtube: 'https://www.youtube.com/watch?v=eMlx5fFNoYc'
---
# Implementing Attention from Scratch

## Complete Multi-Head Attention

```python
import numpy as np

class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Weight matrices
        self.W_q = np.random.randn(d_model, d_model) * 0.01
        self.W_k = np.random.randn(d_model, d_model) * 0.01
        self.W_v = np.random.randn(d_model, d_model) * 0.01
        self.W_o = np.random.randn(d_model, d_model) * 0.01
    
    def split_heads(self, x):
        batch, seq_len, d_model = x.shape
        x = x.reshape(batch, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # (batch, heads, seq, d_k)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # Linear projections
        Q = query @ self.W_q
        K = key @ self.W_k
        V = value @ self.W_v
        
        # Split into heads
        Q = self.split_heads(Q)  # (batch, heads, seq_q, d_k)
        K = self.split_heads(K)  # (batch, heads, seq_k, d_k)
        V = self.split_heads(V)  # (batch, heads, seq_v, d_k)
        
        # Scaled dot-product attention
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores += mask * -1e9
        
        attention = self.softmax(scores)
        context = attention @ V  # (batch, heads, seq_q, d_k)
        
        # Concatenate heads
        context = context.transpose(0, 2, 1, 3)  # (batch, seq_q, heads, d_k)
        context = context.reshape(batch_size, -1, self.d_model)
        
        # Final linear
        output = context @ self.W_o
        
        return output, attention
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

## Test It

```python
# Create model
mha = MultiHeadAttention(d_model=512, num_heads=8)

# Sample input
batch_size, seq_len = 2, 10
x = np.random.randn(batch_size, seq_len, 512)

# Forward pass
output, attention = mha.forward(x, x, x)

print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")
print(f"Attention shape: {attention.shape}")
```

## Complete Transformer Block

```python
class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.dropout = dropout
    
    def forward(self, x, mask=None):
        # Attention
        attn_out, _ = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout_layer(attn_out))
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout_layer(ffn_out))
        
        return x
```

---

## 📹 Recommended Videos

- [Attention in Transformers Visually Explained](https://www.youtube.com/watch?v=eMlx5fFNoYc) — 3Blue1Brown visual explanation
- [Coding Self-Attention from Scratch](https://www.youtube.com/watch?v=QCJQG4DuHT0) — Sebastian Raschka implementation tutorial
- [Building GPT from Scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Karpathy's complete implementation

---

## 📚 Additional Resources

- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — Harvard NLP step-by-step implementation
- [Lilian Weng: Attention](https://lilianweng.github.io/posts/2018-06-24-attention/) — Comprehensive attention mechanisms blog
- [Distill.pub: Attention and Augmented RNNs](https://distill.pub/2016/augmented-rnns/) — Interactive visual explanation
