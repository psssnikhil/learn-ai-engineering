---
title: The Complete Transformer Architecture
description: Understand the full Transformer model that revolutionized AI
duration: 45 min
difficulty: advanced
has_code: true
module: module-06
youtube: 'https://www.youtube.com/watch?v=4Bdc55j80l8'
---
# The Complete Transformer Architecture

## Overview

```
Input → Embedding → Encoder → Decoder → Output
                      ↓           ↓
                  (N layers)  (N layers)
```

**"Attention is All You Need"** (2017)
- No RNNs, no CNNs
- Just attention + feed-forward networks
- Parallel processing
- State-of-the-art results

---

## Encoder Block

```
Input
  ↓
Multi-Head Self-Attention
  ↓
Add & Norm (Residual + Layer Norm)
  ↓
Feed-Forward Network
  ↓
Add & Norm
  ↓
Output
```

**Repeated N times** (BERT uses N=12)

## Implementation

```python
class TransformerEncoderLayer:
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        # Feed-forward
        self.ff = FeedForward(d_model, d_ff)
        
        self.dropout = dropout
    
    def forward(self, x, mask=None):
        # Multi-head attention
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))  # Residual
        
        # Feed-forward
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))  # Residual
        
        return x


class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.01
        self.W2 = np.random.randn(d_ff, d_model) * 0.01
        self.b1 = np.zeros(d_ff)
        self.b2 = np.zeros(d_model)
    
    def forward(self, x):
        # FFN(x) = max(0, xW1 + b1)W2 + b2
        hidden = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        output = hidden @ self.W2 + self.b2
        return output
```

---

## Decoder Block

**Added component**: **Masked** self-attention
- Can't look at future tokens!
- Also has cross-attention to encoder outputs

```
Input
  ↓
Masked Multi-Head Self-Attention
  ↓
Add & Norm
  ↓
Multi-Head Cross-Attention (attends to encoder)
  ↓
Add & Norm
  ↓
Feed-Forward
  ↓
Add & Norm
  ↓
Output
```

---

## Full Transformer

```python
class Transformer:
    def __init__(self, vocab_size, d_model=512, num_heads=8, 
                 num_layers=6, d_ff=2048, max_seq_len=5000):
        
        # Embeddings
        self.embedding = Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(max_seq_len, d_model)
        
        # Encoder layers
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]
        
        # Decoder layers
        self.decoder_layers = [
            TransformerDecoderLayer(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]
        
        # Output
        self.output_proj = Linear(d_model, vocab_size)
    
    def encode(self, src, src_mask=None):
        x = self.embedding(src) + self.pos_encoding(src)
        
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        
        return x
    
    def decode(self, tgt, memory, tgt_mask=None, memory_mask=None):
        x = self.embedding(tgt) + self.pos_encoding(tgt)
        
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        
        return x
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode
        memory = self.encode(src, src_mask)
        
        # Decode
        output = self.decode(tgt, memory, tgt_mask, src_mask)
        
        # Project to vocabulary
        logits = self.output_proj(output)
        
        return logits
```

---

## Key Components Summary

| Component | Purpose |
|-----------|---------|
| **Embeddings** | Convert tokens to vectors |
| **Positional Encoding** | Add position information |
| **Multi-Head Attention** | Capture relationships |
| **Feed-Forward** | Process individually |
| **Layer Norm** | Stabilize training |
| **Residual Connections** | Enable deep networks |
| **Masking** | Control information flow |

---

## Hyperparameters

**BERT Base**:
- Layers: 12
- Hidden size: 768
- Attention heads: 12
- Parameters: 110M

**GPT-3**:
- Layers: 96
- Hidden size: 12,288
- Attention heads: 96
- Parameters: 175B

---

## 🎯 Key Takeaways

1. **Encoder-Decoder** architecture for seq2seq tasks
2. **Self-attention** in encoder, **cross-attention** in decoder
3. **Residual connections** + **Layer Norm** enable deep networks
4. **Parallel** processing (no sequential bottleneck)
5. **Scalable** to billions of parameters
6. Foundation of **all modern LLMs**

---

## 📚 Additional Resources

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — The best visual guide to the full architecture
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Original paper by Vaswani et al.
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — Line-by-line implementation
