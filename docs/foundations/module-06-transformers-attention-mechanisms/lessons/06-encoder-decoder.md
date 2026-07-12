---
title: Encoder-Decoder Architecture
description: Deep dive into encoder-decoder design for sequence-to-sequence tasks
duration: 30 min
difficulty: advanced
has_code: true
module: module-06
youtube: 'https://www.youtube.com/watch?v=L8HKweZIOmg'
---
# Encoder-Decoder Architecture

## Purpose

**Sequence-to-Sequence tasks**:
- Machine translation
- Summarization
- Question answering
- Text generation with context

```
Input (English):  "Hello world"
      ↓ Encoder
   Context Vector
      ↓ Decoder
Output (French):  "Bonjour monde"
```

## Encoder: Understanding the Input

**Job**: Create rich representations of input

```python
class Encoder:
    def forward(self, src):
        # src: (batch, src_len)
        
        # Embed + position
        x = self.embed(src) + self.pos_enc(src)
        
        # Stack of encoder layers
        for layer in self.layers:
            x = layer(x)  # Self-attention + FFN
        
        return x  # Memory: (batch, src_len, d_model)
```

## Decoder: Generating Output

**Job**: Generate output token-by-token, attending to encoder

```python
class Decoder:
    def forward(self, tgt, memory):
        # tgt: (batch, tgt_len)
        # memory: encoder output
        
        x = self.embed(tgt) + self.pos_enc(tgt)
        
        for layer in self.layers:
            # 1. Masked self-attention (can't see future)
            x = layer.self_attn(x, mask=look_ahead_mask)
            
            # 2. Cross-attention to encoder
            x = layer.cross_attn(query=x, key=memory, value=memory)
            
            # 3. Feed-forward
            x = layer.ffn(x)
        
        return x
```

## Training vs Inference

**Training**: Teacher forcing (feed correct previous tokens)
**Inference**: Autoregressive (feed generated tokens)

---

## 📹 Recommended Videos

- [Encoder-Decoder Architecture Explained](https://www.youtube.com/watch?v=L8HKweZIOmg) — Sequence-to-sequence models visually explained
- [Attention Is All You Need (Paper Walkthrough)](https://www.youtube.com/watch?v=XowwKOAWYoQ) — Yannic Kilcher paper review
- [Seq2Seq with Attention](https://www.youtube.com/watch?v=XXtpJxZBa2c) — Jay Alammar's visual guide

---

## 📚 Additional Resources

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Jay Alammar's famous visual guide
- [Seq2Seq Models with Attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) — Visual walkthrough
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — The original transformer paper
