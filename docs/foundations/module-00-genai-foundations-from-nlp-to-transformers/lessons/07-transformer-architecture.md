---
title: The Complete Transformer Architecture
description: >-
  Understand every component of the Transformer — from input embeddings through
  encoder and decoder stacks — and see how they work together
duration: 50 min
difficulty: intermediate
has_code: true
module: module-00
---
# The Complete Transformer Architecture

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand the full Transformer architecture end-to-end | 50 min | Intermediate |
| Learn what each component does and why it exists | | |
| See how encoder-only, decoder-only, and encoder-decoder variants differ | | |
| Build intuition through interactive resources | | |

---

## Architecture Overview

The original Transformer from "Attention Is All You Need" has two halves: an **Encoder** (reads the input) and a **Decoder** (generates the output). Here is the full picture:

```
INPUT                                          OUTPUT
  ↓                                              ↑
[Input Embedding + Positional Encoding]    [Output Embedding + Positional Encoding]
  ↓                                              ↓
┌─────────────────────┐                   ┌─────────────────────┐
│     ENCODER (×N)    │                   │     DECODER (×N)    │
│                     │                   │                     │
│  Multi-Head         │                   │  Masked Multi-Head  │
│  Self-Attention     │                   │  Self-Attention     │
│       ↓             │                   │       ↓             │
│  Add & LayerNorm    │                   │  Add & LayerNorm    │
│       ↓             │                   │       ↓             │
│  Feed-Forward       │    ─────────►     │  Multi-Head         │
│  Network            │   (cross-attn)    │  Cross-Attention    │
│       ↓             │                   │       ↓             │
│  Add & LayerNorm    │                   │  Add & LayerNorm    │
│                     │                   │       ↓             │
└─────────────────────┘                   │  Feed-Forward       │
                                          │  Network            │
                                          │       ↓             │
                                          │  Add & LayerNorm    │
                                          └─────────────────────┘
                                                 ↓
                                          [Linear + Softmax]
                                                 ↓
                                          OUTPUT PROBABILITIES
```

Let us walk through each component.

---

## Component 1: Input Embeddings

The first step converts tokens (integers) into dense vectors the model can process:

```python
import numpy as np

# Vocabulary of 50,000 tokens, each represented as a 512-dim vector
vocab_size = 50000
d_model = 512

# Embedding matrix: each row is a learned vector for one token
embedding_matrix = np.random.randn(vocab_size, d_model) * 0.02

# Convert token IDs to vectors
token_ids = [1234, 5678, 42, 999]  # "The cat sat on"
embeddings = embedding_matrix[token_ids]  # (4, 512)
print(embeddings.shape)  # (4, 512) — 4 tokens, each a 512-dim vector
```

---

## Component 2: Positional Encoding

Self-attention treats all positions equally — it has no notion of word order. Without positional information, "The cat sat on the mat" and "The mat sat on the cat" would be identical.

Positional encoding adds position information to each embedding:

```python
def positional_encoding(seq_len, d_model):
    """
    Sinusoidal positional encoding from the original Transformer paper.
    Each position gets a unique pattern of sine and cosine values.
    """
    position = np.arange(seq_len)[:, np.newaxis]      # (seq_len, 1)
    dim = np.arange(d_model)[np.newaxis, :]            # (1, d_model)

    # Different frequencies for different dimensions
    angle_rates = 1 / np.power(10000, (2 * (dim // 2)) / d_model)
    angles = position * angle_rates

    # Apply sin to even indices, cos to odd indices
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])

    return pe

pe = positional_encoding(100, 512)

# The model input is: token embedding + positional encoding
# model_input = embeddings + pe[:seq_len]
```

Why sinusoidal? The paper showed that these fixed patterns allow the model to:
- Distinguish positions uniquely
- Generalize to sequence lengths not seen during training
- Learn relative positions (the offset between any two positions has a consistent pattern)

Modern models (GPT, LLaMA) often use **learned** positional embeddings or **rotary positional embeddings (RoPE)** instead, but the purpose is the same.

---

## Component 3: The Encoder Block

Each encoder block has two sub-layers:

### Sub-layer 1: Multi-Head Self-Attention

Every word attends to every other word in the input. This is the bidirectional context step — after this layer, each token's representation contains information from the entire input.

### Sub-layer 2: Feed-Forward Network

A simple two-layer neural network applied to each position independently:

```python
def feed_forward(x, W1, b1, W2, b2):
    """
    Position-wise feed-forward network.
    Applied identically to each token position.
    """
    # Expand to higher dimension, apply ReLU, project back
    hidden = np.maximum(0, x @ W1 + b1)  # (seq_len, d_ff)
    output = hidden @ W2 + b2             # (seq_len, d_model)
    return output

# Typical dimensions:
# d_model = 512 → d_ff = 2048 → d_model = 512
# This 4x expansion gives the model more capacity to process information
```

### Residual Connections and Layer Normalization

Each sub-layer is wrapped in a **residual connection** (add the input back to the output) followed by **layer normalization**:

```python
def add_and_norm(x, sublayer_output, gamma, beta):
    """
    Residual connection + layer normalization.
    This is critical for training deep networks.
    """
    # Residual connection: add input to output
    residual = x + sublayer_output

    # Layer normalization: normalize across the feature dimension
    mean = np.mean(residual, axis=-1, keepdims=True)
    std = np.std(residual, axis=-1, keepdims=True) + 1e-6
    normalized = (residual - mean) / std

    # Learnable scale and shift
    return gamma * normalized + beta
```

Why residual connections? Without them, gradients vanish in deep networks. The residual path ensures gradients can flow directly through the network. This is what allows Transformers to be stacked 96+ layers deep.

---

## Component 4: The Decoder Block

The decoder has three sub-layers:

1. **Masked self-attention**: Each position can only attend to earlier positions (causal mask)
2. **Cross-attention**: Decoder attends to the encoder output (this is how the decoder reads the input)
3. **Feed-forward network**: Same as the encoder

```
Decoder processes "I love" to predict "cats":

Step 1 — Masked Self-Attention:
  "I" attends to: ["I"]
  "love" attends to: ["I", "love"]

Step 2 — Cross-Attention:
  Decoder tokens attend to ALL encoder tokens
  (decoder Query × encoder Keys → weights → encoder Values)

Step 3 — Feed-Forward:
  Process each position independently
```

---

## Component 5: Output Layer

The final linear layer projects from `d_model` dimensions to `vocab_size`, and softmax produces probabilities over the entire vocabulary:

```python
def output_layer(decoder_output, W_vocab, bias):
    """Convert decoder output to vocabulary probabilities."""
    logits = decoder_output @ W_vocab + bias  # (seq_len, vocab_size)

    # Softmax over vocabulary for each position
    probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = probs / probs.sum(axis=-1, keepdims=True)

    return probs  # (seq_len, vocab_size)

# For generation, we take the argmax (or sample) from the last position
# next_token = np.argmax(probs[-1])
```

---

## Putting It All Together

```python
def transformer_encoder_block(x, heads, ff_weights):
    """One encoder block."""
    # Sub-layer 1: Multi-head self-attention
    attn_output = multi_head_attention(x, heads)
    x = layer_norm(x + attn_output)

    # Sub-layer 2: Feed-forward
    ff_output = feed_forward(x, *ff_weights)
    x = layer_norm(x + ff_output)

    return x

def transformer_forward(token_ids, num_layers=6):
    """Simplified Transformer forward pass."""
    # 1. Embed tokens
    x = embed(token_ids)

    # 2. Add positional encoding
    x = x + positional_encoding(len(token_ids), d_model)

    # 3. Pass through N encoder blocks
    for layer in range(num_layers):
        x = transformer_encoder_block(x, encoder_heads[layer], ff_weights[layer])

    return x  # Context-aware representations for each token
```

---

## Three Transformer Variants

The original paper used an encoder-decoder model. Modern LLMs use variations:

| Variant | Architecture | Models | Best For |
|---------|-------------|--------|----------|
| **Encoder-only** | Only the encoder stack | BERT, RoBERTa | Understanding text (classification, NER, search) |
| **Decoder-only** | Only the decoder stack | GPT, Claude, LLaMA | Generating text (chatbots, code, writing) |
| **Encoder-decoder** | Both stacks | T5, BART, original Transformer | Translation, summarization |

```
Encoder-only (BERT):
  Input: "The [MASK] sat on the mat"
  Output: Probability distribution for [MASK] → "cat" (0.82)

Decoder-only (GPT):
  Input: "The cat sat on the"
  Output: Next token → "mat" (0.65)

Encoder-decoder (T5):
  Input (encoder): "translate English to French: The cat sat on the mat"
  Output (decoder): "Le chat s'est assis sur le tapis"
```

GPT and Claude are decoder-only — they use **causal (masked) self-attention** and generate text one token at a time, left to right.

---

## Scale of Modern Transformers

| Model | Parameters | Layers | Heads | d_model | Training Data |
|-------|-----------|--------|-------|---------|--------------|
| Original Transformer | 65M | 6 | 8 | 512 | WMT English-German |
| GPT-2 | 1.5B | 48 | 25 | 1600 | WebText (40GB) |
| GPT-3 | 175B | 96 | 96 | 12288 | Common Crawl + books |
| LLaMA 3 (70B) | 70B | 80 | 64 | 8192 | 15T tokens |

The architecture is fundamentally the same — just scaled up dramatically. Understanding the 65M-parameter original gives you the blueprint for all of these.

---

## Interactive Resources

These two resources provide the best visual and interactive understanding of the Transformer:

- **[Jay Alammar: The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)** — A detailed, visual walkthrough of every component. If you read one thing about Transformers, make it this. The step-by-step diagrams of how attention flows through the architecture are invaluable.

- **[Interactive Transformer Explainer](https://poloclub.github.io/transformer-explainer/)** — An interactive visualization where you can feed text into a GPT-2 model and watch the attention patterns, residual streams, and token probabilities in real time. Built by researchers at Georgia Tech.

---

## Key Takeaways

- The Transformer combines embeddings, positional encoding, multi-head attention, feed-forward networks, residual connections, and layer normalization
- Encoder: bidirectional self-attention (sees all tokens)
- Decoder: causal self-attention (sees only past tokens) + cross-attention to encoder
- Residual connections and layer norm enable training very deep networks
- Modern LLMs (GPT, Claude) use decoder-only architecture
- The same fundamental design from 2017 underlies every major language model today, just at much larger scale

## Resources

- [Jay Alammar: The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) -- The definitive visual guide to the Transformer architecture — read this
- [Interactive Transformer Explainer](https://poloclub.github.io/transformer-explainer/) -- Watch a real GPT-2 model process text interactively
- [YouTube: 3Blue1Brown — Transformers Explained](https://www.youtube.com/watch?v=wjZofJX0v4M) -- 3Blue1Brown's visual explanation of the Transformer architecture
- [YouTube: Andrej Karpathy — Let's Build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) -- Build a Transformer from scratch in PyTorch (2 hours, comprehensive)
- [Attention Is All You Need — Original Paper](https://arxiv.org/abs/1706.03762) -- The 2017 paper that started it all

---

Next: From Transformers to Large Language Models
