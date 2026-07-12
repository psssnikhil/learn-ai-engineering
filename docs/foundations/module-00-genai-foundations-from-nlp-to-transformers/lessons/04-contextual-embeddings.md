---
title: From Word2Vec to Contextual Embeddings
description: >-
  Understand the evolution from static word vectors to context-dependent
  representations, and why this shift demanded a new architecture
duration: 35 min
difficulty: intermediate
has_code: false
module: module-00
---
# From Word2Vec to Contextual Embeddings

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand limitations of static embeddings | 35 min | Intermediate |
| Learn how RNNs and LSTMs attempted to solve the context problem | | |
| See why sequential processing was the bottleneck | | |
| Understand why a completely new architecture (Transformers) was needed | | |

---

## The Context Problem Revisited

Static embeddings like Word2Vec assign one vector per word. But language is deeply contextual:

```
"He went to the bank to cash a check"      → bank = financial institution
"He went to the bank to catch a fish"       → bank = river bank
"The bank approved the loan application"    → bank = financial institution
"Flowers grew along the bank of the river"  → bank = river bank
```

We need a model that reads the **entire sentence** and produces a different vector for "bank" depending on surrounding words.

---

## Attempt 1: Recurrent Neural Networks (RNNs)

RNNs process text **one word at a time**, left to right, maintaining a "hidden state" that accumulates information:

```
Input:  "The"  →  "cat"  →  "sat"  →  "on"  →  "the"  →  "mat"
State:  [h0]  →  [h1]   →  [h2]   →  [h3]  →  [h4]   →  [h5]

Each hidden state h_t contains information about all previous words.
```

### RNN Limitations

1. **Vanishing gradients**: Information from early words fades as the sequence gets longer
2. **Sequential processing**: Must process word-by-word, cannot parallelize
3. **Long-range dependencies**: "The cat, which was sitting on the mat near the window of the old house, **was** happy" — by the time the RNN reaches "was", it has largely forgotten "cat"

---

## Attempt 2: LSTMs and GRUs

Long Short-Term Memory (LSTM) networks added "gates" to control what information to keep or forget:

```
LSTM Cell:
  Forget gate:  What old information to discard
  Input gate:   What new information to store
  Output gate:  What to output from the current state
```

LSTMs improved long-range memory significantly, but still suffered from:
- **Sequential processing** (still one word at a time)
- **Limited context window** (practical limit of ~500 tokens)
- **Slow training** (cannot parallelize across GPUs effectively)

---

## Attempt 3: Bidirectional Models (ELMo, 2018)

ELMo ran two LSTMs — one forward, one backward — and combined their outputs:

```
Forward:   "The" → "cat" → "sat" → "on" → "bank"
Backward:  "bank" ← "on" ← "sat" ← "cat" ← "The"

Combined: each word sees context from BOTH directions
```

ELMo produced genuinely contextual embeddings — "bank" got different vectors in financial vs river contexts. But it was **still slow** because of sequential processing.

---

## The Fundamental Bottleneck

| Model | Context | Parallelization | Training Speed |
|-------|---------|----------------|---------------|
| Word2Vec | None (static) | N/A | Fast |
| RNN | Left-to-right only | None (sequential) | Slow |
| LSTM | Better long-range | None (sequential) | Slow |
| Bi-LSTM (ELMo) | Both directions | Limited | Slow |
| **Transformer** | **Full context, all at once** | **Fully parallel** | **Fast** |

The question that led to Transformers was:

> **Can we build a model that looks at ALL words simultaneously (not one at a time) and figures out which words are relevant to each other?**

The answer was the **Attention mechanism** — the subject of the next lesson.

---

## Why This History Matters

Understanding the evolution helps you appreciate **why** Transformers were designed the way they were:

- **Self-attention** replaces sequential processing → enables parallelization
- **Positional encoding** replaces the implicit position that RNNs get from sequential order
- **Multi-head attention** allows attending to multiple types of relationships simultaneously

Every design choice in the Transformer architecture is a direct response to limitations of the RNN/LSTM era.

---

## Key Takeaways

- Static embeddings (Word2Vec) cannot handle polysemy — one vector per word, no context
- RNNs process sequentially and struggle with long-range dependencies
- LSTMs improved memory but remained fundamentally sequential
- ELMo achieved contextual embeddings but was slow due to sequential processing
- The Transformer architecture was designed to solve the parallelization bottleneck while maintaining full context

## Resources

- [YouTube: Illustrated Guide to RNNs](https://www.youtube.com/watch?v=LHXXI4-IEns) -- Visual explanation of recurrent neural networks
- [YouTube: LSTM Explained](https://www.youtube.com/watch?v=YCzL96nL7j0) -- StatQuest's clear LSTM walkthrough
- [Jay Alammar: Visualizing Neural Machine Translation](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) -- Visual guide to sequence-to-sequence models with attention
- [YouTube: Stanford CS224N — RNNs and Language Models](https://www.youtube.com/watch?v=PLryWeHPcBs) -- Stanford NLP lecture on sequential models

---

Next: The Attention Mechanism — "Attention Is All You Need"
