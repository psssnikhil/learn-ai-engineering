---
title: Deep Dives
description: >-
  Engineer-grade mathematical and algorithmic explorations.
  Each page takes one mechanism, derives it from scratch, and walks through it
  numerically — no hand-waving.
---

# Deep Dives

These pages go **below the lesson level**. Each one takes a single mechanism,
derives it from first principles, and walks through it with real numbers or
runnable code. Read them when a lesson leaves you asking "but *why* does that work?"

Find deep dives from lesson **Further Reading** links or the list below.

## Available Deep Dives

### Mathematics of Neural Networks

| Page | What you'll understand after |
|------|------------------------------|
| [Attention Math: Full QKV Derivation](attention-math.md) | Why Q, K, V have their shapes; numerical walkthrough of scaled dot-product attention on a 4-word sentence |
| [Backpropagation Calculus](backpropagation-calculus.md) | Chain rule through a 2-layer network by hand; gradient flow shapes; why vanishing gradients happen |

### Tokenization

| Page | What you'll understand after |
|------|------------------------------|
| [Tokenization Internals: BPE](tokenization-internals.md) | How Byte-Pair Encoding builds a vocabulary from scratch; merge rule walkthrough; why "tokenization" ≠ "words" |

---

## How to Use Deep Dives

1. **Read the prerequisite lesson first.** Each deep dive links back to the
   lesson it expands. The lesson gives intuition; the deep dive gives derivation.
2. **Work the numbers yourself.** Each page includes a small Python snippet you
   can run in a notebook or REPL. The numbers are chosen to be small enough to
   verify by hand.
3. **Don't memorize — understand.** The goal is that after the deep dive, you
   can rederive the key equations on a whiteboard. If you can't, read again.

---

## Planned Deep Dives

| Topic | Status |
|-------|--------|
| Positional Encodings: sinusoidal vs. RoPE | Planned |
| RLHF: PPO objective derivation | Planned |
| LoRA: low-rank approximation theory | Planned |
| Vector similarity: HNSW graph construction | Planned |
| KV cache: memory arithmetic | Planned |
