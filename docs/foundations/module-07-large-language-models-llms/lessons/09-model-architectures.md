---
title: 'Model Architectures - GPT, BERT, T5, LLaMA'
description: Compare major LLM architectures and understand their design choices
duration: 30 min
difficulty: intermediate
has_code: false
module: module-07
youtube: 'https://www.youtube.com/watch?v=MN__lSncZBs'
---
# LLM Architectures Comparison

## 1. GPT (Generative Pre-trained Transformer)

**Type**: Decoder-only
**Training**: Autoregressive (next token prediction)
**Best for**: Text generation, completion

**Architecture**:
```
Tokens → Embeddings → Decoder Blocks × N → Output
```

**Models**:
- GPT-1: 117M params (2018)
- GPT-2: 1.5B params (2019)
- GPT-3: 175B params (2020)
- GPT-4: ~1.7T params (2023)

## 2. BERT (Bidirectional Encoder Representations)

**Type**: Encoder-only
**Training**: Masked language modeling
**Best for**: Classification, NER, QA

**Architecture**:
```
Tokens → Embeddings → Encoder Blocks × N → [CLS] → Classification
```

**Key feature**: Bidirectional context (sees both directions)

**Models**:
- BERT-Base: 110M (12 layers, 768 hidden)
- BERT-Large: 340M (24 layers, 1024 hidden)
- RoBERTa: Optimized BERT
- ALBERT: Parameter-efficient BERT

## 3. T5 (Text-to-Text Transfer Transformer)

**Type**: Encoder-Decoder
**Training**: Span corruption
**Best for**: All NLP tasks (as text-to-text)

**Everything is text-to-text**:
```
Translation: "translate English to German: Hello" → "Hallo"
Summarization: "summarize: Long text..." → "Short summary"
QA: "question: What is AI? context: ..." → "Answer"
```

**Models**:
- T5-Small: 60M
- T5-Base: 220M
- T5-Large: 770M
- T5-11B: 11B

## 4. LLaMA (Large Language Model Meta AI)

**Type**: Decoder-only (like GPT)
**Training**: Autoregressive
**Best for**: Open-source foundation

**Key innovations**:
- RMSNorm instead of LayerNorm
- SwiGLU activation
- Rotary positional embeddings (RoPE)

**Models**:
- LLaMA-7B, 13B, 33B, 65B (2023)
- LLaMA-2: 7B, 13B, 70B (2023)
- LLaMA-3: 8B, 70B, 405B (2024)

## 5. Mistral & Mixtral

**Mistral-7B**: High-performance 7B model
**Mixtral-8x7B**: Mixture of Experts (MoE)
- 8 expert networks
- Only 2 active per token
- 47B total, 13B active

## Quick Comparison

| Model | Type | Size | Use Case |
|-------|------|------|----------|
| GPT | Decoder | Up to 1.7T | Generation |
| BERT | Encoder | Up to 340M | Understanding |
| T5 | Enc-Dec | Up to 11B | All tasks |
| LLaMA | Decoder | Up to 405B | Open foundation |
| Mistral | Decoder | 7B | Efficient performance |
| Claude | Decoder | ? | Safety-focused |
| Gemini | Multimodal | ? | Google's flagship |

---

## 📹 Recommended Videos

- [GPT, BERT, T5 Compared](https://www.youtube.com/watch?v=MN__lSncZBs) — Architecture comparison and when to use each
- [LLaMA Explained](https://www.youtube.com/watch?v=E5OnoYF2oAk) — Meta's open-source LLM architecture
- [A Survey of LLMs](https://www.youtube.com/watch?v=SZorAJ4I-sA) — Comprehensive overview of major architectures

---

## 📚 Additional Resources

- [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774) — OpenAI's GPT-4 paper
- [LLaMA 2 Paper](https://arxiv.org/abs/2307.09288) — Open foundation and fine-tuned chat models
- [Mistral 7B Paper](https://arxiv.org/abs/2310.06825) — Efficient architecture with grouped-query attention
- [Lilian Weng: LLM-Powered Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) — Survey of LLM architectures and agents
