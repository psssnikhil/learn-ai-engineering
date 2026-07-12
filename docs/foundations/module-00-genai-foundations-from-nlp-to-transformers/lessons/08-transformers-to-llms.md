---
title: From Transformers to Large Language Models
description: >-
  Understand how the Transformer architecture became the foundation for modern
  LLMs — pre-training, fine-tuning, RLHF, and the path from GPT-1 to today
duration: 40 min
difficulty: intermediate
has_code: false
module: module-00
---
# From Transformers to Large Language Models

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand how Transformers became LLMs | 40 min | Intermediate |
| Learn the pre-training and fine-tuning paradigm | | |
| Understand RLHF and instruction tuning | | |
| See the timeline from GPT-1 to modern models | | |
| Connect foundations to the rest of this curriculum | | |

---

## The Key Insight: Pre-Training on Language

The Transformer was originally designed for machine translation. But researchers quickly realized something profound:

> **A model that can predict the next word must implicitly learn grammar, facts, reasoning, and world knowledge.**

This led to the **pre-training paradigm**: train a large Transformer on massive amounts of text, then adapt it to specific tasks.

```
Pre-training task: predict the next token

"The capital of France is ___"  → "Paris"
"Water freezes at ___"          → "0" (then "degrees")
"def fibonacci(n):
___"        → "    if n <= 1:"

To get these right, the model must learn:
  - Geography (France → Paris)
  - Physics (water → freezing point)
  - Programming (function definition → body)
  - Grammar, style, reasoning, common sense...
```

---

## The GPT Lineage

### GPT-1 (2018): "Improving Language Understanding by Generative Pre-Training"

- 117 million parameters, 12 layers
- Pre-trained on BookCorpus (7,000 unpublished books)
- Showed that pre-training + fine-tuning beats task-specific models
- Key insight: **generative pre-training transfers to many tasks**

### GPT-2 (2019): "Language Models are Unsupervised Multitask Learners"

- 1.5 billion parameters, 48 layers
- Pre-trained on WebText (40GB of web text)
- Could generate coherent paragraphs of text
- Key insight: **scaling up improves quality dramatically** (and zero-shot performance emerges)

### GPT-3 (2020): "Language Models are Few-Shot Learners"

- 175 billion parameters, 96 layers
- Pre-trained on 300 billion tokens (Common Crawl, books, Wikipedia)
- Demonstrated **in-context learning**: perform tasks from a few examples in the prompt, no fine-tuning needed
- Key insight: **at sufficient scale, the model can learn from the prompt itself**

```
GPT-3 in-context learning:

Prompt: "Translate English to French:
         sea otter → loutre de mer
         cheese → fromage
         computer →"

Output: "ordinateur"

No training. No fine-tuning. Just examples in the prompt.
```

### GPT-4 and Beyond (2023+)

- Estimated hundreds of billions to trillions of parameters
- Multimodal (text + images)
- Dramatically improved reasoning, coding, and instruction following
- Combined with RLHF for alignment

---

## The BERT Alternative: Encoder-Only Models

While GPT focused on generation (decoder-only), Google took a different path with **BERT** (2018):

| | GPT (Decoder-only) | BERT (Encoder-only) |
|---|---|---|
| **Direction** | Left-to-right only | Bidirectional (sees full context) |
| **Pre-training** | Predict next token | Predict masked tokens + next sentence |
| **Strength** | Text generation | Text understanding and classification |
| **Use cases** | Chatbots, writing, code gen | Search, NER, sentiment, question answering |

```
BERT pre-training:

Input:  "The [MASK] sat on the [MASK]"
Target: "The  cat   sat on the  mat"

By predicting masked words from BOTH directions,
BERT builds deep bidirectional understanding.
```

BERT dominated NLP benchmarks from 2018-2022 for understanding tasks (search, classification, entity recognition). GPT-style models eventually surpassed BERT at these tasks too — through sheer scale and instruction tuning.

---

## Making LLMs Useful: Instruction Tuning and RLHF

A raw pre-trained model is a next-token predictor. It will complete text, but it does not follow instructions or have a conversation. Two techniques transformed base models into useful assistants:

### Step 1: Supervised Fine-Tuning (SFT)

Train the model on high-quality instruction-response pairs:

```
Instruction: "Summarize the following article in 3 bullet points..."
Response:    "• Point 1
• Point 2
• Point 3"

Instruction: "Write a Python function to sort a list..."
Response:    "def sort_list(lst):
    return sorted(lst)"
```

This teaches the model the **format** of being helpful — following instructions, answering questions, and staying on topic.

### Step 2: Reinforcement Learning from Human Feedback (RLHF)

```
1. Generate multiple responses to the same prompt
2. Have humans rank them (best → worst)
3. Train a "reward model" to predict human preferences
4. Use the reward model to fine-tune the LLM via reinforcement learning

Prompt: "Explain quantum computing"

Response A: (clear, accurate, well-structured)    → Rank 1 ✓
Response B: (correct but rambling)                 → Rank 2
Response C: (contains errors)                      → Rank 3

The reward model learns: "humans prefer A-style responses"
The LLM is then optimized to produce more A-style outputs.
```

RLHF is what makes modern models:
- **Helpful**: they try to answer the question asked
- **Honest**: they express uncertainty rather than fabricating
- **Harmless**: they decline dangerous or unethical requests

---

## The Modern LLM Training Pipeline

```
Stage 1: PRE-TRAINING
  Data: trillions of tokens from the internet, books, code
  Task: predict the next token
  Result: base model (strong at completing text, not at following instructions)
  Cost: millions of dollars in compute

Stage 2: SUPERVISED FINE-TUNING (SFT)
  Data: ~100K high-quality instruction-response pairs
  Task: learn to follow instructions
  Result: instruction-tuned model (follows instructions but may still be rough)
  Cost: thousands of dollars

Stage 3: RLHF / DPO / Constitutional AI
  Data: human preference rankings or AI feedback
  Task: align with human values and preferences
  Result: aligned model (helpful, honest, safe)
  Cost: tens of thousands of dollars

Stage 4: DEPLOYMENT
  Serving: optimized inference (quantization, batching, caching)
  Safety: content filters, rate limiting, monitoring
```

---

## The Current Landscape

| Family | Creator | Architecture | Key Features |
|--------|---------|-------------|-------------|
| GPT-4o | OpenAI | Decoder-only | Multimodal, strong reasoning |
| Claude | Anthropic | Decoder-only | Constitutional AI alignment, long context |
| Gemini | Google | Decoder-only | Multimodal, long context |
| LLaMA | Meta | Decoder-only | Open weights, widely fine-tuned |
| Mistral | Mistral AI | Decoder-only | Efficient, open weights |

Every one of these is a Transformer. The architecture you learned in the previous lessons is the engine powering all of them. The differences are in:
- **Training data**: what text they learned from
- **Scale**: how many parameters and tokens
- **Alignment**: how they were fine-tuned for safety and helpfulness
- **Optimizations**: architectural tweaks like RoPE, GQA, MoE

---

## Connecting to the Rest of This Curriculum

You now have the foundation to understand everything that follows:

| Curriculum Topic | Connection to Foundations |
|-----------------|------------------------|
| **Prompt Engineering** | You are crafting inputs to the Transformer's attention mechanism |
| **RAG** | Feeding retrieved context into the model's input embeddings |
| **Fine-Tuning** | Continuing gradient descent on the model's weights |
| **Embeddings & Vector Search** | Using the Transformer's internal representations for similarity |
| **Agents** | Chaining multiple forward passes with tool use |
| **Evaluation** | Measuring the quality of the model's probability distributions |
| **Deployment** | Optimizing the Transformer's inference (quantization, KV cache, batching) |

The Transformer is not just background knowledge — it is the foundation of every topic in this curriculum.

---

## Key Takeaways

- Pre-training on next-token prediction forces the model to learn language, facts, and reasoning
- Scale is a key driver: GPT-1 (117M) → GPT-3 (175B) showed dramatic capability jumps
- BERT (encoder-only) excels at understanding; GPT (decoder-only) excels at generation
- Instruction tuning and RLHF transform base models into useful assistants
- The modern training pipeline has four stages: pre-training, SFT, RLHF, deployment
- Every major LLM today is a Transformer — the architecture from the previous lessons
- These foundations directly connect to every advanced topic in this curriculum

## Resources

- [YouTube: 3Blue1Brown — How Might LLMs Store Facts](https://www.youtube.com/watch?v=9-Jl0dxWQs8) -- How Transformers encode knowledge in their parameters
- [YouTube: Andrej Karpathy — Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) -- One-hour overview of LLMs from the former head of AI at Tesla (excellent)
- [YouTube: RLHF Explained](https://www.youtube.com/watch?v=2MBJOuVq380) -- How reinforcement learning from human feedback aligns language models
- [Jay Alammar: How GPT-3 Works](https://jalammar.github.io/how-gpt3-works-visualizations-animations/) -- Visual and animated explanation of GPT-3
- [Anthropic Research: Constitutional AI](https://www.anthropic.com/research/constitutional-ai-harmlessness-from-ai-feedback) -- How Anthropic aligns Claude using AI feedback

---

Congratulations — you now have a solid foundation in the core concepts that power modern AI. The rest of this curriculum builds directly on what you have learned here.
