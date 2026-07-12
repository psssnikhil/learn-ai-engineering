---
title: What is AI Engineering?
description: >-
  Understand the discipline of AI engineering — what it is, how it differs from
  ML engineering and data science, what the modern AI stack looks like, and why
  the skills you build in this curriculum transfer across the rapidly evolving landscape
duration: 45 min
difficulty: beginner
has_code: false
module: module-01
---
# What is AI Engineering?

## Prerequisites

- [Module 00: GenAI Foundations](../../module-00-genai-foundations-from-nlp-to-transformers/lessons/01-prerequisites.md) — or general software engineering background

## What You'll Learn

| Objective | Why It Matters |
|-----------|---------------|
| Define AI engineering precisely | The discipline is new; many job descriptions conflate it with ML and data science |
| Understand the pre-trained model paradigm shift | This explains why the skills needed changed dramatically in 2022 |
| Map the modern AI engineering stack | Helps you choose the right tool for each layer of an AI application |
| Identify what skills transfer across model generations | The landscape changes fast; transferable understanding is the durable asset |

---

## A Precise Definition

**AI Engineering** is the discipline of building reliable, observable, cost-effective software systems that use AI models as components — typically large pre-trained models accessed via API.

The key word is *component*. In AI engineering, a language model (or image model, or embedding model) is a building block — like a database or a message queue — that you integrate, evaluate, and operate. You are not the person who trained the model.

This distinguishes AI engineering from:

- **ML Engineering**: Trains, evaluates, and deploys custom models. Involves PyTorch, data pipelines, distributed training, model optimization. Main output: a trained model.
- **Data Science**: Extracts insights from data. Involves statistics, visualization, hypothesis testing. Main output: analysis and recommendations.
- **AI Engineering**: Builds applications on top of pre-trained models. Involves APIs, prompt design, retrieval systems, evaluation, and production engineering. Main output: a working software system.

| Aspect | AI Engineer | ML Engineer | Data Scientist |
|--------|-------------|-------------|----------------|
| **Core question** | How do I build a reliable system with this model? | How do I train a better model? | What does this data tell us? |
| **Primary tools** | APIs, vector DBs, orchestration frameworks | PyTorch, TensorFlow, HPC clusters | Python, SQL, Jupyter |
| **Model relationship** | Consumer of pre-trained models | Producer of models | Consumer of pre-trained models |
| **Evaluation focus** | End-to-end system quality, latency, cost | Model metrics (loss, accuracy, perplexity) | Statistical significance, business KPIs |
| **Data requirement** | Moderate (for evaluation and retrieval) | Large (for training) | Variable |

!!! note "The Lines Are Blurry"
    In practice, many teams need overlap. A startup AI engineer may fine-tune models occasionally. An ML engineer may consume APIs for rapid prototyping. The distinction is useful for understanding skill focus, not as a rigid boundary.

---

## Why 2022 Changed the Skill Requirements

Before 2022, building AI systems required training your own models. This meant:
- Collecting and labeling thousands to millions of domain-specific examples
- Deep ML expertise to design model architectures and training pipelines
- Significant compute budgets
- 6–18 month development cycles before a working prototype

After GPT-3 became widely available via API (2021) and especially after ChatGPT demonstrated the general-public viability of instruction-following models (2022), the economics changed:

```
Before 2022 (ML-centric):
  Problem: "Classify customer support tickets by category"
  Solution: Collect 50K labeled tickets → fine-tune BERT → deploy model service
  Timeline: 3-6 months
  Cost: significant

After 2022 (AI engineering):
  Problem: "Classify customer support tickets by category"
  Solution: Write a clear classification prompt → call GPT-4o-mini API → evaluate
  Timeline: 1-2 days
  Cost: pennies per thousand tickets at current API pricing
```

This shift did not eliminate ML engineering — training custom models still produces better results for specific domains with sufficient data. But it changed *when* AI engineering (via APIs) is the right first approach.

---

## The Modern AI Engineering Stack

Understanding the stack helps you reason about where a problem lives and what tools to reach for:

```
┌──────────────────────────────────────────────────────┐
│  APPLICATION LAYER                                    │
│  Your product: web app, mobile app, API, CLI         │
│  Handles: UX, session management, business logic     │
└───────────────────────────┬──────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────┐
│  ORCHESTRATION LAYER                                  │
│  Controls: LLM calls, tool use, multi-step workflows  │
│  Examples: LangChain, LlamaIndex, custom Python code  │
│  Patterns: RAG, agents, chain-of-thought, routing    │
└───────────────────────────┬──────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────┐
│  MODEL LAYER                                          │
│  The intelligence: text, code, vision, embeddings    │
│  Examples:                                            │
│    OpenAI: GPT-4o, GPT-4o-mini, text-embedding-3    │
│    Anthropic: Claude 4 Opus/Sonnet/Haiku             │
│    Google: Gemini 2.5 Pro/Flash                      │
│    Open weights: LLaMA 3, Mistral, Qwen              │
└───────────────────────────┬──────────────────────────┘
                            │
┌───────────────────────────▼──────────────────────────┐
│  DATA LAYER                                           │
│  Where information lives and how it is retrieved     │
│  Vector databases: Pinecone, Chroma, pgvector        │
│  Traditional: PostgreSQL, Redis, S3                  │
│  Streaming: Kafka, Kinesis (for real-time events)    │
└──────────────────────────────────────────────────────┘
```

### What AI Engineers Own

You will typically design and build:
- **Prompt templates**: how to structure inputs to the model
- **Retrieval pipelines**: how to find relevant context (RAG)
- **Agent loops**: how the model decides what tools to use
- **Evaluation harnesses**: how to measure if the system is working
- **Observability**: logging, cost tracking, latency monitoring
- **Fallback logic**: what to do when the model fails or returns poor output

You will typically *not* own:
- The model's weights
- The model's training process
- The hosting infrastructure (unless self-hosting open weights)

---

## What You Will Build in This Curriculum

By the end of this program, you will have built production-grade versions of:

**RAG Systems (Modules 3–5):** Embed a document corpus, store embeddings in a vector database, retrieve relevant chunks at query time, and feed them into an LLM prompt. This is the pattern behind document Q&A, internal knowledge search, and customer support bots.

**AI Agents (Modules 6–8):** Systems where the model decides which tools to call, observes the results, and continues until it has an answer. You will build both single-agent and multi-agent architectures.

**Evaluation Pipelines (Module 9):** Systematic ways to measure system quality. The gap between a working demo and a production system is almost always an evaluation gap — you cannot improve what you do not measure.

**Production Deployment (Module 10):** Cost optimization, observability, rate limiting, graceful degradation, and latency management.

---

## Why Understanding Foundations Matters

You might wonder: "If I'm just calling APIs, why did I spend a module on Transformer internals?"

Three reasons:

**Debugging**: When a model gives wrong answers, you need to understand whether the issue is in the prompt (model has the information but is guided wrong), retrieval (model does not have the right context), or model capability (the task exceeds what the model can do). Knowing how attention and in-context learning work helps you diagnose this.

**Informed design**: Understanding that context window has O(n²) cost means you know why chunking strategies matter. Understanding that models are probabilistic next-token predictors means you know why few-shot examples work and why deterministic-seeming prompts sometimes fail.

**Transferability**: Models change rapidly. The API you use today may be deprecated in 18 months. The underlying architecture and training paradigm — which you now understand — is far more stable. An engineer who understands why attention works will adapt to new model families faster than one who only knows how to call a specific API.

---

## The Evaluation-First Mindset

The most important mindset shift for AI engineering:

> **Build your evaluation harness before you build your system.**

In traditional software engineering, you write unit tests before or alongside code. In AI engineering, you need something analogous: a set of test cases with known correct behavior that you can run against your system.

Without this, you are flying blind. You cannot tell if a prompt change made things better or worse. You cannot compare model versions. You cannot catch regressions.

```python
# A minimal evaluation setup for a document Q&A system
eval_cases = [
    {
        "question": "What is the company's refund policy?",
        "expected_answer_contains": "30 days",
        "document": "returns_policy.pdf",
    },
    {
        "question": "Who is the CEO?",
        "expected_answer_contains": "Jane Smith",
        "document": "about_us.pdf",
    },
]

def evaluate_rag_system(system, eval_cases):
    results = []
    for case in eval_cases:
        answer = system.query(case["question"], case["document"])
        passed = case["expected_answer_contains"].lower() in answer.lower()
        results.append({"passed": passed, "answer": answer, "case": case})
    return results
```

This mindset — evaluation first, then build — is what separates AI engineers who produce reliable systems from those who produce impressive demos.

---

## Edge Cases and Misconceptions

**"AI engineering is just prompt engineering."** Prompt design is one component. Production AI engineering also involves retrieval architecture, evaluation design, cost management, observability, latency optimization, and integration with existing systems. Most of the engineering surface area is not in the prompt.

**"Better model = better system."** Not necessarily. A well-designed RAG pipeline with GPT-4o-mini often outperforms a poor RAG pipeline with GPT-4o. The retrieval quality, prompt structure, and evaluation rigor matter at least as much as model quality.

**"Open-source models will always be behind proprietary ones."** This was true in 2020. By 2024, open models like LLaMA 3 70B are competitive with GPT-3.5-class models. The right model choice depends on cost, latency, privacy requirements, and the specific task — not just raw capability.

**"API rate limits are a minor inconvenience."** At production scale, rate limiting is a systems design problem. A document processing pipeline that hits rate limits every thousand requests needs exponential backoff, token bucket algorithms, and possibly multiple provider accounts.

---

## Production Connection

The skills in this module directly apply to production systems:

| Skill | Where It Appears |
|-------|-----------------|
| Stack understanding | Architecture decisions: which layer handles what |
| Evaluation mindset | CI/CD for AI: blocking deploys on quality regression |
| Model selection | Cost optimization: choosing the cheapest model adequate for the task |
| Orchestration | Multi-step workflows: agents, RAG, structured generation |

---

## Key Takeaways

- AI engineering is building software systems with pre-trained models as components — distinct from training models (ML engineering) or analyzing data (data science)
- The 2022 shift to instruction-following models via API dramatically reduced the barrier to building AI-powered systems
- The modern AI stack has four layers: application, orchestration, model, and data
- Understanding Transformer internals helps you debug, design better systems, and adapt as the model landscape evolves
- The most important mindset shift: build your evaluation harness before building the system

---

## Further Reading

- [Chip Huyen: AI Engineering](https://huyenchip.com/2024/07/25/genai-platform.html) — practical overview of the full GenAI platform engineering stack
- [Latent Space: The AI Engineer](https://www.latent.space/p/ai-engineer) — the original essay defining the AI engineer role
- [Simon Willison's AI Notes](https://simonwillison.net/tags/ai/) — practical, engineering-focused observations on working with LLMs in production

---

**Next:** [Your First AI Application](02-first-ai-application.md)
