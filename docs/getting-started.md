---
title: Getting Started
---

# Getting Started

Welcome to the **AI Engineering Handbook** — a free, open-source path from foundations to production agents.

## Who is this for?

<div class="persona-grid">
  <div class="persona-card">
    <div class="persona-card__title">New to AI</div>
    <div class="persona-card__desc">Software engineer or student, no ML background</div>
    <a href="foundations/module-00-genai-foundations-from-nlp-to-transformers/index.md">M00 GenAI Foundations</a> →
    <a href="foundations/module-01-ai-engineering-essentials/index.md">M01 Essentials</a>
  </div>
  <div class="persona-card">
    <div class="persona-card__title">Know ML, need LLMs</div>
    <div class="persona-card__desc">ML practitioner who needs to catch up on transformers and APIs</div>
    <a href="foundations/module-07-large-language-models-llms/index.md">M07 LLMs</a> →
    <a href="build/module-09-rag-retrieval-augmented-generation/index.md">M09 RAG</a>
  </div>
  <div class="persona-card">
    <div class="persona-card__title">Want to build agents</div>
    <div class="persona-card__desc">Engineer focusing on autonomous AI systems</div>
    <a href="build/module-11-ai-agents-fundamentals/index.md">M11 Agents</a> →
    <a href="build/module-18-agent-harness-tools-runtime/index.md">M18 Harness</a> →
    <a href="build/module-12-multi-agent-systems/index.md">M12 Multi-Agent</a>
  </div>
  <div class="persona-card">
    <div class="persona-card__title">Shipping to production</div>
    <div class="persona-card__desc">Need LLMOps, evals, monitoring, safety</div>
    <a href="production/module-10-llmops-production-systems/index.md">M10 LLMOps</a> →
    <a href="production/module-19-llm-evaluation-quality/index.md">M19 Evals</a>
  </div>
</div>

## Local setup

```bash
git clone https://github.com/psssnikhil/learn-ai-engineering.git
cd learn-ai-engineering
pip install -r requirements.txt
mkdocs serve   # http://127.0.0.1:8000
```

You need **Python 3.10+**. API keys (OpenAI, Anthropic, etc.) only for lessons that call live models — most theory lessons need no keys.

### Run Python exercises

```bash
cd docs/foundations/module-01-ai-engineering-essentials/exercises
python 04-starter.py
```

## How to navigate

1. **[Topic Map](topic-map.md)** — find any concept (RAG, harness, evals, observability)
2. **[Learning Path](learning-path.md)** — full module order with lesson counts
3. **Module index** — each module has a lesson table; open lessons from there
4. **[Resources](resources/index.md)** — papers, videos, tools, open-source hubs
5. **[Glossary](glossary.md)** — quick definitions for every key term

## Depth promise

!!! tip "What to expect in every lesson"
    Each lesson covers **why** before **how**: motivation, mental model, worked example, and a hands-on exercise or code walkthrough. No shallow overviews.

## Inspired by (and links to) great OSS hubs

We curate and extend ideas from community handbooks — not duplicate them:

- [Agents Towards Production](https://github.com/NirDiamant/agents-towards-production) — production agent tutorials
- [Microsoft AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners) — structured agent curriculum
- [Awesome Harness Engineering](https://github.com/ai-boost/awesome-harness-engineering) — harness primitives
- [Awesome Agent Evals](https://github.com/benchflow-ai/awesome-evals) — evaluation spine
- [RAG Techniques](https://github.com/NirDiamant/RAG_Techniques) — RAG pattern catalog
- [LLM Course (mlabonne)](https://github.com/mlabonne/llm-course) — engineer track roadmap

See [Open Source Hubs](resources/open-source-hubs.md) for the full list.

## Contribute

Found a gap? See [Roadmap](roadmap.md) and [Contribute](contribute.md).
