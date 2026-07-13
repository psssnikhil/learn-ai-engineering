---
title: Getting Started
---

# Getting Started

Welcome to the **AI Engineering Handbook** — a free, open-source path from foundations to production agents.

!!! tip "Not sure where to begin?"
    **Main entry:** [Start Here](start-here.md) — persona routing, prerequisites, "I want to learn X" tables, and first projects.
    This page covers local setup and site navigation.

## Who is this for?

<div class="persona-grid">
  <div class="persona-card">
    <div class="persona-card__title">New to AI</div>
    <div class="persona-card__desc">Software engineer or student, no ML background</div>
    <a class="persona-card__cta" href="foundations/module-00-genai-foundations-from-nlp-to-transformers/">GenAI Foundations →</a>
    <a class="persona-card__cta" href="foundations/module-01-ai-engineering-essentials/">AI Essentials →</a>
  </div>
  <div class="persona-card">
    <div class="persona-card__title">Know ML, need LLMs</div>
    <div class="persona-card__desc">ML practitioner who needs to catch up on transformers and APIs</div>
    <a class="persona-card__cta" href="foundations/module-07-large-language-models-llms/">Large Language Models →</a>
    <a class="persona-card__cta" href="build/module-09-rag-retrieval-augmented-generation/">RAG →</a>
  </div>
  <div class="persona-card">
    <div class="persona-card__title">Want to build agents</div>
    <div class="persona-card__desc">Engineer focusing on autonomous AI systems</div>
    <a class="persona-card__cta" href="agent-engineering/">Agent Engineering track →</a>
    <a class="persona-card__cta" href="build/module-11-ai-agents-fundamentals/">AI Agents →</a>
    <a class="persona-card__cta" href="build/module-18-agent-harness-tools-runtime/">Agent Harness →</a>
  </div>
  <div class="persona-card">
    <div class="persona-card__title">Using Claude Code / Cursor</div>
    <div class="persona-card__desc">Skills files, loop engineering, context windows</div>
    <a class="persona-card__cta" href="ai-engineering-2026/">Modern AI (2026) →</a>
    <a class="persona-card__cta" href="ai-engineering-2026/claude-code/">Claude Code →</a>
  </div>
  <div class="persona-card">
    <div class="persona-card__title">Shipping to production</div>
    <div class="persona-card__desc">Need LLMOps, evals, monitoring, safety</div>
    <a class="persona-card__cta" href="production/module-10-llmops-production-systems/">LLMOps →</a>
    <a class="persona-card__cta" href="production/module-19-llm-evaluation-quality/">LLM Evaluation →</a>
  </div>
</div>

## How the handbook is organized

| Layer | What it is | Start here |
|-------|------------|------------|
| **16 modules** | Full curriculum across 4 phases (Foundations → Build → Production → Advanced) | [Learning Path](learning-path.md) |
| **Agent Engineering** | Concise 7-lesson track on loops, harness, tools, orchestration, evals | [Agent Engineering](agent-engineering/index.md) |
| **2026 Skills** | Modern IDE-agent skills: Claude Code, skills files, loop & context engineering | [2026 Skills](ai-engineering-2026/index.md) |
| **Hub pages** | Cross-cutting guides that link modules together | [Agentic AI](agentic-ai/index.md), [Evals & Observability](evals-observability/index.md) |

Module IDs like M00, M05, M09 come from the original platform — **gaps (M02–M04, M08) are normal**, not missing lessons.

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

Use the top tabs in this order depending on your goal:

1. **[Start Here](start-here.md)** — persona, goal, prerequisite chains, first project
2. **[Topic Map](topic-map.md)** — find any concept (RAG, harness, evals, observability)
3. **[Learning Path](learning-path.md)** — full module order with lesson counts
4. **[FAQ](faq.md)** — RAG vs fine-tune vs agents, troubleshooting
5. **[Build These First](projects/build-these.md)** — 10 portfolio projects
6. **[Exercises](exercises/index.md)** — hands-on paths by persona
7. **[Agent Engineering](agent-engineering/index.md)** or **[2026 Skills](ai-engineering-2026/index.md)** — focused tracks
8. **Phase tabs** (Foundations, Build, Production, Advanced) — browse modules in the sidebar
9. **[Glossary](glossary.md)** — quick definitions
10. **[Resources](resources/index.md)** — papers, videos, tools, OSS hubs

## How lessons are structured

Each lesson walks through the concept, a worked example, and what to try hands-on. Capstone modules at the end tie everything together.

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
