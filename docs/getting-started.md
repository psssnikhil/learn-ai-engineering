---
title: Getting Started
---

# Getting Started

Welcome to the **AI Engineering Handbook** — free, open-source, and built for engineers who ship.

!!! tip "Not sure where to begin?"
    **[Start Here](start-here.md)** — persona routing, prerequisites, goal tables, first project.  
    This page is for **local setup** and **how the site works**.

---

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
    <div class="persona-card__desc">ML practitioner catching up on transformers and APIs</div>
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

---

## How the site is organized

| Tab | What it is |
|-----|------------|
| **Start Here** | One page to pick your path |
| **Learn** | 16 courses in order (01 → 16) — lessons live inside each course |
| **Reference** | FAQ, Topic Map, Study Plans, Glossary, Resources |
| **Projects** | Portfolio builds ([Build These First](projects/build-these.md)) |
| **Contribute** | How to help + roadmap |

Optional tracks at the bottom of **Learn**: [Agent Engineering](agent-engineering/index.md) · [Modern AI (2026)](ai-engineering-2026/index.md)

---

## Local setup

```bash
git clone https://github.com/psssnikhil/learn-ai-engineering.git
cd learn-ai-engineering
pip install -r requirements.txt
npm install
mkdocs serve   # http://127.0.0.1:8000
```

**Python 3.10+** required. API keys only for lessons that call live models.

### Run Python exercises

```bash
cd docs/foundations/module-01-ai-engineering-essentials/exercises
python 04-prompt-engineering-starter.py
```

### Full production build (what CI runs)

```bash
npm run build:docs
```

---

## How to navigate (recommended order)

1. **[Start Here](start-here.md)** — persona + first project
2. **[Learn](learn/index.md)** — course 01, then 02, then 03…
3. **[Study Plans](learn/study-plans.md)** — week-by-week if you want structure
4. **[Build These First](projects/build-these.md)** — portfolio projects
5. **[Topic Map](topic-map.md)** — jump to any concept
6. **[FAQ](faq.md)** — decisions + troubleshooting

---

## Inspired by great OSS hubs

We curate and extend — not duplicate:

- [Agents Towards Production](https://github.com/NirDiamant/agents-towards-production)
- [Microsoft AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners)
- [Awesome Harness Engineering](https://github.com/ai-boost/awesome-harness-engineering)
- [Awesome Agent Evals](https://github.com/benchflow-ai/awesome-evals)
- [RAG Techniques](https://github.com/NirDiamant/RAG_Techniques)

Full list: [Open Source Hubs](resources/open-source-hubs.md)

---

## Contribute

Found a gap? [Roadmap](roadmap.md) · [Contribute](contribute.md) · [GitHub Issues](https://github.com/psssnikhil/learn-ai-engineering/issues)

If the handbook helps you — **[star the repo](https://github.com/psssnikhil/learn-ai-engineering)**.
