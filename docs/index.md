---
title: AI Engineering Handbook
description: Open-source knowledge base for learning AI engineering
---

# AI Engineering Handbook

Learn to build **production AI systems** — RAG, agents, tool use, and LLMOps.

## Start learning

| Step | Where to go |
|------|-------------|
| 1 | [Learning Path](learning-path.md) — full module order |
| 2 | [M00 · GenAI Foundations](foundations/module-00-genai-foundations-from-nlp-to-transformers/index.md) — first module |
| 3 | [Resources](resources/index.md) — papers, videos, tools |

## Four phases

| Phase | Overview | Topics |
|-------|----------|--------|
| [Foundations](foundations/index.md) | 5 modules | NLP, neural nets, transformers, LLMs |
| [Build](build/index.md) | 5 modules | RAG, agents, vector DBs, prompts |
| [Production](production/index.md) | 2 modules | LLMOps, safety & ethics |
| [Advanced](advanced/index.md) | 2 modules | Fine-tuning, capstones |

## How this repo is organized

```
docs/
  foundations/     → module-00, 01, 05, 06, 07
  build/           → module-09, 11, 12, 13, 14
  production/      → module-10, 16
  advanced/        → module-15, 17

  Each module/
    index.md       → overview + lesson table
    lessons/       → markdown lessons
    exercises/     → Python files (where available)
```

!!! tip "Module numbers match the platform"
    Folder names use canonical IDs (`module-09-rag-...`) so content stays in sync with the AI Engineering Mastery course.

## Contribute

See [CONTRIBUTING.md](https://github.com/psssnikhil/learn-ai-engineering/blob/main/CONTRIBUTING.md) on GitHub.
