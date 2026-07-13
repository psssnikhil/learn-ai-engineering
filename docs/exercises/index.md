---
title: Exercises
description: Hands-on paths by persona — starter and solution files per module
---

# Python Exercises

Hands-on starter and solution files live under each module's `exercises/` folder. Use this page to find **the right path for your background**, not just a file list.

!!! tip "New here?"
    Pick your persona in [Start Here](../start-here.md), then return here for practice files.

---

## Exercise paths by persona

### Complete beginner (no CS background)

| Step | Module | Exercise folder | Focus |
|------|--------|-----------------|-------|
| 1 | M00 | — (in-lesson code) | NumPy, vectors — [L1 Prerequisites](../foundations/module-00-genai-foundations-from-nlp-to-transformers/lessons/01-prerequisites.md) |
| 2 | M01 | [exercises/](../foundations/module-01-ai-engineering-essentials/exercises/) | `02` first app, `03` tokens, `04` prompts |
| 3 | M05 | [exercises/](../foundations/module-05-neural-networks-deep-learning-fundamentals/exercises/) | `01`–`04` after M00 L2 math |
| 4 | M06 | [exercises/](../foundations/module-06-transformers-attention-mechanisms/exercises/) | Attention implementation |

**First project:** [Build These #1 — Doc Q&A bot](../projects/build-these.md#1-doc-qa-bot-rag-starter)

---

### Software engineer, new to AI

| Step | Module | Exercise folder | Focus |
|------|--------|-----------------|-------|
| 1 | M01 | [exercises/](../foundations/module-01-ai-engineering-essentials/exercises/) | All files (`02`–`04`) + [notebook](../foundations/module-01-ai-engineering-essentials/exercises/04-prompt-engineering.ipynb) |
| 2 | — | [Build These #1](../projects/build-these.md#1-doc-qa-bot-rag-starter) | RAG before deep math |
| 3 | M06 | [exercises/](../foundations/module-06-transformers-attention-mechanisms/exercises/) | Optional — when you want internals |
| 4 | M09 | *Coming soon* | Chunking, retrieval — use M09 lesson code until notebooks ship |

**Skip initially:** M05 unless you want training-from-scratch depth.

---

### ML engineer → LLM / agents

| Step | Module | Exercise folder | Focus |
|------|--------|-----------------|-------|
| 1 | M01 | [exercises/](../foundations/module-01-ai-engineering-essentials/exercises/) | Skim `03`–`04` for API/prompt patterns |
| 2 | M06 | [exercises/](../foundations/module-06-transformers-attention-mechanisms/exercises/) | `07`–`09` attention from scratch |
| 3 | M09 | *Coming soon* | RAG pipeline — follow [M09 L5](../build/module-09-rag-retrieval-augmented-generation/lessons/05-Building-a-Basic-RAG-System.md) |
| 4 | M11 | *Coming soon* | Agent loop — follow [M11 L7](../build/module-11-ai-agents-fundamentals/lessons/07-Building-an-Agent.md) |

**Priority projects:** [Build These #2, #5, #8](../projects/build-these.md)

---

### Career switcher (portfolio track)

| Week | Exercises + projects |
|------|---------------------|
| 1 | M01 `02`–`04` |
| 2 | [Build These #1](../projects/build-these.md#1-doc-qa-bot-rag-starter) |
| 3 | M06 `01`–`03` (optional) + [Build These #4](../projects/build-these.md#4-tool-using-research-agent) |
| 4 | [Build These #9](../projects/build-these.md#9-ai-quality-eval-suite) |

Full roadmap: [Start Here — 4-week plan](../start-here.md#your-first-4-weeks-career-switcher-roadmap)

---

## How to run

```bash
# Example: M01 prompt engineering
cd docs/foundations/module-01-ai-engineering-essentials/exercises
pip install openai anthropic python-dotenv  # if needed
python 04-starter.py
```

Compare with `*-solution.py` when stuck. API keys only needed for exercises that call live LLM APIs.

**Jupyter:** [M01 prompt notebook](../foundations/module-01-ai-engineering-essentials/exercises/04-prompt-engineering.ipynb)

---

## Exercise index by module

| Module | Location | Files | Status |
|--------|----------|-------|--------|
| **M01** · AI Engineering Essentials | [exercises/](../foundations/module-01-ai-engineering-essentials/exercises/) | `02`–`04` starter/solution + notebook | ✅ Live |
| **M05** · Neural Networks | [exercises/](../foundations/module-05-neural-networks-deep-learning-fundamentals/exercises/) | `01`–`10` starter/solution | ✅ Live |
| **M06** · Transformers | [exercises/](../foundations/module-06-transformers-attention-mechanisms/exercises/) | `01`–`09` starter/solution | ✅ Live |
| **M09** · RAG | `build/module-09-.../exercises/` | — | 🔜 [Roadmap](../roadmap.md) |
| **M11** · Agents | `build/module-11-.../exercises/` | — | 🔜 [Roadmap](../roadmap.md) |

!!! note "M09 / M11 exercises"
    Listed in module indexes but not yet published as starter files. Use inline lesson code and [Build These](../projects/build-these.md) projects until notebooks land.

---

## After exercises: projects & capstones

| Level | Resource |
|-------|----------|
| **Portfolio projects** | [Build These First](../projects/build-these.md) — 10 ideas mapped to modules |
| **Full capstone briefs** | [M17 Capstones](../advanced/module-17-capstone-projects/index.md) — acceptance criteria, production scope |
| **Quick overview** | [Projects index](../projects/index.md) |

---

## Stuck?

| Issue | Help |
|-------|------|
| Import / env errors | [FAQ — Troubleshooting](../faq.md#troubleshooting) |
| Don't know which exercise next | [Start Here](../start-here.md) persona table |
| Exercise too hard | Read module lesson first; check `*-solution.py` |
