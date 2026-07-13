---
title: Exercises
description: Hands-on paths by persona — starter and solution files per course
---

# Python Exercises

Hands-on starter and solution files live under each course's `exercises/` folder. Use this page to find **the right path for your background**, not just a file list.

!!! tip "New here?"
    Pick your persona in [Start Here](../start-here.md), then return here for practice files.

---

## Exercise paths by persona

### Complete beginner (no CS background)

| Step | Course | Exercise folder | Focus |
|------|--------|-----------------|-------|
| 1 | 01 | — (in-lesson code) | NumPy, vectors — [Prerequisites](../foundations/module-00-genai-foundations-from-nlp-to-transformers/lessons/01-prerequisites.md) |
| 2 | 02 | [exercises/](../foundations/module-01-ai-engineering-essentials/exercises/) | First app, tokens, prompts |
| 3 | 03 | [exercises/](../foundations/module-05-neural-networks-deep-learning-fundamentals/exercises/) | Neural net basics after course 01 math |
| 4 | 04 | [exercises/](../foundations/module-06-transformers-attention-mechanisms/exercises/) | Attention implementation |

**First project:** [Build These #1 — Doc Q&A bot](../projects/build-these.md#1-doc-qa-bot-rag-starter)

---

### Software engineer, new to AI

| Step | Course | Exercise folder | Focus |
|------|--------|-----------------|-------|
| 1 | 02 | [exercises/](../foundations/module-01-ai-engineering-essentials/exercises/) | All files + [notebook](../foundations/module-01-ai-engineering-essentials/exercises/04-prompt-engineering.ipynb) |
| 2 | — | [Build These #1](../projects/build-these.md#1-doc-qa-bot-rag-starter) | RAG before deep math |
| 3 | 04 | [exercises/](../foundations/module-06-transformers-attention-mechanisms/exercises/) | Optional — when you want internals |
| 4 | 06 | *Coming soon* | Chunking, retrieval — use course 06 lesson code until notebooks ship |

**Skip initially:** Course 03 unless you want training-from-scratch depth.

---

### ML engineer → LLM / agents

| Step | Course | Exercise folder | Focus |
|------|--------|-----------------|-------|
| 1 | 02 | [exercises/](../foundations/module-01-ai-engineering-essentials/exercises/) | Skim API/prompt patterns |
| 2 | 04 | [exercises/](../foundations/module-06-transformers-attention-mechanisms/exercises/) | Attention from scratch |
| 3 | 06 | *Coming soon* | RAG pipeline — follow [Course 06 · Build RAG](../build/module-09-rag-retrieval-augmented-generation/lessons/05-Building-a-Basic-RAG-System.md) |
| 4 | 07 | *Coming soon* | Agent loop — follow [Course 07 · Building an Agent](../build/module-11-ai-agents-fundamentals/lessons/07-Building-an-Agent.md) |

**Priority projects:** [Build These #2, #5, #8](../projects/build-these.md)

---

### Career switcher (portfolio track)

| Week | Exercises + projects |
|------|---------------------|
| 1 | Course 02 exercises |
| 2 | [Build These #1](../projects/build-these.md#1-doc-qa-bot-rag-starter) |
| 3 | Course 04 exercises (optional) + [Build These #4](../projects/build-these.md#4-tool-using-research-agent) |
| 4 | [Build These #9](../projects/build-these.md#9-ai-quality-eval-suite) |

Full plan: [Start Here — 4-week roadmap](../start-here.md#your-first-4-weeks-career-switcher-roadmap)

---

## How to run

```bash
# Example: Course 02 prompt engineering
cd docs/foundations/module-01-ai-engineering-essentials/exercises
pip install openai anthropic python-dotenv  # if needed
python 04-starter.py
```

Compare with `*-solution.py` when stuck. API keys only needed for exercises that call live LLM APIs.

**Jupyter:** [Course 02 prompt notebook](../foundations/module-01-ai-engineering-essentials/exercises/04-prompt-engineering.ipynb)

---

## Exercise index by course

| Course | Location | Files | Status |
|--------|----------|-------|--------|
| **02** · AI Engineering Essentials | [exercises/](../foundations/module-01-ai-engineering-essentials/exercises/) | `02`–`04` starter/solution + notebook | ✅ Live |
| **03** · Neural Networks | [exercises/](../foundations/module-05-neural-networks-deep-learning-fundamentals/exercises/) | `01`–`10` starter/solution | ✅ Live |
| **04** · Transformers | [exercises/](../foundations/module-06-transformers-attention-mechanisms/exercises/) | `01`–`09` starter/solution | ✅ Live |
| **06** · RAG | `build/module-09-.../exercises/` | — | 🔜 [Roadmap](../roadmap.md) |
| **07** · Agents | `build/module-11-.../exercises/` | — | 🔜 [Roadmap](../roadmap.md) |

!!! note "Courses 06 & 07 exercises"
    Listed in course indexes but not yet published as starter files. Use inline lesson code and [Build These](../projects/build-these.md) projects until notebooks land.

---

## After exercises: projects & capstones

| Level | Resource |
|-------|----------|
| **Portfolio projects** | [Build These First](../projects/build-these.md) — 10 ideas mapped to courses |
| **Full capstone briefs** | [Course 16 · Capstones](../advanced/module-17-capstone-projects/index.md) |
| **Quick overview** | [Projects index](../projects/index.md) |

---

## Stuck?

| Issue | Help |
|-------|------|
| Import / env errors | [FAQ — Troubleshooting](../faq.md#troubleshooting) |
| Don't know which exercise next | [Start Here](../start-here.md) persona table |
| Exercise too hard | Read the course lesson first; check `*-solution.py` |
