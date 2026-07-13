---
title: Study Plans
description: Week-by-week study plans for beginners, intermediates, and agent engineers
---

# Study Plans

Concrete **week-by-week schedules** by persona. For the full ordered list of courses, see **[Learn](index.md)**.

## Choose your path

| Persona | Background | Time/week | Duration | Jump to |
|---------|------------|-----------|----------|---------|
| **Beginner** | Software engineer, little ML | 8–10 hrs | ~20 weeks | [Beginner plan ↓](#beginner-path-20-weeks) |
| **Intermediate** | Know ML/Python, new to LLMs & agents | 6–8 hrs | ~12 weeks | [Intermediate plan ↓](#intermediate-path-12-weeks) |
| **Agent engineer** | Shipping or designing agent systems | 5–7 hrs | ~8 weeks | [Agent engineer plan ↓](#agent-engineer-path-8-weeks) |

---

## Beginner path (~20 weeks)

**Goal:** Understand transformers and LLMs, build a RAG app, touch agents, learn production basics.

| Week | Focus | Courses | Milestone |
|------|-------|---------|-----------|
| 1–2 | NLP, attention, transformers intro | [01 GenAI Foundations](../foundations/module-00-genai-foundations-from-nlp-to-transformers/index.md) | Sketch a transformer block |
| 3 | First API app | [02 AI Essentials](../foundations/module-01-ai-engineering-essentials/index.md) | Working chat script |
| 4–6 | Neural nets from scratch | [03 Neural Networks](../foundations/module-05-neural-networks-deep-learning-fundamentals/index.md) | Train a small classifier |
| 7–8 | Attention deep dive | [04 Transformers](../foundations/module-06-transformers-attention-mechanisms/index.md) | Compute attention on toy tokens |
| 9–10 | LLM lifecycle | [05 LLMs](../foundations/module-07-large-language-models-llms/index.md) | Explain pretrain → SFT → RLHF |
| 11–12 | RAG | [06 RAG](../build/module-09-rag-retrieval-augmented-generation/index.md) | Local doc Q&A |
| 13 | Prompts | [11 Prompt Engineering](../build/module-14-prompt-engineering-mastery/index.md) | A/B two system prompts |
| 14–15 | Agents | [07 AI Agents](../build/module-11-ai-agents-fundamentals/index.md) | ReAct loop with memory |
| 16 | Vector DBs | [10 Vector Databases](../build/module-13-vector-databases-deep-dive/index.md) | Tune retrieval on your corpus |
| 17 | LLMOps | [12 LLMOps](../production/module-10-llmops-production-systems/index.md) | Logging + caching on RAG app |
| 18 | Evals | [13 LLM Evaluation](../production/module-19-llm-evaluation-quality/index.md) | 10 golden cases in CI |
| 19 | Safety | [14 AI Safety](../production/module-16-ai-safety-ethics/index.md) | Red-team one prompt |
| 20 | Capstone prep | [16 Capstones](../advanced/module-17-capstone-projects/index.md) | Project proposal doc |

**Optional any week:** [Deep Dives](../deep-dives/index.md)

---

## Intermediate path (~12 weeks)

Skip classical ML depth if you already know it; focus on LLMs, RAG, agents, production.

| Weeks | Focus | Courses |
|-------|-------|---------|
| 1–2 | LLM essentials | 02, 05 (+ [Attention deep dive](../deep-dives/attention-math.md)) |
| 3–5 | Build systems | 06 RAG, 11 Prompts, 10 Vector DBs, 07 Agents + [Agent Engineering](../agent-engineering/index.md) |
| 6–8 | Agents in depth | 08 Harness, 09 Multi-Agent, Agent Engineering + [Modern AI 2026](../ai-engineering-2026/index.md) |
| 9–12 | Production + advanced | 12 LLMOps, 13 Evals, 14 Safety, 15 Fine-Tuning or 16 Capstones |

**Milestone (week 8):** One agent with harness, traces, and 5 golden trajectories in CI.

---

## Agent engineer path (~8 weeks)

**Goal:** Production-grade agents — harness, orchestration, observability, evals — fast.

| Week | Focus | Resources |
|------|-------|-----------|
| 1 | Loop + harness | [Agent Loop](../agent-engineering/01-agent-loop.md), [07 Agents](../build/module-11-ai-agents-fundamentals/index.md), [08 Harness](../build/module-18-agent-harness-tools-runtime/index.md) |
| 2 | Tools, MCP, memory | [Tools & MCP](../agent-engineering/03-tools-and-mcp.md), [Memory](../agent-engineering/02-memory.md), [Context Engineering](../ai-engineering-2026/context-engineering.md) |
| 3 | Orchestration | [Orchestration](../agent-engineering/05-orchestration.md), [09 Multi-Agent](../build/module-12-multi-agent-systems/index.md) |
| 4 | Observability | [Observability](../agent-engineering/06-observability-and-tracing.md) |
| 5 | Evals | [Agent Evals](../agent-engineering/07-agent-evals.md), [13 LLM Evaluation](../production/module-19-llm-evaluation-quality/index.md) |
| 6 | 2026 tooling | [Modern AI 2026](../ai-engineering-2026/index.md) |
| 7 | Production | [12 LLMOps](../production/module-10-llmops-production-systems/index.md), safety lessons in 08 + 14 |
| 8 | Integration | Harness + traces + evals + runbook (see [Build These](../projects/build-these.md)) |

---

## Study habits

1. **One lesson → one artifact** — notebook, script, or eval case per lesson
2. **Teach it** — explain attention or ReAct aloud in 5 minutes
3. **Use the glossary** — [Glossary](../glossary.md) when terms blur together

| Stuck on… | Go to |
|-----------|-------|
| Math | [Deep Dives](../deep-dives/index.md) |
| Agent loops forever | [Harness termination](../agent-engineering/04-harness-engineering.md) |
| Bad RAG answers | [06 RAG · Evaluation](../build/module-09-rag-retrieval-augmented-generation/lessons/08-RAG-Evaluation-Metrics.md) |

**Start:** Pick a persona above and block Week 1 on your calendar today.
