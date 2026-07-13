---
title: How to Learn
description: Week-by-week study plans for beginners, intermediates, and agent engineers
---

# How to Learn

This handbook is **140+ lessons** across four phases. This guide gives you **week-by-week plans** by persona — not a replacement for the [Learning Path](../learning-path.md), but a concrete schedule with hours, checkpoints, and hands-on milestones.

## Choose your path

| Persona | Background | Time/week | Duration | Jump to |
|---------|------------|-----------|----------|---------|
| **Beginner** | Software engineer, little ML | 8–10 hrs | ~20 weeks | [Beginner plan ↓](#beginner-path-20-weeks) |
| **Intermediate** | Know ML/Python, new to LLMs & agents | 6–8 hrs | ~12 weeks | [Intermediate plan ↓](#intermediate-path-12-weeks) |
| **Agent engineer** | Shipping or designing agent systems | 5–7 hrs | ~8 weeks | [Agent engineer plan ↓](#agent-engineer-path-8-weeks) |

---

## Beginner path (~20 weeks)

**Goal:** Understand transformers and LLMs, build a RAG app, touch agents, learn production basics.

**Assumptions:** Comfortable with Python and git. No prior neural-network coursework required.

### Phase A — Foundations (Weeks 1–10)

| Week | Focus | Lessons | Hours | Checkpoint |
|------|-------|---------|-------|------------|
| **1** | NLP + math intuition | [M00 L1–L3](../foundations/module-00-genai-foundations-from-nlp-to-transformers/index.md) | 8 | Explain tokens vs words |
| **2** | Attention + transformers intro | [M00 L4–L7](../foundations/module-00-genai-foundations-from-nlp-to-transformers/index.md) | 10 | Sketch encoder block |
| **3** | First API app + tokens/costs | [M01](../foundations/module-01-ai-engineering-essentials/index.md) (all) | 8 | Run a chat app locally |
| **4** | Neurons, loss, gradient descent | [M05 L1–L4](../foundations/module-05-neural-networks-deep-learning-fundamentals/index.md) | 10 | Manual grad on toy net |
| **5** | Backprop + regularization | [M05 L5–L6](../foundations/module-05-neural-networks-deep-learning-fundamentals/index.md) | 8 | Train MNIST classifier |
| **6** | Build NN from scratch | [M05 L7](../foundations/module-05-neural-networks-deep-learning-fundamentals/index.md) | 10 | Forward pass in NumPy |
| **7** | Self-attention deep dive | [M06 L1–L5](../foundations/module-06-transformers-attention-mechanisms/index.md) | 10 | Compute attention on 4 tokens |
| **8** | Transformer training + variants | [M06 L6–L10](../foundations/module-06-transformers-attention-mechanisms/index.md) | 8 | Read scaling laws lesson |
| **9** | LLM lifecycle | [M07 L1–L6](../foundations/module-07-large-language-models-llms/index.md) | 10 | Tokenize a sentence; compare BPE |
| **10** | Instruction tuning + reasoning | [M07 L7–L11](../foundations/module-07-large-language-models-llms/index.md) | 8 | **Milestone:** explain pretrain → SFT → RLHF |

**Optional deep dives (any week):** [Attention math](../deep-dives/attention-math.md), [Tokenization internals](../deep-dives/tokenization-internals.md)

### Phase B — Build (Weeks 11–16)

| Week | Focus | Lessons | Hours | Checkpoint |
|------|-------|---------|-------|------------|
| **11** | RAG fundamentals | [M09 L1–L5](../build/module-09-rag-retrieval-augmented-generation/index.md) | 8 | Chunk docs; embed; query |
| **12** | Advanced RAG + eval | [M09 L6–L11](../build/module-09-rag-retrieval-augmented-generation/index.md) | 10 | Measure retrieval precision |
| **13** | Prompt engineering | [M14 L1–L5](../build/module-14-prompt-engineering-mastery/index.md) | 8 | A/B two system prompts |
| **14** | Agent fundamentals | [M11 L1–L5](../build/module-11-ai-agents-fundamentals/index.md) | 10 | Implement ReAct loop |
| **15** | Agents + memory | [M11 L6–L10](../build/module-11-ai-agents-fundamentals/index.md) | 8 | Agent with vector recall |
| **16** | Vector DB deep dive | [M13 L1–L5](../build/module-13-vector-databases-deep-dive/index.md) | 8 | **Milestone:** ship local RAG Q&A |

### Phase C — Production touch + capstone prep (Weeks 17–20)

| Week | Focus | Lessons | Hours | Checkpoint |
|------|-------|---------|-------|------------|
| **17** | LLMOps intro | [M10 L1–L5](../production/module-10-llmops-production-systems/index.md) | 8 | Add logging to RAG app |
| **18** | Evals + monitoring | [M19](../production/module-19-llm-evaluation-quality/index.md) (all) | 10 | 10 golden Q&A cases in CI |
| **19** | Safety basics | [M16 L1–L5](../production/module-16-ai-safety-ethics/index.md) | 6 | Red-team one prompt |
| **20** | Capstone planning | [M17 L1–L2](../advanced/module-17-capstone-projects/index.md) | 8 | **Milestone:** project proposal doc |

---

## Intermediate path (~12 weeks)

**Goal:** Skip classical ML depth where you already know it; focus on LLMs, RAG, agents, production.

**Assumptions:** Prior ML course or work experience; comfortable training small models.

### Weeks 1–2 — LLM essentials

| Week | Focus | Go to |
|------|-------|-------|
| **1** | M01 + M07 L1–L6 | APIs, tokenization, embeddings, pretraining |
| **2** | M07 L7–L11 + [Deep Dive · Attention](../deep-dives/attention-math.md) | SFT, reasoning models |

*Skip M05 unless you want from-scratch NN practice. Skim M00/M06 as reference.*

### Weeks 3–5 — Build systems

| Week | Focus | Go to |
|------|-------|-------|
| **3** | M09 full module | RAG pipeline end-to-end |
| **4** | M14 + M13 | Prompts + vector DB tuning |
| **5** | M11 + [Agent Engineering track](../agent-engineering/index.md) L1–L4 | ReAct, memory, tools, harness |

### Weeks 6–8 — Agents in depth

| Week | Focus | Go to |
|------|-------|-------|
| **6** | M18 full module | Harness, MCP, permissions |
| **7** | M12 | Multi-agent orchestration |
| **8** | [Agent Engineering](../agent-engineering/index.md) L5–L7 + [2026 Skills](../ai-engineering-2026/index.md) | Orchestration, tracing, evals, context engineering |

### Weeks 9–12 — Production + advanced

| Week | Focus | Go to |
|------|-------|-------|
| **9** | M10 | LLMOps, caching, deployment |
| **10** | M19 + [Evals hub](../evals-observability/index.md) | Trajectory evals, CI gates |
| **11** | M16 L1–L5 | Safety and privacy |
| **12** | M15 L1–L5 or M17 capstone | Fine-tuning intro **or** capstone kickoff |

**Milestone (week 8):** One agent with harness, traces, and 5 golden trajectories in CI.

---

## Agent engineer path (~8 weeks)

**Goal:** Production-grade agents — harness, orchestration, observability, evals — fast.

**Assumptions:** You already call LLM APIs and have shipped at least one prototype agent.

### Week 1 — Loop + harness

| Day | Topic | Resource |
|-----|-------|----------|
| 1–2 | Agent loop, ReAct | [Agent Loop](../agent-engineering/01-agent-loop.md), [M11 L3](../build/module-11-ai-agents-fundamentals/lessons/03-ReAct-Pattern.md) |
| 3–4 | Harness primitives | [Harness Engineering](../agent-engineering/04-harness-engineering.md), [M18](../build/module-18-agent-harness-tools-runtime/index.md) |
| 5 | Hands-on | Add `max_steps`, cost cap, checkpoint to your agent |

### Week 2 — Tools, MCP, memory

| Day | Topic | Resource |
|-----|-------|----------|
| 1–2 | Tool design | [Tools & MCP](../agent-engineering/03-tools-and-mcp.md) |
| 3 | MCP server | [M18 L4](../build/module-18-agent-harness-tools-runtime/lessons/04-mcp-model-context-protocol.md) |
| 4–5 | Memory + context | [Memory](../agent-engineering/02-memory.md), [Context Engineering](../ai-engineering-2026/context-engineering.md) |

### Week 3 — Orchestration

| Day | Topic | Resource |
|-----|-------|----------|
| 1–3 | Patterns | [Orchestration](../agent-engineering/05-orchestration.md), [M12](../build/module-12-multi-agent-systems/index.md) |
| 4–5 | Hands-on | Supervisor + 2 workers; structured handoff packets |

### Week 4 — Observability

| Day | Topic | Resource |
|-----|-------|----------|
| 1–2 | Tracing | [Observability](../agent-engineering/06-observability-and-tracing.md) |
| 3–5 | Hands-on | Langfuse or OTel; dashboard: cost per success, tool errors |

### Week 5 — Evals

| Day | Topic | Resource |
|-----|-------|----------|
| 1–3 | Trajectory evals | [Agent Evals](../agent-engineering/07-agent-evals.md), [M19 L4–L5](../production/module-19-llm-evaluation-quality/index.md) |
| 4–5 | Hands-on | 10 golden trajectories; CI gate on PR |

### Week 6 — 2026 tooling

| Day | Topic | Resource |
|-----|-------|----------|
| 1–2 | Claude Code + skills | [Claude Code](../ai-engineering-2026/claude-code.md), [Skills & Rules](../ai-engineering-2026/skills-and-rules.md) |
| 3–4 | Loop engineering | [Loop Engineering](../ai-engineering-2026/loop-engineering.md) |
| 5 | Hands-on | `CLAUDE.md` + one SKILL.md; outer eval loop on skill change |

### Week 7 — Production LLMOps

| Day | Topic | Resource |
|-----|-------|----------|
| 1–3 | Deploy, cache, cost | [M10](../production/module-10-llmops-production-systems/index.md) |
| 4–5 | Permissions + safety | [M18 L5](../build/module-18-agent-harness-tools-runtime/lessons/05-permissions-and-safety-in-the-harness.md), [M16 L1–L3](../production/module-16-ai-safety-ethics/index.md) |

### Week 8 — Integration milestone

| Deliverable | Criteria |
|-------------|------------|
| **Harness** | max_steps, cost cap, permissions, truncation |
| **Traces** | Full trajectory in Langfuse/OTel |
| **Evals** | ≥10 golden cases; CI blocks regressions |
| **Docs** | Runbook: debug trace_id, kill switch, on-call |
| **Optional** | Multi-agent flow with handoff packet schema |

---

## Study habits that work

### Active learning (not passive reading)

1. **One lesson → one artifact** — notebook, script, or eval case per lesson
2. **Teach it** — explain attention or ReAct aloud in 5 minutes without slides
3. **Spaced review** — revisit M06/M07 summaries before starting M11
4. **Use the glossary** — [Glossary](../glossary.md) when terms blur together

### When you're stuck

| Symptom | Action |
|---------|--------|
| Math lost | [Deep Dives](../deep-dives/index.md) + 3Blue1Brown / Jalammar links in lessons |
| Code won't run | Check lesson `has_code` notebooks; file issue on repo |
| Agent loops forever | [Harness termination](../agent-engineering/04-harness-engineering.md) |
| RAG answers wrong | [M09 eval lesson](../build/module-09-rag-retrieval-augmented-generation/lessons/08-RAG-Evaluation-Metrics.md) before more chunks |

### Weekly rhythm (template)

```
Mon–Wed: 2 lessons (read + code)
Thu:     Hands-on project block (2–3 hrs)
Fri:     1 lesson + write 5 bullet summary
Weekend: Optional capstone / OSS exploration ([Resources](../resources/index.md))
```

---

## Cross-links

| Need | Page |
|------|------|
| Full module list | [Learning Path](../learning-path.md) |
| Topic lookup | [Topic Map](../topic-map.md) |
| Agent-focused hub | [Agentic AI](../agentic-ai/index.md) |
| Evals hub | [Evals & Observability](../evals-observability/index.md) |
| Papers | [Agent Engineering · Related papers](../agent-engineering/related-papers.md) |
| Contributing | [CONTRIBUTING.md](https://github.com/psssnikhil/learn-ai-engineering/blob/main/CONTRIBUTING.md) |

**Start:** Pick your persona table above and block Week 1 on your calendar today.

!!! info "Nav note for site maintainers"
    Add `learn/index.md` to `mkdocs.yml` under a **How to Learn** or **Getting Started** nav entry when updating site navigation.
