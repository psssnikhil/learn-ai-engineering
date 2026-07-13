---
title: Build These First
description: Ten portfolio-ready project ideas mapped to handbook modules
---

# Build These First

Ten projects ordered by difficulty. Each maps to specific modules so you know **what to learn before you build**.

!!! tip "Portfolio rule of three"
    Ship **one RAG app**, **one agent**, and **one production-ready demo with evals** — that's stronger than finishing every lesson without building.

Related: [Course 16 · Capstones](../advanced/module-17-capstone-projects/index.md) (full briefs) · [Exercises](../exercises/index.md) (skill drills)

---

## At a glance

| # | Project | Difficulty | Modules | Est. time |
|---|---------|------------|---------|-----------|
| 1 | [Doc Q&A bot (RAG starter)](#1-doc-qa-bot-rag-starter) | Beginner | Course 02, Course 06 | 1 weekend |
| 2 | [Enterprise RAG with citations](#2-enterprise-rag-with-citations) | Intermediate | Course 06, Course 10, Course 12 | 1–2 weeks |
| 3 | [Semantic search over code/docs](#3-semantic-search-engine) | Intermediate | Course 06, Course 10 | 1 week |
| 4 | [Tool-using research agent](#4-tool-using-research-agent) | Intermediate | Course 07, Course 08 | 1–2 weeks |
| 5 | [Multi-agent research system](#5-multi-agent-research-system) | Advanced | Course 09, Course 06, Course 13 | 2–3 weeks |
| 6 | [Support bot with routing](#6-support-bot-with-routing) | Intermediate | Course 06, Course 07, Course 09 | 2 weeks |
| 7 | [LLM data extraction pipeline](#7-llm-data-extraction-pipeline) | Intermediate | Course 11, Course 12 | 1 week |
| 8 | [Domain style fine-tune](#8-domain-style-fine-tune) | Advanced | Course 15, Course 05 | 2–3 weeks |
| 9 | [AI quality eval suite](#9-ai-quality-eval-suite) | Intermediate | Course 13, Course 14 | 1 week |
| 10 | [Deploy your AI app](#10-deploy-your-ai-app) | Advanced | Course 12, Course 16 | 1–2 weeks |

---

## 1. Doc Q&A bot (RAG starter)

**What:** Upload PDFs, ask questions, get answers grounded in your docs.

**Learn first:**

| Module | Lessons |
|--------|---------|
| [Course 02](../foundations/module-01-ai-engineering-essentials/index.md) | L2 First app, L5 APIs |
| [Course 06](../build/module-09-rag-retrieval-augmented-generation/index.md) | L1–5 (intro → basic RAG) |

**Build checklist:**

- [ ] Chunk documents (500–1000 tokens)
- [ ] Embed + store in Chroma or FAISS
- [ ] Retrieve top-k, inject into prompt
- [ ] Show source snippets in UI

**Stretch:** Add Streamlit or FastAPI frontend.

**Capstone link:** [Capstone project 1 — RAG Knowledge Assistant](../advanced/module-17-capstone-projects/lessons/02-lesson-02.md)

---

## 2. Enterprise RAG with citations

**What:** Production-style RAG with hybrid search, citation links, and basic monitoring.

**Learn first:**

| Module | Lessons |
|--------|---------|
| [Course 06](../build/module-09-rag-retrieval-augmented-generation/index.md) | L6–10 (advanced RAG, hybrid, eval, production) |
| [Course 10](../build/module-13-vector-databases-deep-dive/index.md) | L1–5 (indexing, schema) |
| [Course 12](../production/module-10-llmops-production-systems/index.md) | L2 Observability |

**Build checklist:**

- [ ] Hybrid BM25 + vector search
- [ ] Reranker or score threshold
- [ ] Inline citations `[doc_id:chunk]`
- [ ] Log retrieval scores + latency

**Capstone link:** Capstone project 1 (extended)

---

## 3. Semantic search engine

**What:** Search interface over a corpus (docs, blog, codebase) with filters and highlighting.

**Learn first:**

| Module | Lessons |
|--------|---------|
| [Course 06](../build/module-09-rag-retrieval-augmented-generation/index.md) | L2 Vector DBs, L4 Retrieval |
| [Course 10](../build/module-13-vector-databases-deep-dive/index.md) | L6–8 (schema, scaling, hybrid) |

**Build checklist:**

- [ ] Metadata filters (date, tag, author)
- [ ] Highlight matched spans
- [ ] Evaluate recall@k on 20 hand-labeled queries

**Capstone link:** [Capstone project 6 — Semantic Search Engine](../advanced/module-17-capstone-projects/lessons/06-lesson-06.md)

---

## 4. Tool-using research agent

**What:** Agent that searches the web (or files), summarizes, and cites sources.

**Learn first:**

| Module | Lessons |
|--------|---------|
| [Course 07](../build/module-11-ai-agents-fundamentals/index.md) | L1–4, L7 (loop, ReAct, tools) |
| [Course 08](../build/module-18-agent-harness-tools-runtime/index.md) | L1–3 (harness, loop, tools) |

**Build checklist:**

- [ ] ReAct loop with max 10 steps
- [ ] 2–3 tools (search, read_file, summarize)
- [ ] Structured final report (markdown)
- [ ] Trace log of each step

**Capstone link:** [Capstone project 2 — Autonomous Coding Agent](../advanced/module-17-capstone-projects/lessons/03-lesson-03.md) (adapt tools for research)

---

## 5. Multi-agent research system

**What:** Orchestrator delegates to specialist agents (searcher, analyst, writer).

**Learn first:**

| Module | Lessons |
|--------|---------|
| [Course 09](../build/module-12-multi-agent-systems/index.md) | L1–6 (orchestrator, handoffs) |
| [Course 06](../build/module-09-rag-retrieval-augmented-generation/index.md) | L9 Agentic RAG |
| [Course 13](../production/module-19-llm-evaluation-quality/index.md) | L4 Agent trajectory evals |

**Build checklist:**

- [ ] Orchestrator + 2 worker agents
- [ ] Shared state or blackboard
- [ ] Eval: did final report cite all sources?

**Capstone link:** [Capstone project 3 — Multi-Agent Research](../advanced/module-17-capstone-projects/lessons/04-lesson-04.md)

---

## 6. Support bot with routing

**What:** Customer support bot that routes billing vs technical questions to different knowledge bases.

**Learn first:**

| Module | Lessons |
|--------|---------|
| [Course 06](../build/module-09-rag-retrieval-augmented-generation/index.md) | L1–5 |
| [Course 07](../build/module-11-ai-agents-fundamentals/index.md) | L10 Workflow vs Agent |
| [Course 09](../build/module-12-multi-agent-systems/index.md) | L5 Supervisor/Router |

**Build checklist:**

- [ ] Router classifies intent
- [ ] Separate RAG indexes per domain
- [ ] Escalation to human when confidence low
- [ ] Guardrails ([Course 14 L4](../production/module-16-ai-safety-ethics/lessons/04-lesson-04.md))

---

## 7. LLM data extraction pipeline

**What:** Extract structured JSON from invoices, emails, or PDFs at scale.

**Learn first:**

| Module | Lessons |
|--------|---------|
| [Course 11](../build/module-14-prompt-engineering-mastery/index.md) | L4 Structured output, L6 Chaining |
| [Course 12](../production/module-10-llmops-production-systems/index.md) | L4 Caching, L8 API design |

**Build checklist:**

- [ ] JSON schema validation
- [ ] Batch processing with retries
- [ ] Golden set of 50 examples + accuracy metric

**Capstone link:** [Capstone project 7 — Data Extraction Pipeline](../advanced/module-17-capstone-projects/lessons/07-lesson-07.md)

---

## 8. Domain style fine-tune

**What:** Fine-tune (LoRA) a small model to match your team's writing style or report format.

**Learn first:**

| Module | Lessons |
|--------|---------|
| [Course 05](../foundations/module-07-large-language-models-llms/index.md) | L6–7 Fine-tuning, instruction tuning |
| [Course 15](../advanced/module-15-fine-tuning-custom-models/index.md) | L1–5 (when, data prep, LoRA) |

**Build checklist:**

- [ ] 200+ high-quality (input, output) pairs
- [ ] LoRA fine-tune on 7B or smaller
- [ ] Compare base vs fine-tuned on held-out set
- [ ] Document when RAG would have been enough ([FAQ](../faq.md#rag-vs-fine-tuning-vs-agents))

---

## 9. AI quality eval suite

**What:** Automated eval pipeline for an LLM app — golden dataset, LLM-as-judge, CI gate.

**Learn first:**

| Module | Lessons |
|--------|---------|
| [Course 13](../production/module-19-llm-evaluation-quality/index.md) | All 6 lessons |
| [Course 14](../production/module-16-ai-safety-ethics/index.md) | L3 Hallucination, L8 Red teaming |

**Build checklist:**

- [ ] 30+ golden Q&A pairs
- [ ] Regression test in CI
- [ ] Dashboard or report for pass/fail trends

**Capstone link:** [Capstone project 9 — AI Safety Evaluation Suite](../advanced/module-17-capstone-projects/lessons/09-lesson-09.md)

---

## 10. Deploy your AI app

**What:** Take project #1, #2, or #4 to production — Docker, env config, monitoring, cost caps.

**Learn first:**

| Module | Lessons |
|--------|---------|
| [Course 12](../production/module-10-llmops-production-systems/index.md) | L6–10 (cost, deployment, scaling) |
| [Course 13](../production/module-19-llm-evaluation-quality/index.md) | L5–6 (CI/CD, monitoring) |

**Build checklist:**

- [ ] Dockerfile + health check
- [ ] Secrets via env vars
- [ ] Rate limiting + cost alerts
- [ ] README with architecture diagram

**Capstone link:** [Capstone project 10 — Deploy](../advanced/module-17-capstone-projects/lessons/10-lesson-10.md)

---

## Suggested build order by persona

| Persona | Build order |
|---------|-------------|
| **Complete beginner** | 1 → 4 → 9 |
| **Software engineer** | 1 → 2 → 4 → 10 |
| **ML engineer** | 2 → 5 → 8 |
| **Career switcher** | 1 → 4 → 6 → 9 → 10 |

Start routing: [Start Here](../start-here.md)
