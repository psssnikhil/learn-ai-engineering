---
title: FAQ
description: Common questions, decision tables, and troubleshooting
---

# Frequently Asked Questions

Answers for learners at every level. Can't find yours? [Open an issue](https://github.com/psssnikhil/learn-ai-engineering/issues).

---

## Getting started

### Do I need a CS degree or ML background?

No. The handbook targets **software engineers and career switchers** who can write Python. [Course 01 · Prerequisites](foundations/module-00-genai-foundations-from-nlp-to-transformers/lessons/01-prerequisites.md) covers the math you need. If you've never programmed, learn Python basics elsewhere first, then return to [Start Here](start-here.md).

### I'm a complete beginner. Where do I start?

1. [Prerequisites](foundations/module-00-genai-foundations-from-nlp-to-transformers/lessons/01-prerequisites.md) — Python + NumPy check
2. [Course 01 · GenAI Foundations](foundations/module-00-genai-foundations-from-nlp-to-transformers/index.md)
3. [Course 02 · AI Essentials](foundations/module-01-ai-engineering-essentials/index.md) — first API call
4. [Build These #1](projects/build-these.md#1-doc-qa-bot-rag-starter) — first portfolio piece

Skip courses 03–04 initially if you want to **build first, math later**.

### I'm a software engineer. Can I skip foundations?

Yes — start at [Course 02](foundations/module-01-ai-engineering-essentials/index.md). Skim [Course 05](foundations/module-07-large-language-models-llms/index.md) for transformer context, then jump to [Course 06 RAG](build/module-09-rag-retrieval-augmented-generation/index.md) or [Course 07 Agents](build/module-11-ai-agents-fundamentals/index.md). See [Start Here routing tables](start-here.md#i-want-to-learn).

### I'm an ML engineer. What can I skip?

Skip courses 03–04 if you know backprop and attention. Start at [Course 05 LLMs](foundations/module-07-large-language-models-llms/index.md), then [Course 06 RAG](build/module-09-rag-retrieval-augmented-generation/index.md) / [Course 07 Agents](build/module-11-ai-agents-fundamentals/index.md). Prioritize **evals and production**: [Course 13 Evals](production/module-19-llm-evaluation-quality/index.md) and [Course 08 Harness](build/module-18-agent-harness-tools-runtime/index.md).

### How long does the full curriculum take?

| Pace | Duration |
|------|----------|
| Full-time | 8–12 weeks |
| Part-time (10–15 hr/week) | 4–6 months |
| Reference / jump-in | Use [Topic Map](topic-map.md) or [Study Plans](learn/study-plans.md) |

---

## Curriculum & navigation

### How do I navigate the site?

| Tab | Use when |
|-----|----------|
| **[Start Here](start-here.md)** | Pick persona, goal, prerequisites |
| **[Learn](learn/index.md)** | Follow courses 01–16 in order |
| **[Reference](topic-map.md)** | FAQ, glossary, topic lookup, resources |
| **[Projects](projects/build-these.md)** | Portfolio builds |

### What's the difference between Agent Engineering track and Course 07?

| Resource | Best for |
|----------|----------|
| [Agent Engineering track](agent-engineering/index.md) | Focused 7-lesson agent spine |
| [Course 07 · AI Agents](build/module-11-ai-agents-fundamentals/index.md) | Full module with exercises |
| [Course 08 · Harness](build/module-18-agent-harness-tools-runtime/index.md) | MCP, permissions, runtime |
| [Agentic AI hub](agentic-ai/index.md) | Stack overview + OSS links |

Pick one spine; use the others as reference.

### Are exercises required?

Strongly recommended. [Exercises index](exercises/index.md) · Portfolio work: [Build These First](projects/build-these.md).

### Where are capstone projects?

- Quick ideas: [Build These First](projects/build-these.md)
- Full briefs: [Course 16 · Capstones](advanced/module-17-capstone-projects/index.md)

---

## Technical decisions

### RAG vs fine-tuning vs agents

| Approach | Cost | Time to ship | Best for | Not for |
|----------|------|--------------|----------|---------|
| **Prompt engineering** | Lowest | Hours | Format, prototyping | Large private knowledge bases |
| **RAG** | Medium | Days | Q&A over your docs | Changing model personality |
| **Fine-tuning** | High | Days–weeks | Style, domain behavior | Injecting fresh facts |
| **Agents** | Medium–high | Days–weeks | Multi-step + tools | Simple single-shot Q&A |
| **Workflows** | Low–medium | Days | Deterministic pipelines | Open-ended exploration |

```
Can prompts solve it? → Yes: prompts
                      → No: Need your docs? → Yes: RAG (Course 06)
                                              → No: Consistent style? → Fine-tune (Course 15)
                                                                      → Multi-step + tools? → Agent (Course 07)
```

See [Course 15 · When to fine-tune](advanced/module-15-fine-tuning-custom-models/lessons/01-lesson-01.md) · [Course 07 · Workflow vs agent](build/module-11-ai-agents-fundamentals/lessons/10-Workflow-vs-Agent.md)

### RAG vs Graph RAG?

| Pattern | When |
|---------|------|
| **Vector RAG** | Most doc Q&A |
| **Hybrid search** | Keywords + semantics matter |
| **Graph RAG** | Entity relationships, "how does X connect to Y" |

See [Course 06 · Graph RAG lesson](build/module-09-rag-retrieval-augmented-generation/lessons/11-graph-rag-and-knowledge-graphs.md).

---

## Tools & setup

### What do I need installed?

- Python 3.10+
- `pip install -r requirements.txt`
- API keys only for lessons that call live models

See [Getting Started — Local setup](getting-started.md#local-setup).

### Do I need a GPU?

Not for most lessons. Cloud APIs cover inference. Course 15 (fine-tuning) may use Colab — alternatives noted in lessons.

---

## Troubleshooting

### `ModuleNotFoundError` when running exercises

```bash
cd docs/foundations/module-01-ai-engineering-essentials/exercises
pip install openai anthropic python-dotenv
python 04-prompt-engineering-starter.py
```

See [Exercises index](exercises/index.md).

### API key errors (`401`)

1. `export OPENAI_API_KEY=sk-...` (or `.env` with `python-dotenv`)
2. Confirm key is active in provider dashboard
3. Match env var name to SDK

### Rate limits (`429`)

- Exponential backoff · smaller dev models · batch size limits
- See [Course 02 · Tokens and costs](foundations/module-01-ai-engineering-essentials/lessons/03-tokens-and-costs.md)

### RAG returns irrelevant chunks

1. Chunk size (512–1024 tokens + overlap)
2. Embedding model match to domain
3. Hybrid search + reranking
4. See [Course 06 · RAG evaluation](build/module-09-rag-retrieval-augmented-generation/lessons/08-RAG-Evaluation-Metrics.md)

### Agent loops forever

1. **max iterations** in harness ([Course 08](build/module-18-agent-harness-tools-runtime/lessons/02-agent-loop-and-state.md))
2. Better tool descriptions ([Agent Engineering · Tools](agent-engineering/03-tools-and-mcp.md))
3. Trajectory evals ([Course 13](production/module-19-llm-evaluation-quality/lessons/04-agent-trajectory-evals.md))

### `mkdocs serve` fails

```bash
pip install -r requirements.txt
npm run build:docs   # full build + link fix
mkdocs serve
```

### Math overwhelming?

- [Course 01 · Math foundations](foundations/module-00-genai-foundations-from-nlp-to-transformers/lessons/02-math-foundations.md)
- [Deep Dives](deep-dives/index.md)
- Build first (Courses 02 → 06), return to math when motivated

---

## Career & portfolio

### What should I build for a portfolio?

Three projects beat ten tutorials: **RAG app**, **agent with tools**, **evaluated demo**. See [Build These First](projects/build-these.md).

### Is this enough to get a job?

Covers the **technical stack**. You still need shipped projects and system design practice. Use [Course 16 capstones](advanced/module-17-capstone-projects/index.md) as interview stories.

### How does this compare to other free resources?

We **curate and extend** OSS hubs with a unified path. See [Open Source Hubs](resources/open-source-hubs.md).

---

## Contributing

[Contribute](contribute.md) · [Roadmap](roadmap.md) · [GitHub Issues](https://github.com/psssnikhil/learn-ai-engineering/issues)
