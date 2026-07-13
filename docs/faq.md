---
title: FAQ
description: Common questions, decision tables, and troubleshooting
---

# Frequently Asked Questions

Answers for learners at every level. Can't find yours? [Open an issue](https://github.com/psssnikhil/learn-ai-engineering/issues).

---

## Getting started

### Do I need a CS degree or ML background?

No. The handbook targets **software engineers and career switchers** who can write Python. [M00 L1](foundations/module-00-genai-foundations-from-nlp-to-transformers/lessons/01-prerequisites.md) covers the math you need (vectors, dot products, softmax). If you've never programmed, learn Python basics elsewhere first, then return to [Start Here](start-here.md).

### I'm a complete beginner with no CS background. Where do I start?

1. [M00 L1 Prerequisites](foundations/module-00-genai-foundations-from-nlp-to-transformers/lessons/01-prerequisites.md) — Python + NumPy check
2. [M00](foundations/module-00-genai-foundations-from-nlp-to-transformers/index.md) — NLP → transformers (conceptual)
3. [M01](foundations/module-01-ai-engineering-essentials/index.md) — first API call
4. [Build These #1](projects/build-these.md#1-doc-qa-bot-rag-starter) — first portfolio piece

Skip M05–M06 initially if you want to **build first, math later**. Return for depth when debugging model behavior.

### I'm a software engineer. Can I skip foundations?

Yes — start at [M01](foundations/module-01-ai-engineering-essentials/index.md). Skim [M07 L1–2](foundations/module-07-large-language-models-llms/index.md) if you want transformer context, then jump to [M09 RAG](build/module-09-rag-retrieval-augmented-generation/index.md) or [M11 Agents](build/module-11-ai-agents-fundamentals/index.md) based on your goal. See [Start Here routing tables](start-here.md#i-want-to-learn).

### I'm an ML engineer. What don't I need to re-read?

You can skip M05–M06 if you know backprop and attention. Start at [M07](foundations/module-07-large-language-models-llms/index.md) for LLM-specific content (tokenization, RLHF, APIs), then [M09](build/module-09-rag-retrieval-augmented-generation/index.md) / [M11](build/module-11-ai-agents-fundamentals/index.md) for product engineering. Your gap is usually **evals, harness, and production** — prioritize [M19](production/module-19-llm-evaluation-quality/index.md) and [M18](build/module-18-agent-harness-tools-runtime/index.md).

### How long does the full curriculum take?

| Pace | Duration |
|------|----------|
| Full-time | 8–12 weeks |
| Part-time (10–15 hr/week) | 4–6 months |
| Reference / jump-in | Use [Topic Map](topic-map.md) |

---

## Curriculum & navigation

### What's the difference between Getting Started, Start Here, and Learning Path?

| Page | Use when |
|------|----------|
| [Start Here](start-here.md) | **Pick your persona and goal** — main entry point |
| [Getting Started](getting-started.md) | Local setup (`mkdocs serve`, exercises) |
| [Learning Path](learning-path.md) | Full module order and phase overview |

### What's the difference between Agentic AI, Agent Engineering, and M11?

| Resource | Best for |
|----------|----------|
| [Agentic AI hub](agentic-ai/index.md) | Quick stack overview + OSS links |
| [Agent Engineering track](agent-engineering/index.md) | Focused 7-lesson agent curriculum |
| [M11 + M18 + M12](build/module-11-ai-agents-fundamentals/index.md) | Deep module lessons with exercises |

All three overlap — pick one spine and use the others as reference.

### Are exercises required?

Strongly recommended. Each module lesson ends with hands-on work. Central index: [Exercises](exercises/index.md). Portfolio-level work: [Build These First](projects/build-these.md).

### Where are capstone projects?

- Quick overview: [Projects index](projects/index.md)
- 10 mapped ideas: [Build These First](projects/build-these.md)
- Full briefs with acceptance criteria: [M17 Capstones](advanced/module-17-capstone-projects/index.md)

---

## Technical decisions

### RAG vs fine-tuning vs agents

| Approach | Cost | Time to ship | Best for | Not for |
|----------|------|--------------|----------|---------|
| **Prompt engineering** | Lowest (tokens only) | Hours | Format, simple tasks, prototyping | Large private knowledge bases |
| **RAG** | Medium (embeddings + vector DB) | Days | Factual Q&A over your docs, citations | Changing model personality reliably |
| **Fine-tuning** | High ($100–$10K+) | Days–weeks | Style, format, domain behavior | Injecting fresh facts (use RAG) |
| **Agents** | Medium–high (many LLM calls) | Days–weeks | Multi-step tasks, tool use, research | Simple single-shot Q&A |
| **Workflows** | Low–medium | Days | Deterministic pipelines, approvals | Open-ended exploration |

**Decision flow:**

```
Can prompts solve it? → Yes: prompt engineering
                      → No: Need your documents? → Yes: RAG
                                              → No: Need consistent behavior/style? → Yes: fine-tune
                                                                                    → No: multi-step + tools? → Yes: agent
                                                                                                              → No: try different base model
```

Deep dive: [M15 L1 — When to Fine-Tune](advanced/module-15-fine-tuning-custom-models/lessons/01-lesson-01.md) · [M11 L10 — Workflow vs Agent](build/module-11-ai-agents-fundamentals/lessons/10-Workflow-vs-Agent.md)

### When should I use an agent vs a workflow?

Use a **workflow** when steps are fixed (ingest → classify → route → respond). Use an **agent** when the path depends on intermediate results (research, coding, tool selection). Hybrid is common in production. See [M11 L10](build/module-11-ai-agents-fundamentals/lessons/10-Workflow-vs-Agent.md).

### RAG vs Graph RAG?

| Pattern | When |
|---------|------|
| **Vector RAG** | Most doc Q&A, semantic search |
| **Hybrid search** | Keyword + semantic matter (SKUs, legal cites) |
| **Graph RAG** | Many entities, relationships, "how does X connect to Y" | 

See [M09 L11 Graph RAG](build/module-09-rag-retrieval-augmented-generation/lessons/11-graph-rag-and-knowledge-graphs.md).

### Do I need to fine-tune or is RAG enough?

For **knowledge**: RAG almost always. For **behavior** (tone, JSON schema compliance, medical report format): fine-tune or strict structured output. For **both**: RAG + fine-tuned model is valid but start with one.

---

## Tools & setup

### What do I need installed?

- Python 3.10+
- `pip install -r requirements.txt` (for local site)
- API keys (OpenAI, Anthropic, etc.) only for lessons/exercises that call live models

See [Getting Started — Local setup](getting-started.md#local-setup).

### Which LLM provider should I use?

| Provider | Good for |
|----------|----------|
| **OpenAI** | Broad docs, function calling, fine-tuning API |
| **Anthropic** | Long context, Claude tool use |
| **Open source (Ollama, vLLM)** | Local dev, cost control, fine-tuning |

M01 and M07 cover API patterns that transfer across providers.

### Do I need a GPU?

Not for most of this handbook. Cloud APIs cover inference. Fine-tuning lessons (M15) may use Colab or cloud GPUs — alternatives are noted in lessons.

---

## Troubleshooting

### `ModuleNotFoundError` when running exercises

```bash
cd docs/foundations/module-01-ai-engineering-essentials/exercises
pip install openai anthropic python-dotenv  # per exercise README if present
python 04-prompt-engineering-starter.py
```

Run from the **exercise folder** listed in [Exercises index](exercises/index.md).

### API key errors (`401`, `invalid_api_key`)

1. Set env var: `export OPENAI_API_KEY=sk-...` (or use `.env` with `python-dotenv`)
2. Confirm key is active in provider dashboard
3. Check you're using the right env var name for the SDK (`OPENAI_API_KEY` vs `ANTHROPIC_API_KEY`)

### Rate limits / `429` errors

- Add retries with exponential backoff
- Reduce batch size in embedding exercises
- Use a smaller/cheaper model for dev (`gpt-4o-mini`, `claude-3-haiku`)
- See [M01 L3 Tokens and Costs](foundations/module-01-ai-engineering-essentials/lessons/03-tokens-and-costs.md)

### RAG returns irrelevant chunks

Common fixes (in order):

1. **Chunk size** — try 512–1024 tokens with overlap ([M09 L3](build/module-09-rag-retrieval-augmented-generation/lessons/03-chunking-strategies.md))
2. **Embedding model** — match model to domain
3. **Hybrid search** — add BM25 ([M09 L7](build/module-09-rag-retrieval-augmented-generation/lessons/07-Hybrid-Search.md))
4. **Reranking** — cross-encoder second stage ([M09 L6](build/module-09-rag-retrieval-augmented-generation/lessons/06-Advanced-RAG-Techniques.md))

### Agent loops forever or ignores tools

1. Set **max iterations** in harness ([M18 L2](build/module-18-agent-harness-tools-runtime/lessons/02-agent-loop-and-state.md))
2. Improve **tool descriptions** ([Agent Engineering L3](agent-engineering/03-tools-and-mcp.md))
3. Add **termination conditions** and evals ([M19 L4](production/module-19-llm-evaluation-quality/lessons/04-agent-trajectory-evals.md))

### `mkdocs serve` fails

```bash
pip install -r requirements.txt
mkdocs serve
```

Requires Python 3.10+. If port 8000 is busy: `mkdocs serve -a 127.0.0.1:8001`.

### Math in lessons is overwhelming

- [M00 L2 Math Foundations](foundations/module-00-genai-foundations-from-nlp-to-transformers/lessons/02-math-foundations.md) — rework slowly
- [Deep Dives](deep-dives/index.md) — optional engineer-grade supplements
- You can **build with APIs first** (M01 → M09) and return to M05–M06 when motivated

---

## Career & portfolio

### What should I build for a portfolio?

Three projects beat ten tutorials: one **RAG app**, one **agent with tools**, one **evaluated production demo**. See [Build These First](projects/build-these.md) for ten mapped ideas.

### Is this enough to get a job?

This handbook covers the **technical stack** (LLMs, RAG, agents, evals, production). You still need: shipped projects, system design practice, and domain interest. Use M17 capstones as interview talking points.

### How does this compare to other free resources?

We **curate and extend** OSS hubs — not duplicate them. See [Open Source Hubs](resources/open-source-hubs.md) and [Courses & Communities](resources/courses-and-communities.md). This site adds a unified learning path, hands-on exercises, agent/eval tracks, and production-focused modules.

---

## Contributing

### I found an error or missing topic

[Contribute](contribute.md) · [Roadmap](roadmap.md) · [GitHub Issues](https://github.com/psssnikhil/learn-ai-engineering/issues)

Known gaps: M09/M11 exercise notebooks, more Deep Dives (RoPE, LoRA, KV cache). See [GAPS.md](https://github.com/psssnikhil/learn-ai-engineering/blob/main/GAPS.md) in the repo.
