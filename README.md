# AI Engineering Handbook

A **free, open-source, one-stop curriculum** for learning AI/ML engineering — from transformers and neural networks to RAG, agentic AI, harnesses, tools, orchestration, evals, and production observability.

**Live site:** https://psssnikhil.github.io/learn-ai-engineering/

**New here?** → [Start Here](docs/start-here.md) · [How to Learn](docs/learn/index.md) · [FAQ](docs/faq.md)

---

## What you get

| | |
|---|---|
| **16 modules** | Structured path across 4 phases |
| **140+ lessons** | Markdown lessons with code examples and diagrams |
| **Deep foundations** | M00–M07 expanded to engineer-grade depth (~2,500+ words per lesson) |
| **Hands-on** | Python exercises, capstone projects, Jupyter notebooks |
| **Hub pages** | Topic map, glossary, agentic AI guide, evals & observability guide |
| **Deep dives** | Extra math for attention, backprop, and tokenization |
| **License** | MIT — use, fork, and contribute freely |

---

## Who this is for

| You are… | Start here |
|----------|------------|
| **New to AI** | [M00 · GenAI Foundations](docs/foundations/module-00-genai-foundations-from-nlp-to-transformers/index.md) |
| **Software engineer, need LLMs** | [M01 · AI Engineering Essentials](docs/foundations/module-01-ai-engineering-essentials/index.md) |
| **Know ML, need transformers** | [M06 · Transformers](docs/foundations/module-06-transformers-attention-mechanisms/index.md) |
| **Building agents** | [Agentic AI Hub](docs/agentic-ai/index.md) → M11 → M18 → M12 |
| **Shipping to production** | [M10 · LLMOps](docs/production/module-10-llmops-production-systems/index.md) → [M19 · Evals](docs/production/module-19-llm-evaluation-quality/index.md) |

Full order: [Learning Path](docs/learning-path.md) · Find any topic: [Topic Map](docs/topic-map.md)

---

## Curriculum (4 phases)

### Foundations — core ML & LLMs
| Module | Topic |
|--------|-------|
| M00 | GenAI foundations: NLP → transformers |
| M01 | AI engineering essentials: APIs, tokens, prompts |
| M05 | Neural networks & deep learning |
| M06 | Transformers & attention mechanisms |
| M07 | Large language models (+ reasoning models extension) |

### Build — applications & agents
| Module | Topic |
|--------|-------|
| M09 | RAG (+ Graph RAG extension) |
| M11 | AI agents fundamentals |
| M18 | Agent harness, tools & runtime (MCP, observability) |
| M12 | Multi-agent systems & orchestration |
| M13 | Vector databases |
| M14 | Prompt engineering |

### Production — ship with confidence
| Module | Topic |
|--------|-------|
| M10 | LLMOps, observability, deployment |
| M19 | LLM evaluation & quality engineering |
| M16 | AI safety & ethics |

### Advanced — specialization
| Module | Topic |
|--------|-------|
| M15 | Fine-tuning & custom models |
| M17 | Capstone projects |

---

## Key hub pages

| Page | Purpose |
|------|---------|
| [Agent Engineering](docs/agent-engineering/index.md) | **Dedicated** loops, memory, harness, orchestration, observability, evals |
| [2026 Skills](docs/ai-engineering-2026/index.md) | Claude Code, skills, loop engineering, context engineering |
| [Getting Started](docs/getting-started.md) | Setup, navigation, OSS inspiration |
| [Topic Map](docs/topic-map.md) | Concept → module lookup |
| [Agentic AI](docs/agentic-ai/index.md) | Agents, harness, tools, orchestration |
| [Evals & Observability](docs/evals-observability/index.md) | Quality, tracing, monitoring |
| [Deep Dives](docs/deep-dives/index.md) | Mathematical foundations beyond lessons |
| [Glossary](docs/glossary.md) | Term definitions |
| [Resources](docs/resources/index.md) | Papers, videos, tools, OSS hubs |
| [Roadmap](docs/roadmap.md) | What's shipped and what's next |

---

## Run locally

```bash
git clone https://github.com/psssnikhil/learn-ai-engineering.git
cd learn-ai-engineering
pip install -r requirements.txt
mkdocs serve
```

Open http://127.0.0.1:8000

### Run a Python exercise

```bash
cd docs/foundations/module-01-ai-engineering-essentials/exercises
python 04-prompt-engineering-starter.py
```

Requires **Python 3.10+**. API keys only for lessons that call live LLM APIs.

---

## Repo structure

```
docs/
  foundations/          # M00, M01, M05, M06, M07
  build/                # M09, M11, M12, M13, M14, M18
  production/           # M10, M16, M19
  advanced/             # M15, M17
  deep-dives/           # Extra math (attention, backprop, tokenization)
  agentic-ai/           # Agentic AI hub
  evals-observability/  # Evals & monitoring hub
  resources/            # Papers, videos, tools, OSS hubs
  exercises/            # Exercise index

  Each module/
    index.md            # Overview + lesson table
    lessons/*.md        # Lesson content
    exercises/          # Python starters/solutions (+ notebooks)
```

Module folders use **canonical IDs** (`module-09-rag-...`) to stay aligned with the sibling [ai-learning](https://github.com/psssnikhil/ai-learning) platform.

---

## Content standards

Lessons follow [DEPTH_STANDARDS.md](DEPTH_STANDARDS.md):

1. Prerequisites → intuition → theory → worked example → code
2. Edge cases and common misconceptions
3. Production connection (why it matters for AI engineering)
4. Key takeaways and further reading

---

## Maintainer scripts

```bash
npm install
npm run migrate            # Re-import platform lessons (preserves handbook-native content)
npm run extract-resources  # Refresh paper/video/tool indexes from lesson links
npm run docs:build         # Build static site to ./site
```

**Handbook-native content** (not overwritten by migrate): M18, M19, M12 lessons 04–10, deep dives, hub pages.

---

## Contributing

1. Pick a lesson or hub page to improve
2. Follow [DEPTH_STANDARDS.md](DEPTH_STANDARDS.md)
3. Preview with `mkdocs serve`
4. Open a PR

See [CONTRIBUTING.md](CONTRIBUTING.md) · Known gaps: [GAPS.md](GAPS.md) · Public roadmap: [docs/roadmap.md](docs/roadmap.md)

---

## Related projects

| Repo | Relationship |
|------|--------------|
| [ai-learning](https://github.com/psssnikhil/ai-learning) | Sibling platform — lessons migrate from here |
| [Agents Towards Production](https://github.com/NirDiamant/agents-towards-production) | OSS inspiration — production agents |
| [Awesome Harness Engineering](https://github.com/ai-boost/awesome-harness-engineering) | OSS inspiration — harness patterns |
| [Awesome Agent Evals](https://github.com/benchflow-ai/awesome-evals) | OSS inspiration — evaluation |

---

## License

MIT — see [LICENSE](LICENSE).
