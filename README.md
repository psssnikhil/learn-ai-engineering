# AI Engineering Handbook

**The free, open-source path from zero to production AI** — transformers, RAG, agents, harnesses, evals, and LLMOps in one sequential curriculum.

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Live site](https://img.shields.io/badge/docs-live-indigo.svg)](https://psssnikhil.github.io/learn-ai-engineering/)
[![GitHub stars](https://img.shields.io/github/stars/psssnikhil/learn-ai-engineering?style=social)](https://github.com/psssnikhil/learn-ai-engineering)

**[Read the handbook →](https://psssnikhil.github.io/learn-ai-engineering/)** · **[Start Here](https://psssnikhil.github.io/learn-ai-engineering/start-here/)** · **[Browse 16 courses](https://psssnikhil.github.io/learn-ai-engineering/learn/)**

If this helps you learn or build — **star the repo**. It’s the main way others discover open-source projects like this.

---

## Why this exists

Most AI content is either **too shallow** (blog posts) or **too scattered** (awesome lists with no order). This handbook is:

- **One path** — 16 courses in order, from NLP basics to capstone projects
- **Engineer-grade depth** — foundations and production lessons written for builders, not tourists
- **Agent-native** — RAG, harnesses, MCP, multi-agent orchestration, trajectory evals
- **Free forever** — MIT license, no paywall, community-maintained

---

## What's inside

| | |
|---|---|
| **16 courses** | Sequential curriculum across 4 parts |
| **140+ lessons** | Markdown with code, diagrams, and worked examples |
| **2 optional tracks** | Agent Engineering · Modern AI (2026) |
| **Hub pages** | Start Here, Topic Map, FAQ, Study Plans, Build These |
| **Deep dives** | Attention math, backprop, tokenization |
| **Hands-on** | Python exercises, capstone projects, notebooks |

---

## Who this is for

| You are… | Start here |
|----------|------------|
| **New to AI** | [Start Here](https://psssnikhil.github.io/learn-ai-engineering/start-here/) → Course 01 |
| **Software engineer** | [AI Engineering Essentials](https://psssnikhil.github.io/learn-ai-engineering/foundations/module-01-ai-engineering-essentials/) |
| **ML engineer → LLMs** | [Large Language Models](https://psssnikhil.github.io/learn-ai-engineering/foundations/module-07-large-language-models-llms/) |
| **Building agents** | [Agent Engineering track](https://psssnikhil.github.io/learn-ai-engineering/agent-engineering/) |
| **Shipping to production** | [LLMOps](https://psssnikhil.github.io/learn-ai-engineering/production/module-10-llmops-production-systems/) → [Evals](https://psssnikhil.github.io/learn-ai-engineering/production/module-19-llm-evaluation-quality/) |

Full order: [Learn overview](https://psssnikhil.github.io/learn-ai-engineering/learn/) · [Study Plans](https://psssnikhil.github.io/learn-ai-engineering/learn/study-plans/)

---

## Curriculum (16 courses)

### Part 1 — Understand AI
| # | Course |
|---|--------|
| 01 | GenAI Foundations |
| 02 | AI Engineering Essentials |
| 03 | Neural Networks & Deep Learning |
| 04 | Transformers & Attention |
| 05 | Large Language Models |

### Part 2 — Build applications
| # | Course |
|---|--------|
| 06 | RAG |
| 07 | AI Agents |
| 08 | Agent Harness & Runtime |
| 09 | Multi-Agent Systems |
| 10 | Vector Databases |
| 11 | Prompt Engineering |

### Part 3 — Production
| # | Course |
|---|--------|
| 12 | LLMOps & Production Systems |
| 13 | LLM Evaluation & Quality |
| 14 | AI Safety & Ethics |

### Part 4 — Advanced
| # | Course |
|---|--------|
| 15 | Fine-Tuning & Custom Models |
| 16 | Capstone Projects |

---

## Run locally

```bash
git clone https://github.com/psssnikhil/learn-ai-engineering.git
cd learn-ai-engineering
pip install -r requirements.txt
npm install
npm run build:docs    # build static site
mkdocs serve          # preview at http://127.0.0.1:8000
```

---

## Maintainer scripts

```bash
npm run sync-nav       # Regenerate nav from curriculum.yml
npm run migrate        # Sync lessons from ai-learning platform
npm run build:docs     # mkdocs build + GitHub Pages link fix
```

See [MAINTAINING.md](MAINTAINING.md) for adding new courses.

---

## Contributing

1. Pick a lesson or hub page
2. Preview with `mkdocs serve`
3. Open a PR

[CONTRIBUTING.md](CONTRIBUTING.md) · [Roadmap](docs/roadmap.md) · [Known gaps](GAPS.md)

---

## Brand

| | |
|---|---|
| **Name** | AI Engineering Handbook |
| **Tagline** | The open-source path from zero to production AI |
| **Site** | [psssnikhil.github.io/learn-ai-engineering](https://psssnikhil.github.io/learn-ai-engineering/) |
| **License** | MIT |

---

## Related projects

| Repo | Relationship |
|------|--------------|
| [ai-learning](https://github.com/psssnikhil/ai-learning) | Sibling platform — lesson source |
| [Agents Towards Production](https://github.com/NirDiamant/agents-towards-production) | OSS inspiration |
| [Awesome Harness Engineering](https://github.com/ai-boost/awesome-harness-engineering) | OSS inspiration |

---

## License

MIT — see [LICENSE](LICENSE).
