# AI Engineering Handbook

Open-source knowledge base for learning **AI engineering** — from transformers and LLMs to RAG, agents, multi-agent systems, and production LLMOps.

Read it on the web, clone it for offline study, or contribute new lessons and resources.

## What is this?

A **lightweight, growing repository** of:

- **Markdown lessons** — structured learning path across 4 phases
- **Python exercises** — `starter.py` / `solution.py` pairs where hands-on practice matters
- **Jupyter notebooks** — added incrementally for interactive sections
- **Curated resources** — papers, videos, tools, and links in `/resources`

This is the **content layer**. The [AI Engineering Mastery](https://github.com/psssnikhil/ai-for-all) platform is a separate product that can consume this corpus over time.

## Quick start

### Read online

GitHub Pages (after deploy): `https://psssnikhil.github.io/learn-ai-engineering/`

### Browse locally

```bash
git clone https://github.com/psssnikhil/learn-ai-engineering.git
cd learn-ai-engineering
pip install -r requirements.txt
mkdocs serve
# Open http://127.0.0.1:8000
```

### Run exercises

```bash
cd docs/foundations/01-ai-engineering-essentials/exercises
python 04-starter.py
```

## Learning path

| Phase | Topics |
|-------|--------|
| **Foundations** | GenAI basics, neural nets, transformers, LLMs |
| **Build** | RAG, agents, multi-agent, vector DBs, prompts |
| **Production** | LLMOps, monitoring, safety & ethics |
| **Advanced** | Fine-tuning, capstone projects |

See [docs/learning-path.md](docs/learning-path.md) for the full module list.

## Contributing

We welcome PRs for:

- Lesson fixes and clarifications
- New resources in `resources/`
- Python exercises and notebooks
- Filling gaps (e.g. Multi-Agent Systems module)

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT — see [LICENSE](LICENSE). External links and third-party resources remain property of their respective authors.

## Sync from platform

Content is migrated from the `ai-learning` platform with:

```bash
npm install
npm run migrate
```
