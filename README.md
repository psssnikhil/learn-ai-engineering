# AI Engineering Handbook

Open-source knowledge base for learning **AI engineering** — agents, RAG, LLMOps, and more.

**Live site:** https://psssnikhil.github.io/learn-ai-engineering/

## Structure

```
docs/
  foundations/     module-00, 01, 05, 06, 07
  build/           module-09, 11, 12, 13, 14
  production/      module-10, 16
  advanced/        module-15, 17

  Each module/
    index.md       overview + lesson table
    lessons/       markdown lessons
    exercises/     Python starter/solution files
```

Module folder names use **canonical IDs** (`module-09-rag-...`) so content stays aligned with the [AI Engineering Mastery](https://github.com/psssnikhil/ai-for-all) platform.

## Quick start

```bash
git clone https://github.com/psssnikhil/learn-ai-engineering.git
cd learn-ai-engineering
pip install -r requirements.txt
mkdocs serve   # http://127.0.0.1:8000
```

## Sync from platform

```bash
npm install
npm run migrate            # re-import lessons from ai-learning
npm run extract-resources  # refresh paper/video indexes
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Known gaps: [GAPS.md](GAPS.md).

## License

MIT — see [LICENSE](LICENSE).
