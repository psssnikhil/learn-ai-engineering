---
title: Contribute
---

# Contribute

The AI Engineering Handbook is MIT-licensed and community-driven. Help make it the best one-stop shop for AI/ML engineering.

## Ways to contribute

| Type | How |
|------|-----|
| **Fix a lesson** | Edit markdown in `docs/`, open a PR |
| **Add a lesson** | Follow module structure under `docs/{phase}/module-NN-*/lessons/` |
| **Curate resources** | Edit `docs/resources/essential-*.md` |
| **Report gaps** | Open a GitHub issue or update [Roadmap](roadmap.md) |

## Content standards

1. **Frontmatter** — `title`, `description`, `duration`, `difficulty`, `module`
2. **Structure** — objectives table → content → key takeaways → next lesson link
3. **Code** — runnable Python 3.10+ where `has_code: true`
4. **Links** — prefer handbook cross-links + curated OSS references
5. **Images** — use open educational sources (Jalammar, Wikimedia) with attribution; mermaid for diagrams

## Local preview

```bash
git clone https://github.com/psssnikhil/learn-ai-engineering.git
cd learn-ai-engineering
pip install -r requirements.txt
mkdocs serve
```

## Sync from platform

Lessons migrated from the [ai-learning](https://github.com/psssnikhil/ai-learning) platform use:

```bash
npm run migrate
```

Handbook-native modules (M18, M19) are **not** overwritten by migrate.

## Module numbering

- Use canonical IDs: `module-09`, `module-18`, etc.
- Folder pattern: `module-NN-{slug}/`

See [CONTRIBUTING.md](https://github.com/psssnikhil/learn-ai-engineering/blob/main/CONTRIBUTING.md) on GitHub for the full guide.
