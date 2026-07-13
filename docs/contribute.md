---
title: Contribute
---

# Contribute

The AI Engineering Handbook is MIT-licensed and community-driven. **PRs welcome** — typos, clearer explanations, exercises, and new lessons.

If this project helps you, **[star it on GitHub](https://github.com/psssnikhil/learn-ai-engineering)** so others can find it.

## Quick wins

| Type | How |
|------|-----|
| **Fix a lesson** | Edit markdown in `docs/`, open a PR |
| **Fix a broken link** | Use trailing-slash paths in HTML (`href="page/"`), not `.md` |
| **Add an exercise** | `exercises/*-starter.py` + solution in a course folder |
| **Report a gap** | [Open an issue](https://github.com/psssnikhil/learn-ai-engineering/issues/new/choose) |

## Content standards

1. **Frontmatter** — `title`, `description`, `duration`, `difficulty`
2. **Structure** — objectives → content → key takeaways → next lesson
3. **Code** — Python 3.10+, runnable where `has_code: true`
4. **Tone** — learner-facing only (no author meta, word counts, or internal standards on pages)
5. **Links** — markdown links OK; raw HTML must use `href="path/"` not `href="path.md"`

## Local preview

```bash
pip install -r requirements.txt
npm install
mkdocs serve          # preview
npm run build:docs    # full build + link fix (same as CI)
```

## Adding a course to the curriculum

1. Add content under `docs/{phase}/module-NN-{slug}/`
2. Edit `curriculum.yml` in order
3. Run `npm run sync-nav`

See [MAINTAINING.md](https://github.com/psssnikhil/learn-ai-engineering/blob/main/MAINTAINING.md).

## Sync from platform

```bash
npm run migrate
```

Handbook-native courses (Harness, Evals, Multi-Agent extensions) are preserved.

Full guide: [CONTRIBUTING.md](https://github.com/psssnikhil/learn-ai-engineering/blob/main/CONTRIBUTING.md)
