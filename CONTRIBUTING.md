# Contributing to AI Engineering Handbook

Thank you for helping grow this open knowledge base.

## Ways to contribute

1. **Fix a lesson** — typos, clearer explanations, updated code samples
2. **Add a resource** — papers, videos, or tools in `resources/`
3. **Add an exercise** — `exercises/*-starter.py` + `*-solution.py` in a module folder
4. **Add a notebook** — `.ipynb` alongside an exercise for interactive learning
5. **Fill content gaps** — see module READMEs with incomplete lesson counts

## Lesson frontmatter

```yaml
---
title: "Your Lesson Title"
description: "One-line summary"
duration: "30 min"
difficulty: beginner   # beginner | intermediate | advanced
has_code: false
youtube: "https://..."  # optional
objectives:           # optional
  - "What the learner will know"
---
```

## File layout

```
docs/
  {phase}/                    # foundations | build | production | advanced
    {NN}-{module-slug}/
      index.md                # module overview
      01-lesson-slug.md
      exercises/
        01-starter.py
        01-solution.py
        01-lesson.ipynb       # optional
```

## Pull request checklist

- [ ] Markdown renders correctly (`mkdocs serve` locally)
- [ ] Links work (no broken URLs where you can help it)
- [ ] Original writing or properly attributed quotes
- [ ] No secrets, API keys, or `.env` files
- [ ] Python exercises run with stated dependencies

## Content gaps (known)

- **Multi-Agent Systems** — partial module; more lessons planned
- **Notebooks** — being added module by module
- **Central bibliography** — run `npm run extract-resources` to refresh link indexes

## Code of conduct

Be constructive, cite sources, and optimize for learners at different levels.
