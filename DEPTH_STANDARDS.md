# Content Depth Standards

All handbook lessons must meet this bar. Start accessible; end at engineer-grade depth.

## Structure (every lesson)

1. **Prerequisites** — what you must know; links to prior lessons
2. **What You'll Learn** — objectives table
3. **Intuition first** — plain-language mental model before jargon
4. **Core theory** — definitions, math (with intuition), diagrams
5. **Worked example** — step-by-step, numbers or code
6. **Implementation** — runnable Python where `has_code: true`
7. **Edge cases & misconceptions** — what beginners get wrong
8. **Production connection** — why this matters for AI engineering
9. **Key takeaways** — 5–8 bullets
10. **Further reading** — papers, Jalammar, OSS repos
11. **Next lesson** — link

## Depth targets

| Level | Word count | Code blocks | Diagrams |
|-------|------------|-------------|----------|
| Beginner intro | 1,500–2,500 | 2–4 | 1–2 mermaid/ascii |
| Intermediate | 2,500–4,000 | 4–8 | 2–3 |
| Advanced | 3,500–5,000 | 6–12 | 2–4 |

## Quality anti-patterns (remove these)

- Marketing fluff ("biggest shift since the internet")
- Salary tables without educational value
- Stock Unsplash images without pedagogical purpose
- Bullet lists without explanation
- "See docs" without teaching the concept

## Quality patterns (add these)

- "Why not X?" comparisons
- Numerical walkthroughs (e.g. attention scores on a 4-word sentence)
- Shape annotations on tensors: `(batch, seq, dim)`
- Links to [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- `!!! note` / `!!! warning` admonitions for subtle points

## Agent ownership (avoid conflicts)

| Agent | Owns |
|-------|------|
| Quality | `docs/foundations/module-00-*`, `module-01-*`, `module-05-*` lessons |
| Coverage | New lessons, extensions, notebooks, `GAPS.md`, `roadmap.md`, `docs/deep-dives/` content |
| Site UX | `mkdocs.yml`, `docs/index.md`, `docs/stylesheets/`, theme, hub pages |

## Deep Dives nav (action required by Site agent)

`docs/deep-dives/` was created by the Coverage agent (2026-07-12) and contains:
- `index.md` — hub page
- `attention-math.md`
- `backpropagation-calculus.md`
- `tokenization-internals.md`

**Site agent**: please add a `Deep Dives` nav section to `mkdocs.yml` pointing to
these pages. Suggested nav block:

```yaml
- Deep Dives:
  - deep-dives/index.md
  - deep-dives/attention-math.md
  - deep-dives/backpropagation-calculus.md
  - deep-dives/tokenization-internals.md
```
