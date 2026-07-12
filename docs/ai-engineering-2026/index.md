---
title: AI Engineering 2026
description: Modern skills — Claude Code, agent skills, loop engineering, context engineering
---

# AI Engineering in 2026

New primitives that didn't exist in the 2023 "just call the API" era. This section covers **skills you need now** — complementary to the core curriculum.

```mermaid
flowchart LR
  CC[Claude Code] --> Loop[Loop engineering]
  Skills[Skills & rules] --> Loop
  Loop --> Harness[Harness]
  Ctx[Context engineering] --> Harness
  Harness --> Ship[Production agents]
```

## Topics in this section

| Topic | What you'll learn | Page |
|-------|-------------------|------|
| **Claude Code** | Agentic terminal coding, permissions, workflows | [Claude Code](claude-code.md) |
| **Skills & rules** | SKILL.md, Cursor skills, persistent agent instructions | [Skills & Rules](skills-and-rules.md) |
| **Loop engineering** | Inner/outer loops, /loop, scheduled agents, harness cycles | [Loop Engineering](loop-engineering.md) |
| **Context engineering** | What goes in the window — the new prompt engineering | [Context Engineering](context-engineering.md) |

## How this fits the handbook

| 2026 skill | Handbook foundation |
|------------|---------------------|
| Claude Code | [Agent Engineering](../agent-engineering/index.md) → [Harness](../agent-engineering/04-harness-engineering.md) |
| Skills files | [M14 · Prompts](../build/module-14-prompt-engineering-mastery/index.md) |
| Loop engineering | [Agent Loop](../agent-engineering/01-agent-loop.md), [M18](../build/module-18-agent-harness-tools-runtime/index.md) |
| Context engineering | [M01 · Tokens](../foundations/module-01-ai-engineering-essentials/lessons/03-tokens-and-costs.md), [Memory](../agent-engineering/02-memory.md) |

## What's still evolving (2026)

| Trend | Status | Handbook coverage |
|-------|--------|-------------------|
| Reasoning models (o3, R1) | Production | [M07 L11](../foundations/module-07-large-language-models-llms/lessons/11-reasoning-models-and-test-time-compute.md) |
| MCP everywhere | Standardizing | [M18 L4](../build/module-18-agent-harness-tools-runtime/lessons/04-mcp-model-context-protocol.md) |
| Computer use agents | Early production | Planned |
| Agent skills marketplaces | Emerging | [Skills & Rules](skills-and-rules.md) |
| Eval-driven development | Best practice | [M19](../production/module-19-llm-evaluation-quality/index.md) |

## OSS hubs to watch

- [Awesome Harness Engineering](https://github.com/ai-boost/awesome-harness-engineering)
- [Anthropic Claude Code docs](https://docs.anthropic.com/en/docs/claude-code)
- [Cursor Skills](https://cursor.com/docs/context/skills) (ecosystem)
- [Related papers](related-papers.md) — DeepSeek-R1, SWE-agent, DSPy, Lost in the Middle, and more

**Start:** [Claude Code →](claude-code.md)
