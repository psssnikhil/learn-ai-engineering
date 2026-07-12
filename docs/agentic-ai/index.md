---
title: Agentic AI
---

# Agentic AI — One-Stop Guide

Everything you need to design, build, and ship **autonomous AI systems**: single agents, harnesses, tools, and multi-agent orchestration.

## The agentic stack

```mermaid
flowchart TB
  subgraph Runtime["Harness & Runtime"]
    Loop[Agent Loop]
    State[State / Memory]
    Term[Termination]
    Perm[Permissions]
  end
  subgraph Capabilities["Capabilities"]
    LLM[LLM Reasoning]
    Tools[Tools & MCP]
    RAG[RAG / Retrieval]
  end
  subgraph Scale["Scale Out"]
    Orch[Orchestrator]
    Workers[Worker Agents]
    Handoff[Handoffs]
  end
  User --> Runtime
  Runtime --> Capabilities
  Runtime --> Scale
  Runtime --> Obs[Observability]
```

## Learning path (agentic track)

| Step | Module | What you'll learn |
|------|--------|-------------------|
| 1 | [M11 · Agents](../build/module-11-ai-agents-fundamentals/index.md) | Agent loop, ReAct, tool use, frameworks |
| 2 | [M18 · Harness & Tools](../build/module-18-agent-harness-tools-runtime/index.md) | Runtime primitives, MCP, safety, tracing |
| 3 | [M12 · Multi-Agent](../build/module-12-multi-agent-systems/index.md) | Orchestration, coordination, patterns |
| 4 | [M09 · Agentic RAG](../build/module-09-rag-retrieval-augmented-generation/index.md) | Retrieval-driven agents |
| 5 | [M17 · Capstones](../advanced/module-17-capstone-projects/index.md) | End-to-end agent projects |

## Core concepts

| Concept | Handbook | OSS inspiration |
|---------|----------|-----------------|
| **Agent loop** | [M11 L1](../build/module-11-ai-agents-fundamentals/lessons/01-Introduction-to-Agents.md) | [Microsoft AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners) |
| **Harness** | [M18](../build/module-18-agent-harness-tools-runtime/index.md) | [Awesome Harness Engineering](https://github.com/ai-boost/awesome-harness-engineering) |
| **Tools & MCP** | [M18 L3–4](../build/module-18-agent-harness-tools-runtime/index.md) | [Model Context Protocol](https://modelcontextprotocol.io/) |
| **Orchestration** | [M12](../build/module-12-multi-agent-systems/index.md) | [Agents Towards Production](https://github.com/NirDiamant/agents-towards-production) |
| **Workflow vs agent** | [M11 L10](../build/module-11-ai-agents-fundamentals/lessons/10-Workflow-vs-Agent.md) | LangGraph docs |

## When to use what

| Problem | Pattern | Module |
|---------|---------|--------|
| Single task with tools | ReAct agent | M11 |
| Long-running coding agent | Harness + sandbox | M18 |
| Research across sources | Multi-agent + RAG | M12, M09 |
| Deterministic pipeline | Workflow (not agent) | M11 L10 |
| Customer support | Orchestrator + specialists | M12 L3 |

## Visual references

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — attention foundation for reasoning models
- [How GPT-3 Works](https://jalammar.github.io/how-gpt3-works-visualizations-animations/) — token prediction → tool selection analogy

## Related

- [Evals & Observability](../evals-observability/index.md) — measure and debug agent runs
- [Topic Map](../topic-map.md) — find any concept
- [Glossary](../glossary.md) — agent, harness, MCP, trajectory eval
