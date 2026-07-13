---
title: Topic Map
---

# Topic Map

Find any concept across the handbook. Module IDs match the platform (`module-09` = RAG).

<div class="quick-nav">
  <a class="quick-nav__item" href="getting-started.md">Getting Started</a>
  <a class="quick-nav__item" href="learning-path.md">Learning Path</a>
  <a class="quick-nav__item" href="agent-engineering/index.md">Agent Engineering</a>
  <a class="quick-nav__item" href="ai-engineering-2026/index.md">2026 Skills</a>
  <a class="quick-nav__item" href="glossary.md">Glossary</a>
</div>

!!! tip "Can't find it?"
    Try [Glossary](glossary.md) for term definitions, or use the search bar (⌘K / Ctrl+K).

## Full learning arc

```mermaid
flowchart TB
  subgraph F["Foundations"]
    M00[M00 GenAI / Transformers]
    M01[M01 AI Engineering]
    M05[M05 Neural Networks]
    M06[M06 Attention]
    M07[M07 LLMs]
  end
  subgraph B["Build"]
    M09[M09 RAG]
    M11[M11 Agents]
    M18[M18 Harness & Tools]
    M12[M12 Multi-Agent]
    M13[M13 Vector DBs]
    M14[M14 Prompts]
  end
  subgraph P["Production"]
    M10[M10 LLMOps]
    M19[M19 Evals]
    M16[M16 Safety]
  end
  subgraph A["Advanced"]
    M15[M15 Fine-Tuning]
    M17[M17 Capstones]
  end
  subgraph T["Supplemental tracks"]
    AE[Agent Engineering]
    SK[2026 Skills]
  end
  M00 --> M01 --> M09
  M06 --> M07 --> M09
  M09 --> M11 --> M18 --> M12
  M11 --> M10 --> M19
  M10 --> M17
  M15 --> M17
  M11 --> AE
  AE --> SK
```

## Concept → module index

| Topic | Primary modules | Also covered in |
|-------|-----------------|-----------------|
| **Neural networks & deep learning** | [M05](foundations/module-05-neural-networks-deep-learning-fundamentals/index.md) | M00 |
| **Transformers & attention** | [M00](foundations/module-00-genai-foundations-from-nlp-to-transformers/index.md), [M06](foundations/module-06-transformers-attention-mechanisms/index.md) | M07 |
| **LLMs & APIs** | [M07](foundations/module-07-large-language-models-llms/index.md) | M01 |
| **Reasoning models** | [M07 L11](foundations/module-07-large-language-models-llms/lessons/11-reasoning-models-and-test-time-compute.md) | — |
| **Prompt engineering** | [M14](build/module-14-prompt-engineering-mastery/index.md) | M01 |
| **Context engineering** | [2026 Skills](ai-engineering-2026/context-engineering.md) | M01 L3, [Agent Engineering · Memory](agent-engineering/02-memory.md) |
| **RAG** | [M09](build/module-09-rag-retrieval-augmented-generation/index.md) | M13, M17 |
| **Vector search** | [M13](build/module-13-vector-databases-deep-dive/index.md) | M09 |
| **AI agents** | [M11](build/module-11-ai-agents-fundamentals/index.md) | M09 L9, [Agent Engineering](agent-engineering/index.md) |
| **Agent harness & runtime** | [M18](build/module-18-agent-harness-tools-runtime/index.md) | [Agent Engineering · Harness](agent-engineering/04-harness-engineering.md) |
| **Tools & MCP** | [M18](build/module-18-agent-harness-tools-runtime/index.md) | M11 L4, [Agent Engineering · Tools](agent-engineering/03-tools-and-mcp.md) |
| **Loop engineering** | [2026 Skills](ai-engineering-2026/loop-engineering.md) | [Agent Engineering · Loop](agent-engineering/01-agent-loop.md), M18 |
| **Orchestration** | [M12](build/module-12-multi-agent-systems/index.md) | M11 L8, M18, [Agent Engineering · Orchestration](agent-engineering/05-orchestration.md) |
| **Multi-agent systems** | [M12](build/module-12-multi-agent-systems/index.md) | M17 |
| **Claude Code & IDE agents** | [2026 Skills](ai-engineering-2026/claude-code.md) | [Skills & Rules](ai-engineering-2026/skills-and-rules.md) |
| **LLMOps** | [M10](production/module-10-llmops-production-systems/index.md) | M17 |
| **Observability & monitoring** | [M10](production/module-10-llmops-production-systems/index.md) | M18, M19, [Agent Engineering · Observability](agent-engineering/06-observability-and-tracing.md) |
| **Evaluation** | [M19](production/module-19-llm-evaluation-quality/index.md) | M09 L8, M16, [Agent Engineering · Evals](agent-engineering/07-agent-evals.md) |
| **Safety & red teaming** | [M16](production/module-16-ai-safety-ethics/index.md) | M14, M19 |
| **Fine-tuning** | [M15](advanced/module-15-fine-tuning-custom-models/index.md) | M07 |

## Agentic AI stack

```mermaid
flowchart LR
  User[User / API] --> Harness[Agent Harness]
  Harness --> LLM[LLM]
  Harness --> Tools[Tools / MCP]
  Harness --> Memory[Memory]
  Harness --> Orch[Orchestrator]
  Orch --> A1[Agent A]
  Orch --> A2[Agent B]
  Harness --> Obs[Observability]
  Obs --> Evals[Evals]
```

Deep dive: [Agentic AI hub](agentic-ai/index.md) · **[Agent Engineering track](agent-engineering/index.md)**

## Agent engineering (dedicated track)

| Topic | Page | Related module |
|-------|------|----------------|
| Agent loop | [01 · Loop](agent-engineering/01-agent-loop.md) | M11 |
| Memory | [02 · Memory](agent-engineering/02-memory.md) | M11 L5 |
| Tools & MCP | [03 · Tools](agent-engineering/03-tools-and-mcp.md) | M18 |
| Harness engineering | [04 · Harness](agent-engineering/04-harness-engineering.md) | M18 |
| Orchestration | [05 · Orchestration](agent-engineering/05-orchestration.md) | M12 |
| Observability & tracing | [06 · Observability](agent-engineering/06-observability-and-tracing.md) | M10, M18 |
| Agent evals | [07 · Evals](agent-engineering/07-agent-evals.md) | M19 |

## 2026 skills

| Topic | Page | Related module |
|-------|------|----------------|
| Overview | [AI Engineering 2026](ai-engineering-2026/index.md) | — |
| Claude Code | [claude-code.md](ai-engineering-2026/claude-code.md) | Agent Engineering |
| Skills & rules | [skills-and-rules.md](ai-engineering-2026/skills-and-rules.md) | M14 |
| Loop engineering | [loop-engineering.md](ai-engineering-2026/loop-engineering.md) | M18, Agent Engineering |
| Context engineering | [context-engineering.md](ai-engineering-2026/context-engineering.md) | M01, Agent Engineering |

## Deep Dives (mathematical foundations)

Go beyond lesson summaries for full derivations:

- [Attention Math](deep-dives/attention-math.md) — QKV derivation with numpy
- [Backpropagation Calculus](deep-dives/backpropagation-calculus.md) — chain rule through a 2-layer net
- [Tokenization Internals](deep-dives/tokenization-internals.md) — BPE merge rules worked by hand

See [Deep Dives hub](deep-dives/index.md).

## Visual references (open educational resources)

| Diagram | Source | Topic |
|---------|--------|-------|
| [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | Jay Alammar | M00, M06 |
| [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) | Jay Alammar | M07 |
| [LLM Visualization](https://bbycroft.net/llm) | Brendan Bycroft | M07 |
| [RAG diagram](https://github.com/NirDiamant/RAG_Techniques) | RAG Techniques repo | M09 |

## Cross-cutting guides

- [Getting Started](getting-started.md) — pick your entry point by background
- [Learning Path](learning-path.md) — full 16-module curriculum
- [Agentic AI](agentic-ai/index.md) — agents, harness, tools, orchestration
- [Evals & Observability](evals-observability/index.md) — quality, tracing, monitoring
- [Glossary](glossary.md) — term definitions
