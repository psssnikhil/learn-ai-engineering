---
title: Evals & Observability
---

# Evals & Observability

Production AI fails silently without **measurement**. This hub ties together evaluation, tracing, monitoring, and safety across the handbook.

## The quality loop

```mermaid
flowchart LR
  Build[Build / Prompt] --> Offline[Offline Evals]
  Offline --> Gate{Quality Gate}
  Gate -->|Pass| Deploy[Deploy]
  Gate -->|Fail| Build
  Deploy --> Online[Online Monitoring]
  Online --> Alert[Alerts & Feedback]
  Alert --> Build
```

## Learning path

| Step | Course | Focus |
|------|--------|-------|
| 1 | [Course 13 · LLM Evals](../production/module-19-llm-evaluation-quality/index.md) | Golden sets, LLM-as-judge, CI/CD gates |
| 2 | [Course 12 · LLMOps](../production/module-10-llmops-production-systems/index.md) | Observability, caching, A/B tests |
| 3 | [Course 08 · Harness L6](../build/module-18-agent-harness-tools-runtime/lessons/06-observability-in-the-harness.md) | Agent step tracing |
| 4 | [Course 14 · Safety](../production/module-16-ai-safety-ethics/index.md) | Red teaming, adversarial evals |
| 5 | [Course 06 · RAG L8](../build/module-09-rag-retrieval-augmented-generation/lessons/08-RAG-Evaluation-Metrics.md) | RAG-specific metrics |

## Eval types

| Type | When | Handbook |
|------|------|----------|
| **Unit / regression** | Every PR | [Course 13](../production/module-19-llm-evaluation-quality/index.md) |
| **RAG retrieval** | Chunking / index changes | [Course 06 · RAG eval metrics](../build/module-09-rag-retrieval-augmented-generation/lessons/08-RAG-Evaluation-Metrics.md) |
| **Agent trajectory** | Tool-use agents | [Course 13 · Agent trajectory evals](../production/module-19-llm-evaluation-quality/lessons/04-agent-trajectory-evals.md) |
| **Safety / red team** | Pre-release | [Course 14 · Red teaming](../production/module-16-ai-safety-ethics/lessons/08-lesson-08.md) |
| **Online / drift** | Production | [Course 13 · Production monitoring](../production/module-19-llm-evaluation-quality/lessons/06-production-monitoring-and-alerts.md) |

## Observability pillars

| Pillar | What to capture | Tools (OSS) |
|--------|-----------------|-------------|
| **Traces** | Full request path, agent steps | Langfuse, OpenTelemetry, LangSmith |
| **Logs** | Prompts, tool I/O (redacted) | structlog, CloudWatch |
| **Metrics** | Latency, tokens, cost, error rate | Prometheus, Grafana |
| **Feedback** | Thumbs, edits, escalations | Custom + [Course 13 monitoring](../production/module-19-llm-evaluation-quality/lessons/06-production-monitoring-and-alerts.md) |

## OSS hubs

- [Awesome Agent Evals](https://github.com/benchflow-ai/awesome-evals) — eval frameworks catalog
- [DeepEval](https://github.com/confident-ai/deepeval) — pytest-style LLM tests
- [Promptfoo](https://github.com/promptfoo/promptfoo) — red team + CI evals
- [RAGAS](https://github.com/explodinggradients/ragas) — RAG metrics

## Related

- [Agentic AI](../agentic-ai/index.md) — what you're measuring
- [Essential Papers](../resources/essential-papers.md) — HELM, MT-Bench, etc.
- [Roadmap](../roadmap.md) — upcoming observability labs
