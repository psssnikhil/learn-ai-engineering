---
title: Glossary
---

# AI Engineering Glossary

Quick definitions with links to handbook modules.

| Term | Definition | Learn more |
|------|------------|------------|
| **Agent** | System that uses an LLM to plan, call tools, and act in a loop toward a goal | [M11](build/module-11-ai-agents-fundamentals/index.md) |
| **Agent harness** | Runtime wrapping the agent loop: permissions, tools, state, termination, observability | [M18](build/module-18-agent-harness-tools-runtime/index.md) |
| **Attention** | Mechanism letting models weigh relevance of input tokens | [M00](foundations/module-00-genai-foundations-from-nlp-to-transformers/index.md) |
| **Chunking** | Splitting documents into retrieval-sized pieces | [M09](build/module-09-rag-retrieval-augmented-generation/index.md) |
| **Context window** | Max tokens a model can process in one request | [M01](foundations/module-01-ai-engineering-essentials/index.md) |
| **Embedding** | Dense vector representation of text for similarity search | [M13](build/module-13-vector-databases-deep-dive/index.md) |
| **Eval / evaluation** | Measuring LLM/agent quality offline or online | [M19](production/module-19-llm-evaluation-quality/index.md) |
| **Fine-tuning** | Training a base model on domain-specific data | [M15](advanced/module-15-fine-tuning-custom-models/index.md) |
| **Golden dataset** | Curated input/output pairs for regression testing | [M19](production/module-19-llm-evaluation-quality/index.md) |
| **Guardrails** | Constraints on model outputs (format, safety, scope) | [M14](build/module-14-prompt-engineering-mastery/index.md) |
| **Hallucination** | Model generating plausible but false content | [M16](production/module-16-ai-safety-ethics/index.md) |
| **Harness** | See *Agent harness* | [M18](build/module-18-agent-harness-tools-runtime/index.md) |
| **LLMOps** | Operating LLM apps in production (deploy, monitor, version) | [M10](production/module-10-llmops-production-systems/index.md) |
| **LLM-as-judge** | Using an LLM to score another model's outputs | [M19](production/module-19-llm-evaluation-quality/index.md) |
| **MCP** | Model Context Protocol — standard for tool/resource servers | [M18](build/module-18-agent-harness-tools-runtime/index.md) |
| **Observability** | Traces, logs, metrics for debugging LLM pipelines | [M10](production/module-10-llmops-production-systems/index.md) |
| **Orchestration** | Coordinating multiple agents or steps in a workflow | [M12](build/module-12-multi-agent-systems/index.md) |
| **PEFT / LoRA** | Parameter-efficient fine-tuning methods | [M15](advanced/module-15-fine-tuning-custom-models/index.md) |
| **Prompt injection** | Attack where user input overrides system instructions | [M16](production/module-16-ai-safety-ethics/index.md) |
| **RAG** | Retrieval-Augmented Generation — fetch context then generate | [M09](build/module-09-rag-retrieval-augmented-generation/index.md) |
| **ReAct** | Reason + Act pattern for tool-using agents | [M11](build/module-11-ai-agents-fundamentals/index.md) |
| **RLHF** | Reinforcement Learning from Human Feedback | [M07](foundations/module-07-large-language-models-llms/index.md) |
| **Tool use** | LLM calling external functions/APIs | [M11](build/module-11-ai-agents-fundamentals/index.md) |
| **Trajectory eval** | Scoring full agent run (steps, tools, outcome) | [M19](production/module-19-llm-evaluation-quality/index.md) |
| **Loop engineering** | Designing inner/outer agent control cycles | [2026 · Loop Engineering](ai-engineering-2026/loop-engineering.md) |
| **Context engineering** | Curating what enters the context window | [2026 · Context Engineering](ai-engineering-2026/context-engineering.md) |
| **Skills (SKILL.md)** | Reusable agent instruction files | [2026 · Skills & Rules](ai-engineering-2026/skills-and-rules.md) |
| **Claude Code** | Agentic terminal coding harness | [2026 · Claude Code](ai-engineering-2026/claude-code.md) |
| **Transformer** | Architecture based on self-attention | [M06](foundations/module-06-transformers-attention-mechanisms/index.md) |
| **Vector database** | Store optimized for similarity search on embeddings | [M13](build/module-13-vector-databases-deep-dive/index.md) |
