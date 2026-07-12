---
title: Related Papers — 2026 Skills
description: Famous papers behind Claude Code, skills, loops, and context engineering
---

# Related Papers — 2026 Skills

Papers and technical reports behind the 2026 agent engineering stack.

## Reasoning models & test-time compute

| Paper / report | Why it matters | Link |
|----------------|----------------|------|
| **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** | Open reasoning model — RL-trained chain-of-thought | [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) |
| **Let's Verify Step by Step (Process Reward Models)** | Reward each reasoning step, not just the final answer | [arXiv:2305.20050](https://arxiv.org/abs/2305.20050) |
| **Scaling LLM Test-Time Compute Optimally** | How much inference compute to spend per task | [arXiv:2408.03314](https://arxiv.org/abs/2408.03314) |
| **OpenAI o1 System Card** | Production reasoning model design choices | [OpenAI PDF](https://openai.com/index/openai-o1-system-card/) |

## Context & memory engineering

| Paper | Why it matters | Link |
|-------|----------------|------|
| **Lost in the Middle: How Language Models Use Long Contexts** | Why context window placement matters — center gets ignored | [arXiv:2307.03172](https://arxiv.org/abs/2307.03172) |
| **MemGPT: Towards LLMs as Operating Systems** | Virtual memory paging for long conversations | [arXiv:2310.08560](https://arxiv.org/abs/2310.08560) |
| **Large Language Models as Tool Makers** | Agents that create reusable tools/skills | [arXiv:2305.17126](https://arxiv.org/abs/2305.17126) |

## Skills, programs & automated optimization

| Paper | Why it matters | Link |
|-------|----------------|------|
| **Voyager: An Open-Ended Embodied Agent with Large Language Models** | Growing skill library from experience | [arXiv:2305.16291](https://arxiv.org/abs/2305.16291) |
| **DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines** | Optimize prompts/programs automatically | [arXiv:2310.03714](https://arxiv.org/abs/2310.03714) |
| **TextGrad: Automatic Differentiation via Text** | Gradient-free optimization of agent pipelines | [arXiv:2406.07496](https://arxiv.org/abs/2406.07496) |

## Coding agents

| Paper | Why it matters | Link |
|-------|----------------|------|
| **SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering** | ACI design — foundation for coding agents like Claude Code | [arXiv:2405.15703](https://arxiv.org/abs/2405.15703) |
| **SWE-bench: Can Language Models Resolve Real-World GitHub Issues?** | The benchmark coding agents are measured against | [arXiv:2310.06770](https://arxiv.org/abs/2310.06770) |
| **Devin / Cognition technical blog** | End-to-end software engineering agent architecture | [cognition.ai](https://www.cognition.ai/blog/introducing-devin) |

## Harness & safety

| Paper | Why it matters | Link |
|-------|----------------|------|
| **Constitutional AI: Harmlessness from AI Feedback** | Self-critique loops for safer agents | [arXiv:2212.08073](https://arxiv.org/abs/2212.08073) |
| **AgentPoison: Red-teaming LLM Agents via Poisoning Memory** | Why memory and tool outputs need sanitization | [arXiv:2407.12784](https://arxiv.org/abs/2407.12784) |

## Protocols & standards (not papers, but essential)

| Resource | What it is | Link |
|----------|------------|------|
| **Model Context Protocol (MCP)** | Standard for tool/resource servers | [modelcontextprotocol.io](https://modelcontextprotocol.io/) |
| **OpenTelemetry GenAI Semantic Conventions** | Standard spans for LLM/agent tracing | [OTel docs](https://opentelemetry.io/docs/specs/semconv/gen-ai/) |

## Also see

- [Agent Engineering · Related Papers](../agent-engineering/related-papers.md)
- [Essential Papers (handbook)](../resources/essential-papers.md)
- [M07 L11 · Reasoning Models](../foundations/module-07-large-language-models-llms/lessons/11-reasoning-models-and-test-time-compute.md)
