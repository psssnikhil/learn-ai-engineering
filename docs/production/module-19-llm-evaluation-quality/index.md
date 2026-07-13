---
title: LLM Evaluation & Quality Engineering
module_id: module-19
phase: Production
---
Build systematic evaluation pipelines that catch regressions, gate deployments, and monitor quality in production.

| | |
|---|---|
| **Module ID** | `module-19` |
| **Phase** | Production |
| **Lessons** | 6 |
| **Est. hours** | ~5h |

## What You'll Build

By the end of this module you'll have a complete AI quality pipeline: a golden dataset you can version and grow, an LLM-as-judge system calibrated against human labels, agent trajectory evaluations that verify tool sequences, CI/CD eval gates that block regressions before merge, and production monitoring that detects drift within hours rather than days.

## Lessons

| # | Lesson | Duration | Level |
|---|--------|----------|-------|
| 1 | [Why LLM Evals Matter](lessons/01-why-llm-evals-matter.md) | 45 min | intermediate |
| 2 | [Golden Datasets & Benchmarks](lessons/02-golden-datasets-and-benchmarks.md) | 50 min | intermediate |
| 3 | [LLM-as-Judge](lessons/03-llm-as-judge.md) | 50 min | advanced |
| 4 | [Agent Trajectory Evals](lessons/04-agent-trajectory-evals.md) | 50 min | advanced |
| 5 | [CI/CD for AI Quality](lessons/05-ci-cd-for-ai-quality.md) | 50 min | advanced |
| 6 | [Production Monitoring & Alerts](lessons/06-production-monitoring-and-alerts.md) | 50 min | advanced |

## Lesson Summaries

| # | What you learn |
|---|---------------|
| 1 | Offline vs online evaluation, regression testing with baselines, quality gates at PR / staging / canary, eval tooling overview |
| 2 | Anatomy of a golden test case, collecting from production logs and edge cases, RAGAS metrics, dataset versioning |
| 3 | Rubric design for LLM judges, seven judge biases and how to mitigate them, calibration against human labels, when not to use LLM-as-judge |
| 4 | Step-level vs outcome-level evaluation, tool call correctness, forbidden tool sequences, composite trajectory scoring |
| 5 | Change detection to run evals only on AI-related PRs, GitHub Actions integration, canary eval gates, shadow traffic |
| 6 | Four monitoring pillars (quality, latency, cost, reliability), statistical drift detection, stratified sampling, user feedback → golden set loop |

**Start here:** [Why LLM Evals Matter](lessons/01-why-llm-evals-matter.md)
