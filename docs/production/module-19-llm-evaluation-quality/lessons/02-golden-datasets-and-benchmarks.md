---
title: Golden Datasets & Benchmarks
description: >-
  Build curated golden test sets, apply RAGAS metrics, and design domain-specific
  benchmarks that predict production quality
duration: 40 min
difficulty: intermediate
has_code: false
module: module-19
---
# Golden Datasets & Benchmarks

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Build and maintain golden datasets | 40 min | Intermediate |
| Apply RAGAS metrics for RAG evaluation | | |
| Design domain-specific benchmarks | | |
| Avoid common dataset pitfalls | | |

---

## What Is a Golden Dataset?

A **golden dataset** (also called a golden set or eval set) is a curated collection of input-output pairs that represent the quality bar your application must meet. It is the foundation of every offline eval pipeline.

Unlike a training dataset, a golden set is small, hand-validated, and stable. Think 50–500 cases, not 50,000. Each case should be something you would be embarrassed to get wrong in production.

### Anatomy of a Golden Test Case

```python
golden_case = {
    "id": "support-refund-042",
    "input": "I was charged twice for order #8821. How do I get a refund?",
    "context": {  # Optional: for RAG/agent evals
        "retrieved_docs": ["refund_policy_v3.md", "billing_faq.md"],
        "user_tier": "premium",
    },
    "expected": {
        "must_mention": ["refund", "billing", "48 hours"],
        "must_not_mention": ["I cannot help", "contact a human"],
        "format": "json",
        "schema": {"action": str, "eta_hours": int},
    },
    "metadata": {
        "category": "billing",
        "difficulty": "medium",
        "source": "production_incident_2025-11-03",
        "human_validated": True,
    },
}
```

Every field serves a purpose:

| Field | Purpose |
|-------|---------|
| `id` | Stable identifier for tracking regressions across runs |
| `input` | The user query or task |
| `context` | Retrieved docs, user state, tool availability |
| `expected` | Ground truth, constraints, or rubric criteria |
| `metadata` | Category, difficulty, provenance for slicing results |

---

## Building a Golden Set from Scratch

### Step 1: Seed from Production

Start with real user queries, not synthetic ones. Pull from logs, support tickets, and search analytics.

```python
def seed_golden_set_from_logs(logs: list[dict], n: int = 100) -> list[dict]:
    """Sample diverse production queries to seed the golden set."""
    from collections import Counter

    # Stratify by intent category for coverage
    by_category = {}
    for log in logs:
        cat = log.get("intent_category", "uncategorized")
        by_category.setdefault(cat, []).append(log)

    cases = []
    per_category = max(1, n // len(by_category))
    for category, entries in by_category.items():
        # Take most frequent + random sample for diversity
        sorted_entries = sorted(entries, key=lambda x: x["frequency"], reverse=True)
        cases.extend(sorted_entries[:per_category])

    return [{"id": f"seed-{i}", "input": c["query"], "metadata": {"category": c.get("intent_category")}} 
            for i, c in enumerate(cases[:n])]
```

### Step 2: Add Edge Cases and Failures

Production logs skew toward happy paths. Deliberately add:

- **Ambiguous queries** — "fix my account" (which account? what is broken?)
- **Adversarial inputs** — prompt injections, jailbreak attempts, PII in queries
- **Multi-turn context** — follow-up questions that depend on prior messages
- **Known failure modes** — every production incident becomes a test case
- **Boundary conditions** — empty input, max-length input, non-English queries

### Step 3: Human Validation

Every golden case needs a human-reviewed expected output or rubric. LLM-generated labels are a starting point, not the final answer.

```python
# Validation workflow
validation_queue = [
    {"case_id": "support-refund-042", "status": "pending_review", "reviewer": None},
    {"case_id": "rag-hallucination-017", "status": "pending_review", "reviewer": None},
]

# Rules for validation:
# 1. Two reviewers must agree on expected output (for subjective cases)
# 2. Disagreements go to a tiebreaker or get a rubric instead of exact match
# 3. Re-validate quarterly — product changes make old expectations stale
```

### Step 4: Version and Maintain

Golden sets decay. Product changes, new features, and model updates make old test cases irrelevant.

| Maintenance Action | Frequency |
|--------------------|-----------|
| Add cases from production failures | Continuous |
| Review stale cases (product changed) | Monthly |
| Full re-validation by domain experts | Quarterly |
| Prune redundant or trivial cases | Quarterly |
| Version tag on every change | Every edit |

Store golden sets in version control (JSON, YAML, or a dataset tool). Tag releases: `golden-set-v3.2`.

---

## RAGAS: Metrics for RAG Applications

[RAGAS](https://github.com/explodinggradients/ragas) (Retrieval-Augmented Generation Assessment) provides standardized metrics for evaluating RAG pipelines. It is integrated into [DeepEval](https://github.com/confident-ai/deepeval) and widely used in production.

### Core RAGAS Metrics

| Metric | What It Measures | When It Matters |
|--------|------------------|-----------------|
| **Faithfulness** | Is the answer grounded in retrieved context? | Always — the #1 RAG failure mode |
| **Answer Relevancy** | Does the answer address the question? | Always |
| **Context Precision** | Are retrieved chunks relevant to the query? | Retrieval tuning |
| **Context Recall** | Did retrieval find all necessary information? | Missing information complaints |
| **Answer Correctness** | Factual alignment with ground truth | When you have reference answers |

```python
# RAGAS-style evaluation with DeepEval
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

def evaluate_rag_response(query: str, context: list[str], answer: str) -> dict:
    test_case = LLMTestCase(
        input=query,
        actual_output=answer,
        retrieval_context=context,
    )

    faithfulness = FaithfulnessMetric(threshold=0.8)
    relevancy = AnswerRelevancyMetric(threshold=0.7)

    faithfulness.measure(test_case)
    relevancy.measure(test_case)

    return {
        "faithfulness": faithfulness.score,
        "answer_relevancy": relevancy.score,
        "faithfulness_passed": faithfulness.is_successful(),
        "relevancy_passed": relevancy.is_successful(),
    }
```

### Interpreting RAGAS Scores

- **Faithfulness < 0.7** — your model is hallucinating beyond retrieved context. Fix: stronger grounding instructions, citation requirements, or better retrieval.
- **Context Precision < 0.6** — retrieval is returning noise. Fix: reranking, chunk size tuning, metadata filtering.
- **Context Recall < 0.6** — retrieval is missing relevant docs. Fix: hybrid search, query expansion, more chunks.
- **Answer Relevancy < 0.7** — the model is answering the wrong question. Fix: prompt engineering, query rewriting.

RAGAS metrics use LLM-as-judge under the hood. They are fast to set up but should be calibrated against human labels (covered in Lesson 3).

---

## Domain-Specific Benchmarks

Generic benchmarks (MMLU, HumanEval) measure model capability. **Domain benchmarks** measure whether your *application* works for your users.

### Designing a Domain Benchmark

```python
# Customer support benchmark structure
support_benchmark = {
    "name": "acme-support-v1",
    "categories": {
        "billing": {"cases": 25, "min_pass_rate": 0.95},
        "technical": {"cases": 30, "min_pass_rate": 0.90},
        "account_management": {"cases": 20, "min_pass_rate": 0.95},
        "edge_cases": {"cases": 15, "min_pass_rate": 0.80},
        "safety": {"cases": 10, "min_pass_rate": 1.00},  # Zero tolerance
    },
    "metrics": {
        "resolution_rate": "Did the answer solve the user's problem?",
        "policy_compliance": "Did the answer follow company policies?",
        "tone": "Was the response professional and empathetic?",
        "escalation_correctness": "Did it escalate when it should have?",
    },
}
```

### Benchmark Design Principles

1. **Represent real user intent distribution** — if 40% of queries are billing, 40% of your benchmark should be billing
2. **Include difficulty tiers** — easy (FAQ), medium (multi-step), hard (ambiguous or adversarial)
3. **Set per-category thresholds** — safety at 100%, edge cases at 80%, everything else in between
4. **Track over time** — plot benchmark scores per release to see trends, not just pass/fail
5. **Share with stakeholders** — a benchmark score is your quality SLA with the business

### Comparing with Promptfoo

[Promptfoo](https://github.com/promptfoo/promptfoo) excels at running your golden set against multiple prompt/model configurations side by side:

Run with `npx promptfoo eval` in CI. Results are diffable across prompt versions.

---

## Common Pitfalls

| Pitfall | Why It Hurts | Fix |
|---------|--------------|-----|
| **Too few cases** | High variance, false confidence | Minimum 50 per category |
| **All easy cases** | Benchmark says 99%, users say 60% | Add hard and adversarial cases |
| **Stale expectations** | Product changed, tests still pass | Quarterly re-validation |
| **LLM-only labels** | Garbage in, garbage out | Human validation on every case |
| **No provenance** | Cannot trace why a case exists | Tag source and date in metadata |
| **Single metric** | Optimizing faithfulness while relevancy drops | Multi-metric dashboards |

---

## Curated Resources

The [benchflow-ai/awesome-evals](https://github.com/benchflow-ai/awesome-evals) repository maintains a comprehensive list of eval frameworks, datasets, papers, and best practices. Use it to discover domain-specific benchmarks and stay current as the field evolves.

---

## Key Takeaways

- Golden datasets are small, human-validated, and versioned — they are your quality contract
- Seed from production logs, add edge cases, validate with humans, maintain quarterly
- **RAGAS metrics** (faithfulness, relevancy, context precision/recall) are the standard for RAG eval
- **Domain benchmarks** with per-category thresholds matter more than generic model benchmarks
- Use DeepEval for pytest-style RAG evals and Promptfoo for prompt/model comparison in CI
- Every production failure should become a new golden test case

---

## Next Lesson

**Lesson 3: LLM-as-Judge** — Learn to design rubrics, understand judge bias, calibrate against human labels, and know when automated judging is the wrong tool.
