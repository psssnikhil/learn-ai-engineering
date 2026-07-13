---
title: RAG Evaluation Metrics
description: >-
  Measure retrieval and generation quality with Recall, MRR, faithfulness, and
  answer relevance metrics
duration: 55 min
difficulty: advanced
has_code: true
module: module-09
youtube: 'https://www.youtube.com/watch?v=1c9iyoVIwDs'
---

## Prerequisites

- [Lesson 05 — Building a Basic RAG System](05-Building-a-Basic-RAG-System.md): end-to-end RAG pipeline to evaluate
- [Lesson 06 — Advanced RAG Techniques](06-Advanced-RAG-Techniques.md): understanding the retrieval and generation stages separately
- Familiarity with Python dictionaries and list comprehensions
- Basic statistics (mean, percentile) — no advanced math required

---

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand why RAG has two distinct failure modes requiring separate metrics | 55 min | Advanced |
| Apply Recall@K and MRR to measure retrieval quality | | |
| Compute NDCG when relevance has multiple grades | | |
| Measure faithfulness and answer relevance with LLM-as-judge | | |
| Build a full end-to-end eval pipeline with the RAGAS framework | | |
| Design an eval dataset from real user queries | | |

---

## Intuition First

Imagine a researcher assistant that reliably finds the right books from the library (retrieval), but then summarizes them inaccurately (generation). Or one that summarizes faithfully but finds the wrong books. You'd diagnose and fix these differently.

RAG evaluation has the same structure. Before you optimize anything, you need to answer two questions:

1. **Retrieval question:** Did the system surface the documents that *contain* the right answer?
2. **Generation question:** Given the right documents, did the LLM answer faithfully and relevantly?

Most teams skip this diagnostic and throw more compute at the wrong stage. A retrieval Recall@5 of 0.90 means the LLM had the information 90% of the time — if answers are still wrong, the problem is generation, not retrieval. Conversely, a faithfulness score of 0.95 with Recall@5 of 0.40 means the LLM is faithful but starved of the right context.

**Always evaluate retrieval and generation separately first.**

---

## Building a Labeled Eval Dataset

A RAG eval set is a collection of `(question, relevant_doc_ids, reference_answer)` triples. Quality of your eval set determines the reliability of your metrics.

### Sourcing Questions

| Source | Pros | Cons |
|--------|------|------|
| Real user queries (production logs) | Reflects actual distribution | May have PII; need filtering |
| Support tickets | High business value | May be ambiguous |
| Synthetic from LLM | Fast to generate | Distribution drift; validates itself |
| Human annotators | Highest quality | Expensive, slow |

**Best practice:** Generate synthetic questions with an LLM, then have subject-matter experts review and correct relevance labels. This gives you speed + domain accuracy.

```python
def generate_eval_questions(
    document: str,
    doc_id: str,
    client,
    n: int = 3,
) -> list[dict]:
    """Generate n (question, answer, doc_id) triples from one document."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    f"Generate {n} diverse questions that can be answered from the "
                    "document below. Also write the correct short answer for each. "
                    "Format as JSON array: "
                    '[{"question": "...", "answer": "..."}]'
                ),
            },
            {"role": "user", "content": document},
        ],
        response_format={"type": "json_object"},
    )
    import json
    items = json.loads(response.choices[0].message.content).get("items", [])
    return [
        {
            "question": item["question"],
            "reference_answer": item["answer"],
            "relevant_doc_ids": [doc_id],
        }
        for item in items
    ]
```

### Eval Set Size Guidelines

| Corpus size | Minimum eval set | For confident metrics |
|-------------|------------------|-----------------------|
| < 1,000 docs | 50 questions | 150 questions |
| 1,000–10,000 docs | 100 questions | 300 questions |
| > 10,000 docs | 200 questions | 500 questions |

Stratify across document types and query styles (factual, procedural, comparative).

---

## Retrieval Metrics

### Recall@K

**What it measures:** Of all relevant documents for a query, what fraction did the system retrieve in its top K results?

\[
\text{Recall@K} = \frac{|\text{Top-K retrieved} \cap \text{Relevant}|}{|\text{Relevant}|}
\]

```python
def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    Compute Recall@K for a single query.
    
    Args:
        retrieved_ids: Ordered list of retrieved document IDs, best-first.
        relevant_ids: Ground-truth relevant document IDs.
        k: Cutoff position.
    
    Returns:
        Recall score in [0, 1].
    """
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(top_k & relevant) / len(relevant)


# Example
retrieved = ["doc_a", "doc_b", "doc_c", "doc_d", "doc_e"]
relevant = ["doc_b", "doc_f"]

print(recall_at_k(retrieved, relevant, k=5))  # 0.5 — found 1 of 2 relevant docs
print(recall_at_k(retrieved, relevant, k=10)) # still 0.5 — doc_f not in top 10
```

**Worked example with numbers:**

Your system retrieves K=5 results for 4 queries:

| Query | Relevant docs | Found in top-5 | Recall@5 |
|-------|--------------|----------------|----------|
| "reset password" | [doc_01, doc_03] | [doc_01] | 0.50 |
| "API rate limit" | [doc_07] | [doc_07] | 1.00 |
| "billing FAQ" | [doc_12, doc_15, doc_18] | [doc_12, doc_18] | 0.67 |
| "SSO setup" | [doc_22] | [] | 0.00 |

Mean Recall@5 = (0.50 + 1.00 + 0.67 + 0.00) / 4 = **0.54**

This is below the minimum viable threshold. You know retrieval is the bottleneck — fix chunking or embedding before tuning the LLM.

### Mean Reciprocal Rank (MRR)

**What it measures:** On average, how high up in the ranked list does the *first* relevant document appear?

\[
\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}
\]

Where `rank_i` is the position (1-indexed) of the first relevant document for query `i`. If no relevant document is found, that query contributes 0.

```python
def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Reciprocal rank for a single query — rewards first-position hits."""
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0

def mean_reciprocal_rank(results: list[tuple[list[str], list[str]]]) -> float:
    scores = [reciprocal_rank(retrieved, relevant) for retrieved, relevant in results]
    return sum(scores) / len(scores) if scores else 0.0
```

**MRR vs Recall@K:** MRR cares about *where* the first relevant document appears; Recall@K cares about *how many* relevant documents appear. Use MRR when the user will read the top result; use Recall@K when context is built from multiple retrieved chunks.

### NDCG — Normalized Discounted Cumulative Gain

When some documents are *more* relevant than others (e.g., a 3-point scale: highly relevant=3, partially relevant=1, irrelevant=0), NDCG captures graded relevance:

\[
\text{DCG@K} = \sum_{i=1}^{K} \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}
\]

\[
\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}
\]

Where IDCG@K is the ideal DCG (perfect ranking). NDCG=1.0 means your ranking is perfect; NDCG=0.0 means completely wrong.

```python
import math

def dcg_at_k(relevances: list[int], k: int) -> float:
    """
    Compute DCG@K given a list of relevance grades in retrieved order.
    relevances[i] = grade of the (i+1)th retrieved document.
    """
    return sum(
        (2 ** rel - 1) / math.log2(rank + 2)
        for rank, rel in enumerate(relevances[:k])
    )

def ndcg_at_k(relevances: list[int], k: int) -> float:
    ideal = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(relevances, k) / idcg

# Example: 5 retrieved docs with grades [3, 0, 1, 3, 0]
grades = [3, 0, 1, 3, 0]  # highly relevant, irrelevant, partial, highly, irrelevant
print(f"NDCG@5: {ndcg_at_k(grades, k=5):.3f}")  # ~0.78
```

### Metric Targets (Rule of Thumb)

| Metric | Minimum viable | Good | Excellent |
|--------|---------------|------|-----------|
| Recall@5 | 0.60 | 0.80 | 0.90+ |
| Recall@10 | 0.75 | 0.90 | 0.95+ |
| MRR | 0.50 | 0.70 | 0.85+ |
| NDCG@5 | 0.60 | 0.78 | 0.90+ |

These are starting points. Domain matters — a medical knowledge base may require Recall@10 > 0.95 before deployment.

---

## Generation Metrics

### Faithfulness (Groundedness)

**What it measures:** Are all claims in the generated answer supported by the retrieved context? A faithful answer never introduces facts not present in the chunks — it may be incomplete, but it doesn't hallucinate.

```python
def check_faithfulness(
    answer: str,
    context: str,
    client,
    model: str = "gpt-4o-mini",
) -> float:
    """
    LLM-as-judge faithfulness check.
    Returns a score from 0.0 to 1.0.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a factual auditor. Given a context and an answer, "
                    "identify each claim in the answer and check if it is "
                    "directly supported by the context. "
                    "Output a JSON object: "
                    '{"supported": <int>, "total": <int>, "faithfulness": <float>}'
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nAnswer:\n{answer}",
            },
        ],
        response_format={"type": "json_object"},
    )
    import json
    result = json.loads(response.choices[0].message.content)
    return result.get("faithfulness", 0.0)
```

### Answer Relevance

**What it measures:** Does the answer address the question, regardless of whether it was grounded? An answer can be faithful to a wrong context but irrelevant to the question.

```python
def check_relevance(question: str, answer: str, client) -> float:
    """
    Score answer relevance from 1 (not relevant) to 5 (fully addresses question).
    Returns normalized score [0, 1].
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Rate how well the answer addresses the question. "
                    "Score 1-5 where: 1=completely off-topic, 3=partially relevant, "
                    "5=fully answers the question. Reply with a number only."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\nAnswer: {answer}",
            },
        ],
    )
    score = float(response.choices[0].message.content.strip())
    return (score - 1) / 4  # normalize to [0, 1]
```

### Context Precision

**What it measures:** Of the retrieved chunks, what fraction were actually used/relevant to the answer? Low context precision means you retrieved too much noise — prompting you to improve retrieval filtering or add compression.

```python
def check_context_precision(
    question: str,
    chunks: list[str],
    answer: str,
    client,
) -> float:
    """Fraction of retrieved chunks that contributed to the answer."""
    relevant_count = 0
    for chunk in chunks:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n"
                        f"Chunk: {chunk}\n"
                        f"Answer: {answer}\n\n"
                        "Did this chunk contribute to the answer? Reply YES or NO."
                    ),
                }
            ],
        )
        if response.choices[0].message.content.strip().upper().startswith("YES"):
            relevant_count += 1
    return relevant_count / len(chunks) if chunks else 0.0
```

---

## End-to-End Eval Loop

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class EvalResult:
    question: str
    recall_at_5: float
    faithfulness: float
    relevance: float
    context_precision: float


def evaluate_rag(
    rag_system,
    eval_set: list[dict],
    client,
    k: int = 5,
) -> dict[str, Any]:
    results: list[EvalResult] = []

    for item in eval_set:
        question = item["question"]
        relevant_ids = item["relevant_doc_ids"]

        # Retrieval metrics
        retrieved_ids, chunks = rag_system.retrieve_with_ids(question, k=k)
        r_at_k = recall_at_k(retrieved_ids, relevant_ids, k=k)

        # Generation metrics
        answer, context_str = rag_system.answer_with_context(question)
        faith = check_faithfulness(answer, context_str, client)
        rel = check_relevance(question, answer, client)
        cp = check_context_precision(question, chunks, answer, client)

        results.append(EvalResult(
            question=question,
            recall_at_5=r_at_k,
            faithfulness=faith,
            relevance=rel,
            context_precision=cp,
        ))

    n = len(results)
    return {
        "n_questions": n,
        "recall_at_5": sum(r.recall_at_5 for r in results) / n,
        "faithfulness": sum(r.faithfulness for r in results) / n,
        "relevance": sum(r.relevance for r in results) / n,
        "context_precision": sum(r.context_precision for r in results) / n,
        "failures": [r for r in results if r.recall_at_5 < 0.5 or r.faithfulness < 0.7],
    }
```

!!! warning "LLM-as-judge calibration"
    LLM judges correlate well with human judgment for aggregate trends (~0.7–0.8 Pearson correlation) but have systematic biases — they tend to score verbose answers higher and favor their own output patterns. Validate your judge against 50+ human labels before treating scores as ground truth.

---

## RAGAS: Automated RAG Evaluation Framework

[RAGAS](https://docs.ragas.io/) implements all the metrics above (faithfulness, answer relevance, context precision, context recall) in a single framework, with a clean `evaluate()` API.

```python
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

# Build RAGAS-format dataset
data = {
    "question": [item["question"] for item in eval_set],
    "answer": [rag_system.answer(q) for q in questions],
    "contexts": [rag_system.retrieve_texts(q) for q in questions],
    "ground_truth": [item["reference_answer"] for item in eval_set],
}
dataset = Dataset.from_dict(data)

result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
)
print(result)
# {'faithfulness': 0.83, 'answer_relevancy': 0.91, ...}
```

RAGAS uses an LLM internally to judge faithfulness and relevance — the same pattern as the custom code above. It also supports non-OpenAI models for the judge (Anthropic, open-source) and integrates with LangSmith for trace storage.

---

## Interpreting Your Metrics

| Pattern | What it means | Fix |
|---------|--------------|-----|
| Low Recall@5, high faithfulness | LLM is faithful but context is wrong | Improve chunking, embedding, or retrieval method |
| High Recall@5, low faithfulness | Right context retrieved, LLM ignores it | Improve system prompt; use stronger model |
| Low context precision | Too many irrelevant chunks retrieved | Tighten K; add re-ranking or compression |
| Low relevance, high faithfulness | Answer is grounded but doesn't address question | Fix retrieval; improve query understanding |

---

## Common Misconceptions

**"End-to-end answer accuracy is enough."** A correct final answer can come from a retrieved document that wasn't the intended ground-truth document. End-to-end accuracy masks retrieval quality — always measure retrieval separately.

**"Faithfulness and accuracy are the same."** Faithfulness means the answer is grounded in the *retrieved* context. Accuracy means the answer is factually *correct*. If the retrieved context is wrong, an answer can be 100% faithful and completely inaccurate.

**"More eval questions is always better."** Quality matters more than quantity. 50 carefully labeled, diverse questions beat 500 synthetic questions generated from a single prompt without review.

**"RAGAS scores are universal benchmarks."** RAGAS is a relative improvement tool for your system. Scores vary by domain, question difficulty, and judge model. Never compare RAGAS scores across systems with different domains.

---

## Production Tips

- **Run evals on every index change:** Treat your eval set as a regression test. A chunk size change or embedding model upgrade can silently degrade Recall@5.
- **Track metrics over time:** Plot weekly trends. A slow decline in faithfulness often signals document drift — the corpus content is changing faster than the system adapts.
- **Segment by query type:** Overall Recall@5 = 0.78 might hide that procedural queries (0.90) mask failed comparative queries (0.52). Segment metrics by question category.
- **Keep eval set fresh:** Periodically add new questions from recent user sessions. Distribution shift is real — yesterday's user queries are the best predictor of tomorrow's.
- **Human spot-check 10% of LLM judgments:** Even with good calibration, LLM judges make systematic errors. Monthly spot-checks keep your metrics honest.

---

## Key Takeaways

- RAG has two failure modes — **retrieval failure** and **generation failure** — fix the right one by measuring separately
- **Recall@K** measures fraction of relevant docs found; **MRR** rewards high rank of the first relevant doc; **NDCG** captures graded relevance
- **Faithfulness** checks if the answer is grounded in context; **answer relevance** checks if it addresses the question; **context precision** checks if retrieved chunks were useful
- Build an eval set from real user queries (50–200) with labeled relevant doc IDs before shipping any system change
- Use **RAGAS** to automate LLM-as-judge metrics; validate the judge against human labels before trusting the scores
- Segment metrics by query type and track trends over time — a global average masks category-level regressions

---

## Related Papers

| Paper | Year | Key contribution |
|-------|------|-----------------|
| [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) | 2023 | Faithfulness, answer relevancy, and context metrics for RAG |
| [Evaluating Correctness and Faithfulness of Instruction-Following Models for Question Answering](https://arxiv.org/abs/2307.16877) | 2023 | LLM-as-judge calibration against human judgments |
| [ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems](https://arxiv.org/abs/2311.09476) | 2023 | Statistical testing for RAG pipeline comparisons |
| [RGB: A Comprehensive Benchmark for RAG](https://arxiv.org/abs/2309.01431) | 2023 | Multi-dimension RAG evaluation across LLMs |

---

## Next Lesson

**[Lesson 09 — Agentic RAG](09-Agentic-RAG.md):** Move from fixed retrieval pipelines to AI agents that decide when and what to retrieve, iterate on their own answers, and fall back to web search when the knowledge base falls short.
