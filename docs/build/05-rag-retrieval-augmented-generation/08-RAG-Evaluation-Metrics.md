---
title: RAG Evaluation Metrics
description: >-
  Measure retrieval and generation quality with Recall, MRR, faithfulness, and
  answer relevance metrics
duration: 40 min
difficulty: advanced
has_code: false
youtube: 'https://www.youtube.com/watch?v=1c9iyoVIwDs'
---
## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Evaluate retrieval separately from generation | 40 min | Advanced |
| Apply Recall@K, MRR, and NDCG | | |
| Measure answer faithfulness and relevance | | |
| Build a repeatable eval pipeline | | |

---

## Why You Must Evaluate RAG

RAG has two failure modes — and fixing the wrong one wastes weeks:

1. **Retrieval failure** — relevant chunks never reach the LLM
2. **Generation failure** — good context, but the LLM ignores or misuses it

```
Eval question: "Did we retrieve the right docs?"
Eval question: "Did the LLM answer faithfully from those docs?"
```

> **Tip:** Always evaluate retrieval and generation separately before evaluating end-to-end answer quality.

---

## Building an Eval Dataset

Create 50–200 `(question, expected_answer, relevant_doc_ids)` triples from real user queries.

```python
eval_set = [
    {
        "question": "How do I reset my password?",
        "relevant_doc_ids": ["doc_auth_01", "doc_auth_03"],
        "reference_answer": "Go to Settings → Security → Reset Password.",
    },
    {
        "question": "What is the API rate limit?",
        "relevant_doc_ids": ["doc_api_limits"],
        "reference_answer": "100 requests per minute on the free tier.",
    },
]
```

---

## Retrieval Metrics

### Recall@K

Fraction of relevant documents found in the top-K results.

```python
def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    if not relevant:
        return 0.0
    return len(top_k & relevant) / len(relevant)

# Example
retrieved = ["doc_a", "doc_b", "doc_c", "doc_d", "doc_e"]
relevant = ["doc_b", "doc_f"]

print(recall_at_k(retrieved, relevant, k=5))  # 0.5 — found 1 of 2
```

### Mean Reciprocal Rank (MRR)

Rewards placing the first relevant document higher in the ranking.

```python
def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    relevant = set(relevant_ids)
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0

def mean_reciprocal_rank(results: list[tuple[list[str], list[str]]]) -> float:
    scores = [reciprocal_rank(r, rel) for r, rel in results]
    return sum(scores) / len(scores)
```

### Metric targets (rule of thumb)

| Metric | Minimum viable | Good | Excellent |
|--------|---------------|------|-----------|
| Recall@5 | 0.60 | 0.80 | 0.90+ |
| Recall@10 | 0.75 | 0.90 | 0.95+ |
| MRR | 0.50 | 0.70 | 0.85+ |

---

## Generation Metrics

### Faithfulness (groundedness)

Does the answer stick to the retrieved context?

```python
def check_faithfulness(answer: str, context: str, client) -> bool:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Context:
{context}

"
                    f"Answer:
{answer}

"
                    "Is every claim in the answer supported by the context? "
                    "Reply YES or NO only."
                ),
            }
        ],
    )
    return response.choices[0].message.content.strip().upper().startswith("YES")
```

### Answer Relevance

Does the answer address the question (regardless of context)?

```python
def check_relevance(question: str, answer: str, client) -> float:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Question: {question}
Answer: {answer}

"
                    "Rate relevance from 1 (not relevant) to 5 (fully relevant). "
                    "Reply with a number only."
                ),
            }
        ],
    )
    return float(response.choices[0].message.content.strip())
```

---

## End-to-End Eval Loop

```python
def evaluate_rag(rag_system, eval_set: list[dict]) -> dict:
    retrieval_scores = []
    faithfulness_scores = []

    for item in eval_set:
        retrieved_ids = rag_system.retrieve_ids(item["question"], k=10)
        recall = recall_at_k(retrieved_ids, item["relevant_doc_ids"], k=5)
        retrieval_scores.append(recall)

        answer, context = rag_system.answer_with_context(item["question"])
        faithfulness_scores.append(check_faithfulness(answer, context, client))

    return {
        "recall_at_5": sum(retrieval_scores) / len(retrieval_scores),
        "faithfulness_rate": sum(faithfulness_scores) / len(faithfulness_scores),
    }
```

> **Warning:** LLM-as-judge metrics are useful for iteration but should be validated against human labels before production sign-off.

---

## Tools for RAG Evaluation

| Tool | Best for |
|------|----------|
| [RAGAS](https://docs.ragas.io/) | Automated faithfulness, relevance, context precision |
| [LangSmith](https://docs.smith.langchain.com/) | Tracing + eval datasets |
| [Phoenix (Arize)](https://docs.arize.com/phoenix) | Retrieval visualization |
| Custom scripts | Full control, no vendor lock-in |

---

## Recommended Videos

- [RAG Evaluation Explained](https://www.youtube.com/watch?v=1c9iyoVIwDs)
- [RAGAS Framework Walkthrough](https://www.youtube.com/watch?v=RhHHiLuAvM0)

---

## Additional Resources

- [RAGAS Documentation](https://docs.ragas.io/)
- [Anthropic: Evaluating RAG](https://docs.anthropic.com/en/docs/test-and-evaluate/overview)
- [OpenAI Evals Guide](https://platform.openai.com/docs/guides/evals)
