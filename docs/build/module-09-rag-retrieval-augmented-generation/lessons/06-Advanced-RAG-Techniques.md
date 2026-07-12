---
title: Advanced RAG Techniques
description: >-
  Level up RAG with query rewriting, re-ranking, contextual compression, and
  multi-step retrieval
duration: 45 min
difficulty: advanced
has_code: false
module: module-09
youtube: 'https://www.youtube.com/watch?v=TRjq7t2Ms5I'
---
## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Apply query transformation before retrieval | 45 min | Advanced |
| Use cross-encoders for re-ranking | | |
| Compress context to fit token budgets | | |
| Chain retrieval steps for complex questions | | |

---

## When Basic RAG Falls Short

Naive RAG — embed query, fetch top-K, generate — breaks down when:

- Users ask **multi-part questions** ("Compare plan A vs plan B pricing")
- Queries use **ambiguous shorthand** ("the new policy")
- Documents are **long** and top-K chunks exceed context limits
- **Precision** matters more than recall (legal, medical, finance)

> **Tip:** Advanced RAG adds stages *before* and *after* vector search. Each stage costs latency — add them when baseline RAG fails evals, not preemptively.

---

## Query Transformation

Rewrite or expand the user query before embedding.

**HyDE (Hypothetical Document Embeddings):** Ask the LLM to write a hypothetical answer, then embed *that* for search.

```python
def hyde_query(user_query: str, client) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Write a short paragraph that would answer: {user_query}
"
                    "Do not say you don't know — write the best plausible answer."
                ),
            }
        ],
    )
    return response.choices[0].message.content
```

**Multi-query:** Generate several search queries from one user question.

```python
def multi_query(user_query: str, client) -> list[str]:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Generate 3 different search queries for: {user_query}
"
                    "Return one query per line."
                ),
            }
        ],
    )
    return [q.strip() for q in response.choices[0].message.content.split("
") if q.strip()]
```

---

## Re-Ranking with Cross-Encoders

Bi-encoders (embedding search) are fast but approximate. **Cross-encoders** score query-document pairs jointly — slower but much more accurate.

```
Stage 1: Vector search → top 50 candidates (fast, high recall)
Stage 2: Cross-encoder re-rank → top 5 (slow, high precision)
Stage 3: LLM generation
```

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, documents: list[str], top_k: int = 5) -> list[str]:
    pairs = [[query, doc] for doc in documents]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]
```

---

## Contextual Compression

When retrieved chunks are noisy or too long, filter them before sending to the LLM.

```python
def compress_context(query: str, chunks: list[str], client) -> str:
    joined = "

".join(f"[{i}] {c}" for i, c in enumerate(chunks))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": (
                    f"Question: {query}

"
                    f"Passages:
{joined}

"
                    "Extract only the sentences relevant to the question. "
                    "Omit irrelevant passages entirely."
                ),
            }
        ],
    )
    return response.choices[0].message.content
```

---

## Multi-Step Retrieval

For complex questions, decompose into sub-queries and retrieve separately.

```
"Which plan has better support AND lower price?"
  → Sub-query 1: "support features by plan"
  → Sub-query 2: "pricing by plan"
  → Merge contexts → Single LLM answer
```

> **Warning:** Each extra LLM call adds cost and latency. Use multi-step retrieval when single-pass eval scores are consistently below target.

---

## Technique Selection Guide

| Technique | Adds latency | Best when |
|-----------|-------------|-----------|
| Query rewriting | Low–medium | Ambiguous or short queries |
| HyDE | Medium | Vocabulary mismatch between query and docs |
| Re-ranking | Medium | High precision needed, moderate corpus |
| Compression | Medium | Long chunks, tight context budget |
| Multi-step | High | Multi-hop or comparative questions |

---

## Recommended Videos

- [Advanced RAG Techniques](https://www.youtube.com/watch?v=TRjq7t2Ms5I)
- [Re-ranking for RAG](https://www.youtube.com/watch?v=EE-xtp2iBL4)

---

## Additional Resources

- [LangChain Advanced RAG](https://python.langchain.com/docs/how_to/#retrieval)
- [Cohere Rerank API](https://docs.cohere.com/docs/rerank)
- [LlamaIndex Query Transformations](https://docs.llamaindex.ai/en/stable/optimizing/advanced_retrieval/query_transformations/)
