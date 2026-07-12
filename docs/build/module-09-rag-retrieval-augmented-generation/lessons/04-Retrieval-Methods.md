---
title: Retrieval Methods
description: >-
  Compare dense, sparse, and hybrid retrieval — and learn when to use each in
  production RAG systems
duration: 35 min
difficulty: intermediate
has_code: false
module: module-09
youtube: 'https://www.youtube.com/watch?v=LxDvRsVp31c'
---
## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand dense vs sparse retrieval | 35 min | Intermediate |
| Compare BM25 and embedding search | | |
| Design hybrid retrieval pipelines | | |
| Choose the right method per use case | | |

---

## Why Retrieval Method Matters

Retrieval decides **which context** the LLM sees. A strong generator with weak retrieval still hallucinates — it simply invents answers from irrelevant chunks.

```
User query → Retriever → Top-K chunks → LLM → Answer
              ↑
         This step is critical
```

> **Tip:** Measure retrieval quality separately from generation quality. If Recall@K is low, better prompts won't save you.

---

## Dense Retrieval (Semantic Search)

**Dense retrieval** converts queries and documents into embedding vectors, then finds nearest neighbors in vector space.

**Best for:**
- Paraphrased questions ("reset password" vs "recover credentials")
- Conceptual queries across varied wording
- Multilingual or informal user language

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def embed(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

query = embed("How do I recover my account?")
doc_a = embed("Password reset instructions for users")
doc_b = embed("Quarterly sales report Q3 2024")

print(cosine_similarity(query, doc_a))  # High — semantically related
print(cosine_similarity(query, doc_b))  # Low — unrelated topic
```

**Trade-offs:**

| Pros | Cons |
|------|------|
| Handles synonyms and paraphrases | Requires embedding model + vector DB |
| Strong semantic matching | Can miss exact IDs, SKUs, error codes |
| Scales with approximate search (ANN) | More expensive at index time |

---

## Sparse Retrieval (Keyword / BM25)

**Sparse retrieval** scores documents by term overlap — classic search engine behavior. BM25 is the most common algorithm.

**Best for:**
- Exact product codes, legal citations, API names
- Queries with rare, discriminative terms
- Low-latency search without GPU embedding calls

```python
from rank_bm25 import BM25Okapi

corpus = [
    "Reset your password from account settings",
    "API error 429 means rate limit exceeded",
    "Employee handbook section 4.2 vacation policy",
]

tokenized = [doc.lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized)

query = "error 429 rate limit".lower().split()
scores = bm25.get_scores(query)

for doc, score in sorted(zip(corpus, scores), key=lambda x: x[1], reverse=True):
    print(f"{score:.2f}  {doc}")
```

**Trade-offs:**

| Pros | Cons |
|------|------|
| Fast, no embedding cost at query time | Misses semantic paraphrases |
| Excellent for exact token matches | Weak on conceptual questions |
| Interpretable scores | Sensitive to vocabulary mismatch |

---

## Hybrid Search

**Hybrid search** combines dense and sparse scores — often the best default for production RAG.

Common fusion strategies:

1. **Weighted sum** — `score = α * dense + (1-α) * sparse`
2. **Reciprocal Rank Fusion (RRF)** — merge ranked lists without normalizing scores
3. **Retrieve then re-rank** — cast a wide net, then use a cross-encoder

```python
def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)

dense_ranking = ["doc_a", "doc_c", "doc_b"]
sparse_ranking = ["doc_b", "doc_a", "doc_d"]

print(reciprocal_rank_fusion([dense_ranking, sparse_ranking]))
```

> **Warning:** Don't assume hybrid always wins. Start with dense-only, add sparse when you see failures on exact-match queries.

---

## Choosing a Strategy

| Query type | Recommended approach |
|------------|---------------------|
| FAQ / support ("how do I…") | Dense or hybrid |
| SKU / error code lookup | Sparse or hybrid |
| Legal / compliance docs | Hybrid + metadata filters |
| Internal wiki search | Dense + re-ranker |

---

## Recommended Videos

- [Dense vs Sparse Retrieval](https://www.youtube.com/watch?v=LxDvRsVp31c) — Understanding retrieval approaches
- [BM25 Explained](https://www.youtube.com/watch?v=a3sg6MH8m4k) — Classic keyword-based retrieval

---

## Additional Resources

- [SBERT: Sentence Transformers](https://www.sbert.net/) — Dense retrieval models
- [BM25 Algorithm](https://en.wikipedia.org/wiki/Okapi_BM25) — How BM25 scoring works
- [Pinecone: Retrieval](https://www.pinecone.io/learn/series/rag/) — RAG retrieval series
