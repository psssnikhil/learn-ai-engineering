---
title: Hybrid Search with Vector Databases
description: >-
  Combine semantic vector search with keyword-based BM25 search for more
  accurate and reliable retrieval
duration: 55 min
difficulty: intermediate
has_code: true
module: module-13
---
# Hybrid Search with Vector Databases

## Prerequisites

Before this lesson you should be comfortable with:

- **Module 09, Lesson 07 — Hybrid Search** — BM25 basics, why keyword + semantic matters
- **Module 13, Lessons 01–07** — embeddings, indexing, schema design, scaling
- **Python** — running benchmark scripts

Module 09 introduced hybrid search conceptually. This lesson goes production-deep: RRF implementation, alpha tuning with benchmark scripts, Weaviate/Qdrant native hybrid, and eval methodology to prove hybrid beats pure vector on your queries.

---

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Explain when vector-only search fails and hybrid fixes it | 10 min | Intermediate |
| Implement Reciprocal Rank Fusion (RRF) from scratch | 15 min | Intermediate |
| Run an alpha sweep benchmark comparing BM25, vector, and hybrid | 15 min | Intermediate |
| Build a production hybrid search pipeline with eval metrics | 15 min | Intermediate |

---

## Intuition First: Two Experts, One Answer

Imagine asking two librarians for book recommendations:

**The keyword expert** excels at exact matches. Ask for "error code E4041" and they find the precise troubleshooting page instantly. Ask "how to make my app faster" and they shrug — no matching words.

**The semantic expert** understands intent. "How to make my app faster" retrieves performance optimization guides. But ask for "Python 3.11.4 release notes" and they return generic Python tutorials — close in meaning, wrong version.

**Hybrid search** asks both experts, then merges their ranked lists. Documents both experts recommend rise to the top. Documents only one expert finds still appear — but lower. You get precision on exact terms *and* recall on conceptual queries.

---

## Why Vector-Only Search Fails

| Query Type | BM25 (Keyword) | Vector (Semantic) | Hybrid |
|-----------|----------------|-------------------|--------|
| "error code 404" | Excellent | Poor (matches "error" broadly) | Excellent |
| "how to fix slow queries" | Good | Excellent | Excellent |
| "Python TypeError: NoneType" | Excellent | Moderate | Excellent |
| "ways to make my app faster" | Poor (no keyword overlap) | Excellent | Excellent |
| "GDPR Article 17" | Excellent (exact legal ref) | Moderate | Excellent |
| "refund policy" | Good | Good | Good |

Pure vector search systematically fails on: error codes, SKUs, legal citations, version numbers, proper nouns, and acronym-heavy technical docs. Hybrid search is not optional for production RAG — it's the default architecture.

---

## Reciprocal Rank Fusion (RRF)

RRF merges ranked lists from different retrieval methods without requiring score normalization (BM25 scores and cosine similarities aren't comparable directly).

\[
\text{RRF}(d) = \sum_{i} \frac{1}{k + \text{rank}_i(d)}
\]

where \(k\) is a smoothing constant (default 60) and \(\text{rank}_i(d)\) is document \(d\)'s rank in list \(i\).

```python
def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Merge multiple ranked lists using RRF."""
    scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# Example
bm25_results = ["doc-3", "doc-1", "doc-7", "doc-5"]
vector_results = ["doc-1", "doc-3", "doc-2", "doc-8"]

merged = reciprocal_rank_fusion([bm25_results, vector_results])
for doc_id, score in merged[:5]:
    print(f"{doc_id}: {score:.4f}")
# doc-1 and doc-3 rank highest — appear in both lists
```

### RRF Parameter Tuning

| k Value | Effect | When to Use |
|---------|--------|-------------|
| 10 | Aggressive — top ranks dominate | When one method is much better |
| 60 | Standard default | General purpose |
| 100 | Smooth — lower ranks contribute more | When both methods are noisy |

---

## Hybrid Search with Weaviate

Weaviate provides native hybrid search combining BM25 and vector:

```python
import weaviate

client = weaviate.connect_to_local()
documents = client.collections.get("Documents")

results = documents.query.hybrid(
    query="how to optimize database performance",
    alpha=0.5,
    limit=10,
    return_metadata=weaviate.classes.query.MetadataQuery(score=True),
)

for obj in results.objects:
    print(f"[{obj.metadata.score:.3f}] {obj.properties['title']}")
```

### Alpha Parameter

`alpha` controls the BM25 vs vector balance:

| alpha | Behavior | Best For |
|-------|----------|----------|
| 0.0 | Pure BM25 (keyword) | Error codes, SKUs, exact terms |
| 0.25 | Mostly keyword | Technical docs with jargon |
| 0.5 | Balanced (default starting point) | General-purpose RAG |
| 0.75 | Mostly semantic | Conversational/natural language queries |
| 1.0 | Pure vector | Conceptual/exploratory search |

---

## Step-by-Step: Build a Hybrid Search Pipeline

```python
"""Complete hybrid search with BM25 + vector + RRF."""
from openai import OpenAI
import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

client = OpenAI()

class HybridSearcher:
    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.doc_map = {d["id"]: d for d in documents}

        # BM25 index
        tokenized = [d["text"].lower().split() for d in documents]
        self.bm25 = BM25Okapi(tokenized)

        # Vector index
        self.chroma = chromadb.Client()
        self.collection = self.chroma.create_collection("hybrid")
        self.collection.add(
            documents=[d["text"] for d in documents],
            ids=[d["id"] for d in documents],
        )

    def _bm25_search(self, query: str) -> list[str]:
        scores = self.bm25.get_scores(query.lower().split())
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.documents[i]["id"] for i in ranked]

    def _vector_search(self, query: str) -> list[str]:
        results = self.collection.query(
            query_texts=[query], n_results=len(self.documents)
        )
        return results["ids"][0]

    def search(self, query: str, k: int = 5, alpha: float = 0.5) -> list[dict]:
        """Hybrid search with configurable alpha weighting."""
        bm25_ranking = self._bm25_search(query)
        vector_ranking = self._vector_search(query)

        if alpha == 0.0:
            ranking = bm25_ranking
        elif alpha == 1.0:
            ranking = vector_ranking
        else:
            # Weighted RRF: duplicate lists proportional to alpha
            bm25_weight = max(1, int((1 - alpha) * 10))
            vector_weight = max(1, int(alpha * 10))
            lists = [bm25_ranking] * bm25_weight + [vector_ranking] * vector_weight
            ranking = [doc_id for doc_id, _ in reciprocal_rank_fusion(lists)]

        return [self.doc_map[doc_id] for doc_id in ranking[:k]]
```

---

## Hybrid Search Benchmark Script

Prove hybrid beats pure vector on your eval set:

```python
"""Benchmark BM25-only, vector-only, and hybrid search."""
import numpy as np

EVAL_QUERIES = [
    {
        "query": "error code E4041 troubleshooting",
        "relevant": {"doc-error-codes", "doc-troubleshooting"},
    },
    {
        "query": "how to improve application performance",
        "relevant": {"doc-optimization", "doc-performance-tuning"},
    },
    {
        "query": "GDPR Article 17 right to erasure",
        "relevant": {"doc-gdpr-art17", "doc-data-privacy"},
    },
    # Add 20-50 queries from your domain
]

def recall_at_k(ranking: list[str], relevant: set, k: int) -> float:
    retrieved = set(ranking[:k])
    return len(retrieved & relevant) / len(relevant) if relevant else 0.0

def benchmark_searcher(searcher, eval_queries, k=5):
    results = {"bm25": [], "vector": [], "hybrid_0.5": [], "hybrid_0.25": []}

    for item in eval_queries:
        q = item["query"]
        rel = item["relevant"]

        bm25_ranking = searcher._bm25_search(q)
        vector_ranking = searcher._vector_search(q)
        hybrid_50 = [d["id"] for d in searcher.search(q, k=len(searcher.documents), alpha=0.5)]
        hybrid_25 = [d["id"] for d in searcher.search(q, k=len(searcher.documents), alpha=0.25)]

        results["bm25"].append(recall_at_k(bm25_ranking, rel, k))
        results["vector"].append(recall_at_k(vector_ranking, rel, k))
        results["hybrid_0.5"].append(recall_at_k(hybrid_50, rel, k))
        results["hybrid_0.25"].append(recall_at_k(hybrid_25, rel, k))

    print(f"{'Method':<15} {'Recall@' + str(k):>10}")
    print("-" * 27)
    for method, recalls in results.items():
        print(f"{method:<15} {np.mean(recalls):>10.3f}")
```

Run this on 30+ labeled queries. Typical results on mixed technical corpora:

```
Method          Recall@5
---------------------------
bm25                 0.620
vector               0.710
hybrid_0.5           0.845
hybrid_0.25          0.820
```

Hybrid at alpha=0.5 often beats both individual methods by 10–20% on mixed query types.

### Alpha Sweep

```python
def alpha_sweep(searcher, eval_queries, k=5):
    print(f"{'alpha':>6} {'recall@' + str(k):>10}")
    for alpha in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
        recalls = []
        for item in eval_queries:
            ranking = [d["id"] for d in searcher.search(item["query"], k=100, alpha=alpha)]
            recalls.append(recall_at_k(ranking, item["relevant"], k))
        print(f"{alpha:>6} {np.mean(recalls):>10.3f}")
```

Pick the alpha that maximizes recall@K on your eval set. Technical docs often prefer 0.25–0.4; conversational support bots prefer 0.6–0.75.

---

## Production Hybrid Architectures

| Architecture | Components | Best For |
|-------------|-----------|----------|
| **Native hybrid** | Weaviate, Qdrant, Elasticsearch 8+ | Simplest ops, built-in alpha |
| **RRF merge** | Separate BM25 + vector indexes | Maximum control, any vector DB |
| **Vector + reranker** | Vector search → cross-encoder rerank | Highest precision, +50-100ms |
| **Elasticsearch dense_vector** | Single system for BM25 + kNN | Existing ES infrastructure |

### When to Add a Reranker After Hybrid

Hybrid search improves recall (finding relevant docs). A cross-encoder reranker (Cohere Rerank, bge-reranker) improves precision (ordering the top results correctly). Pipeline:

```
Query → Hybrid (BM25 + Vector) → Top 50 candidates → Cross-encoder rerank → Top 5 to LLM
```

Adds 50–100ms but often improves precision@5 by 10–15% over hybrid alone.

---

## Failure Modes

**Alpha never tuned.** Default 0.5 works generically but suboptimal for your query mix. Always run the alpha sweep benchmark.

**BM25 tokenization mismatch.** BM25 on raw text vs stemmed/tokenized text produces different results. Standardize preprocessing.

**RRF with highly unequal list lengths.** If BM25 returns 1000 results but vector returns 10, RRF is biased. Truncate both to same depth (e.g., top 100) before fusion.

**Ignoring stop words in technical docs.** Standard stop word removal hurts queries like "return to main menu." Use domain-specific tokenization for technical corpora.

**Hybrid without eval.** Deploying hybrid because "it's best practice" without measuring on your data. Hybrid can hurt if your queries are purely semantic.

---

## Production Notes

- Default to hybrid for production RAG. Pure vector is a simplification that fails on exact-match queries.
- Store BM25 index alongside vector index; rebuild both on document updates.
- Log which method contributed each result (BM25 rank, vector rank, RRF score) for debugging.
- Re-run alpha sweep when query patterns shift (e.g., adding a code documentation corpus).

---

## Key Takeaways

- Hybrid search combines keyword precision with semantic recall — essential for production RAG.
- RRF is the standard merge algorithm; no score normalization needed.
- Run the alpha sweep benchmark on 30+ labeled queries to find your optimal balance.
- Weaviate and Qdrant offer native hybrid; for other DBs, implement RRF yourself.
- Consider adding a cross-encoder reranker after hybrid for maximum precision.

---

## Resources

- [Weaviate: Hybrid Search](https://weaviate.io/developers/weaviate/search/hybrid)
- [RRF Paper (Cormack et al., 2009)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [BM25 Explained](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables)

---

## Next Lesson

**[Lesson 9: Vector Database Evaluation and Testing](09-lesson-09.md)** — Recall@K, MRR, ANN recall measurement, eval dataset construction, and production monitoring.
