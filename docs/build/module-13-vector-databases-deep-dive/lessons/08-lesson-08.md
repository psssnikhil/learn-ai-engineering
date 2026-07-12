---
title: Hybrid Search with Vector Databases
description: >-
  Combine semantic vector search with keyword-based BM25 search for more
  accurate and reliable retrieval
duration: 40 min
difficulty: intermediate
has_code: false
module: module-13
---
# Hybrid Search with Vector Databases

## Learning Objectives

By the end of this lesson, you will be able to:
- Explain why combining keyword and semantic search improves results
- Implement hybrid search using Reciprocal Rank Fusion (RRF)
- Use Weaviate's built-in hybrid search
- Choose the right alpha parameter to balance keyword vs semantic results

---

## Why Hybrid Search?

Neither keyword search nor vector search alone is perfect:

| Query Type | Keyword (BM25) | Vector (Semantic) |
|-----------|----------------|-------------------|
| "error code 404" | Excellent (exact match) | Poor (may match "error" concepts generally) |
| "how to fix slow queries" | Good | Excellent (understands intent) |
| "Python TypeError" | Excellent | Moderate |
| "ways to make my app faster" | Poor (no keyword overlap) | Excellent |

**Hybrid search** combines both to handle all query types well.

---

## Reciprocal Rank Fusion (RRF)

RRF merges ranked lists from different search methods into a single ranking:

```python
def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60
) -> list[tuple[str, float]]:
    """Merge multiple ranked lists using RRF.

    Args:
        ranked_lists: List of ranked document ID lists
        k: Smoothing constant (default 60)

    Returns:
        Merged ranking as (doc_id, score) pairs
    """
    scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# Example: merge BM25 and vector results
bm25_results = ["doc-3", "doc-1", "doc-7", "doc-5"]
vector_results = ["doc-1", "doc-3", "doc-2", "doc-8"]

merged = reciprocal_rank_fusion([bm25_results, vector_results])
# doc-1 and doc-3 rank highest (appear in both lists)
```

---

## Hybrid Search with Weaviate

Weaviate has built-in hybrid search that combines BM25 and vector search:

```python
import weaviate

client = weaviate.connect_to_local()  # or connect_to_weaviate_cloud()

# Create collection with both vector and keyword indexing
documents = client.collections.get("Documents")

# Hybrid search: alpha controls the balance
# alpha=0 → pure keyword, alpha=1 → pure vector
results = documents.query.hybrid(
    query="how to optimize database performance",
    alpha=0.5,      # Equal weight to keyword and vector
    limit=10,
    return_metadata=weaviate.classes.query.MetadataQuery(score=True)
)

for obj in results.objects:
    print(f"[{obj.metadata.score:.3f}] {obj.properties['title']}")
```

---

## Tuning the Alpha Parameter

The `alpha` parameter controls how much weight to give vector vs keyword search:

| alpha | Behavior | Best For |
|-------|----------|----------|
| 0.0 | Pure keyword (BM25) | Exact term matching, code search |
| 0.25 | Mostly keyword | Technical documentation |
| 0.5 | Balanced | General purpose (good default) |
| 0.75 | Mostly semantic | Conversational queries |
| 1.0 | Pure vector | Conceptual/exploratory search |

```python
# Experiment with different alpha values
for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
    results = documents.query.hybrid(
        query="Python type checking error",
        alpha=alpha,
        limit=5,
    )
    print(f"
alpha={alpha}:")
    for obj in results.objects:
        print(f"  {obj.properties['title']}")
```

---

## Building a Simple Hybrid Search Pipeline

```python
from openai import OpenAI
import chromadb
from rank_bm25 import BM25Okapi

class HybridSearcher:
    def __init__(self, documents: list[dict]):
        self.documents = documents
        self.client = OpenAI()

        # Set up BM25 for keyword search
        tokenized = [doc["text"].lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

        # Set up ChromaDB for vector search
        self.chroma = chromadb.Client()
        self.collection = self.chroma.create_collection("hybrid")
        self.collection.add(
            documents=[d["text"] for d in documents],
            ids=[d["id"] for d in documents],
        )

    def search(self, query: str, k: int = 5, alpha: float = 0.5):
        """Hybrid search with configurable keyword/vector balance."""
        # BM25 keyword search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranking = [self.documents[i]["id"]
                        for i in sorted(range(len(bm25_scores)),
                                       key=lambda i: bm25_scores[i],
                                       reverse=True)]

        # Vector search
        vector_results = self.collection.query(
            query_texts=[query], n_results=len(self.documents)
        )
        vector_ranking = vector_results["ids"][0]

        # Merge with RRF
        merged = reciprocal_rank_fusion([bm25_ranking, vector_ranking])
        return merged[:k]
```

---

## Key Takeaways

- Hybrid search combines keyword matching (BM25) with semantic understanding (vectors)
- Reciprocal Rank Fusion is the standard algorithm for merging ranked results
- Start with alpha=0.5 (balanced) and tune based on your query patterns
- Weaviate provides built-in hybrid search; for other databases, implement RRF yourself
- Hybrid search is especially valuable when queries mix exact terms with natural language

## Resources

- [YouTube: Hybrid Search Explained](https://www.youtube.com/watch?v=lYxGYXjfrNI) -- Visual guide by Weaviate
- [Weaviate: Hybrid Search](https://weaviate.io/developers/weaviate/search/hybrid) -- Official documentation
- [BM25 Explained](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables) -- Deep dive into the BM25 algorithm
- [RRF Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) -- Original Reciprocal Rank Fusion paper

---

Next: Vector Database Evaluation and Testing
