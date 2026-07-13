---
title: Hybrid Search
description: >-
  Combine dense embeddings and sparse keyword search for robust retrieval across
  diverse query types
duration: 50 min
difficulty: intermediate
has_code: true
module: module-09
youtube: 'https://www.youtube.com/watch?v=OujMiengFaE'
---

## Prerequisites

- [Lesson 04 — Retrieval Methods](04-Retrieval-Methods.md): dense vs sparse retrieval fundamentals
- [Lesson 02 — Vector Databases](02-vector-databases.md): how embeddings are stored and searched
- Familiarity with cosine similarity and basic probability concepts
- Python experience with lists and dictionaries

---

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Explain why neither dense nor sparse retrieval alone is sufficient | 50 min | Intermediate |
| Understand BM25 scoring and where it beats embeddings | | |
| Implement Reciprocal Rank Fusion from scratch | | |
| Apply weighted score fusion with a tunable alpha parameter | | |
| Configure hybrid search in Weaviate, Pinecone, and Elasticsearch | | |
| Tune alpha weights for different domains using an eval set | | |

---

## Intuition First

Imagine you're searching a technical wiki. You type: `"ECONNREFUSED error port 5432"`.

A **dense embedding model** reads that query and produces a vector representing its general meaning: "network connection refused on a specific port." It will retrieve documents about PostgreSQL connection failures and Docker networking — useful conceptual matches.

But it might miss the one document that contains the *exact* string `ECONNREFUSED 5432` in a troubleshooting table, because embedding distance rewards semantic similarity, not lexical overlap.

Now flip it. You type: `"How do I make the application more resilient to database outages?"`. A keyword search on BM25 looks for the literal words "resilient", "database", "outages" — but your best documentation might use "fault tolerance", "retry logic", and "connection pool". BM25 misses it entirely.

**Hybrid search** runs both in parallel and merges the ranked results. It covers:

- Exact matches: error codes, SKUs, names, versions (sparse wins)
- Semantic matches: paraphrases, synonyms, conceptual queries (dense wins)
- Mixed: most real-world enterprise queries

---

## How BM25 Works

BM25 (Best Match 25) is the industry-standard sparse retrieval function. It scores a document `d` for query `q` as:

\[
\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}
\]

Where:
- \( f(t, d) \) — term frequency of token `t` in document `d`
- \( \text{IDF}(t) = \log\left(\frac{N - n_t + 0.5}{n_t + 0.5} + 1\right) \) — inverse document frequency; rare terms score higher
- \( |d| \) — document length; `avgdl` — average document length across corpus
- \( k_1 \approx 1.5 \) — term frequency saturation parameter
- \( b \approx 0.75 \) — length normalization parameter

**Plain-language interpretation:**
- IDF rewards *rare* terms (high signal) and penalizes common words like "the" (low signal)
- The saturation function prevents one term from dominating just by appearing many times
- Length normalization prevents long documents from winning by sheer word count

**Worked example:**

Corpus: 3 documents
- Doc A (50 tokens): "Reset your password in Settings → Security → Reset Password"
- Doc B (120 tokens): "Password security best practices include 2FA and strong password choices"
- Doc C (80 tokens): "API authentication uses tokens, not passwords"

Query: `"reset password"`

Term `"reset"` appears: Doc A (2×), Doc B (0×), Doc C (0×). IDF is high (appears in 1/3 docs).
Term `"password"` appears: Doc A (2×), Doc B (2×), Doc C (1×). IDF is lower (appears in all 3).

BM25 ranks Doc A highest — exact keyword match with high-IDF terms. Doc B ranks second (has "password" but not "reset"). This is exactly what you want for this query.

---

## How Hybrid Search Works

```
                    ┌─ Dense search (embeddings) ─→ Ranked list A
User query ────────►│
                    └─ Sparse search (BM25) ───→ Ranked list B
                                    ↓
                            Fusion (RRF / weighted)
                                    ↓
                              Final top-K
```

The challenge is that dense scores (cosine similarity, typically 0.0–1.0) and sparse scores (BM25, typically 0.0–20.0) are on completely different scales. Two fusion strategies handle this:

1. **Reciprocal Rank Fusion (RRF)** — ignores scores entirely, fuses based on ranks
2. **Weighted score fusion** — normalizes both score ranges, then blends with a weight α

---

## Reciprocal Rank Fusion (RRF)

RRF is the most robust and most commonly recommended approach. It requires no score normalization because it only uses rank positions.

**Formula:**

\[
\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}
\]

Where `R` is the set of ranked lists, `rank_r(d)` is document `d`'s position in list `r` (1-indexed), and `k = 60` is a smoothing constant that prevents the top-1 result from dominating completely.

**Worked example:**

| Document | Dense rank | Sparse rank | RRF score |
|----------|-----------|-------------|-----------|
| doc_password | 1 | 2 | 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = **0.03252** |
| doc_security | 2 | 1 | 1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = **0.03252** |
| doc_billing | 3 | — | 1/(60+3) + 0 = **0.01587** |
| doc_api_errors | — | 3 | 0 + 1/(60+3) = **0.01587** |

Both `doc_password` and `doc_security` tie at the top because both retrievers agree on their relevance (in different orders). Documents only one retriever found rank lower.

```python
def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Merge N ranked lists using RRF.
    
    Args:
        rankings: Each inner list is a ranked ordering of document IDs,
                  best-first. Documents absent from a list score 0 for that list.
        k: Smoothing constant (default 60, from the original RRF paper).
    
    Returns:
        Sorted list of (doc_id, rrf_score) tuples, highest score first.
    """
    scores: dict[str, float] = {}

    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# Example usage
dense_results = ["doc_password", "doc_security", "doc_billing"]
sparse_results = ["doc_security", "doc_password", "doc_api_errors"]

fused = reciprocal_rank_fusion([dense_results, sparse_results])
for doc_id, score in fused:
    print(f"{score:.5f}  {doc_id}")

# Output:
# 0.03252  doc_password
# 0.03252  doc_security
# 0.01587  doc_billing
# 0.01587  doc_api_errors
```

**Why k=60?** The original RRF paper found k=60 robust across benchmarks — it ensures that the score difference between rank 1 and rank 2 is small enough that agreement across retrievers matters more than absolute rank. You can tune k on your eval set, but the improvement over the default is usually marginal.

---

## Weighted Score Fusion

When both retrievers return normalized scores (0.0–1.0), you can blend them linearly:

\[
\text{score}(d) = \alpha \cdot \text{score}_\text{dense}(d) + (1 - \alpha) \cdot \text{score}_\text{sparse}(d)
\]

```python
def min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Normalize scores to [0, 1] range."""
    if not scores:
        return {}
    min_s = min(scores.values())
    max_s = max(scores.values())
    span = max_s - min_s
    if span == 0:
        return {k: 1.0 for k in scores}
    return {k: (v - min_s) / span for k, v in scores.items()}


def weighted_fusion(
    dense_scores: dict[str, float],
    sparse_scores: dict[str, float],
    alpha: float = 0.5,
) -> list[tuple[str, float]]:
    """
    Blend dense and sparse scores.
    
    Args:
        alpha: Weight for dense scores. 1.0 = dense only, 0.0 = sparse only.
    """
    dense_norm = min_max_normalize(dense_scores)
    sparse_norm = min_max_normalize(sparse_scores)

    all_ids = set(dense_norm) | set(sparse_norm)
    combined = {
        doc_id: alpha * dense_norm.get(doc_id, 0.0) + (1 - alpha) * sparse_norm.get(doc_id, 0.0)
        for doc_id in all_ids
    }
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)
```

| Alpha | Behavior | When to use |
|-------|----------|-------------|
| 1.0 | Dense only | Purely semantic queries, no exact-match needs |
| 0.7 | Dense-heavy | General knowledge base, support FAQs |
| 0.5 | Balanced | Default starting point for most domains |
| 0.3 | Sparse-heavy | Technical docs with codes, IDs, version numbers |
| 0.0 | Sparse only | Pure keyword lookup |

!!! tip "How to tune alpha"
    Build an eval set of 50–100 labeled queries. For each alpha in `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`, compute Recall@5. Plot the curve and pick the alpha at the peak. Most enterprise corpora land between 0.5 and 0.7.

---

## Hybrid in Vector Databases

Most modern vector databases support hybrid search natively, handling the BM25 index alongside the dense index internally.

### Weaviate (alpha parameter)

```python
import weaviate

client = weaviate.connect_to_local()

# Weaviate alpha: 0.0 = pure BM25, 1.0 = pure vector
response = (
    client.collections.get("Document")
    .query.hybrid(
        query="password reset",
        alpha=0.75,  # dense-heavy
        limit=5,
        return_properties=["content", "title"],
    )
)

for obj in response.objects:
    print(obj.properties["title"])
```

### Pinecone (sparse-dense vectors)

Pinecone implements hybrid by storing separate sparse and dense vectors per document, then combining at query time with a weighted sum.

```python
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

pc = Pinecone(api_key="your-api-key")
index = pc.Index("my-hybrid-index")

# Encode documents (at index time)
bm25 = BM25Encoder()
bm25.fit(corpus_documents)

def upsert_hybrid(doc_id: str, text: str, dense_embedding: list[float]):
    sparse_vector = bm25.encode_documents([text])[0]
    index.upsert(vectors=[{
        "id": doc_id,
        "values": dense_embedding,
        "sparse_values": sparse_vector,
        "metadata": {"text": text},
    }])

def query_hybrid(query: str, dense_embedding: list[float], top_k: int = 5, alpha: float = 0.7):
    sparse_query = bm25.encode_queries([query])[0]
    # Pinecone blends: alpha * dense + (1-alpha) * sparse
    return index.query(
        vector=dense_embedding,
        sparse_vector=sparse_query,
        top_k=top_k,
        include_metadata=True,
    )
```

### Elasticsearch (kNN + BM25 hybrid)

Elasticsearch 8.x supports hybrid queries combining `knn` and `match` clauses:

```python
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

def hybrid_search_es(query: str, query_vector: list[float], top_k: int = 5):
    return es.search(
        index="documents",
        body={
            "query": {
                "bool": {
                    "should": [
                        # BM25 component
                        {"match": {"content": {"query": query, "boost": 0.3}}},
                    ]
                }
            },
            # Dense component
            "knn": {
                "field": "embedding",
                "query_vector": query_vector,
                "k": top_k * 2,
                "num_candidates": 100,
                "boost": 0.7,
            },
            "size": top_k,
        },
    )
```

---

## Full Hybrid RAG Pipeline

Putting it all together: a standalone hybrid retrieval function that works with any embedding model and BM25 index:

```python
from rank_bm25 import BM25Okapi  # pip install rank-bm25
import numpy as np

class HybridRetriever:
    def __init__(self, documents: list[str], embedder, alpha: float = 0.6):
        self.documents = documents
        self.embedder = embedder
        self.alpha = alpha

        # Build BM25 index
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

        # Build dense index (numpy for demo; use FAISS/vector DB in production)
        self.dense_matrix = np.array(embedder.encode(documents))

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        # Sparse scores
        sparse_scores_raw = self.bm25.get_scores(query.lower().split())

        # Dense scores (cosine similarity)
        query_vec = np.array(self.embedder.encode([query])[0])
        dense_scores_raw = self.dense_matrix @ query_vec  # shape: (N,)

        # Normalize to [0, 1]
        def norm(arr):
            span = arr.max() - arr.min()
            return (arr - arr.min()) / span if span > 0 else np.ones_like(arr)

        sparse_norm = norm(sparse_scores_raw)
        dense_norm = norm(dense_scores_raw)

        # Weighted fusion
        combined = self.alpha * dense_norm + (1 - self.alpha) * sparse_norm
        top_indices = np.argsort(combined)[::-1][:top_k]

        return [self.documents[i] for i in top_indices]
```

---

## When to Use Hybrid

| Scenario | Recommendation |
|----------|---------------|
| Customer support FAQ | Hybrid (α ≈ 0.6) — users paraphrase; some want exact procedure names |
| API/SDK documentation | Hybrid (α ≈ 0.4) — function names and error codes need exact matching |
| Narrative knowledge base (Wiki, HR handbook) | Dense-heavy (α ≈ 0.8) — concepts dominate |
| Product catalog with model numbers, SKUs | Sparse-heavy (α ≈ 0.2) — exact match is the signal |
| Scientific literature | Dense-heavy (α ≈ 0.75) — concepts, methods, not keywords |

!!! warning "Hybrid doubles index complexity"
    You now maintain two indexes (vector + inverted) that must stay in sync on every document update. If your eval set shows sparse adds less than 5% Recall@5 improvement over dense alone, the operational overhead may not be worth it.

---

## Tuning Alpha with an Eval Set

The alpha tuning process is straightforward once you have a labeled eval set from [Lesson 08 — RAG Evaluation Metrics](08-RAG-Evaluation-Metrics.md):

```python
def find_best_alpha(
    eval_set: list[dict],
    dense_retriever,
    sparse_retriever,
    alphas: list[float] | None = None,
    k: int = 5,
) -> dict[str, float]:
    """Grid search over alpha values; return Recall@K per alpha."""
    if alphas is None:
        alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    results = {}
    for alpha in alphas:
        recalls = []
        for item in eval_set:
            dense_scores = dense_retriever.score(item["question"])
            sparse_scores = sparse_retriever.score(item["question"])
            fused = weighted_fusion(dense_scores, sparse_scores, alpha=alpha)
            retrieved_ids = [doc_id for doc_id, _ in fused[:k]]
            recalls.append(recall_at_k(retrieved_ids, item["relevant_doc_ids"], k))
        results[alpha] = sum(recalls) / len(recalls)
        print(f"alpha={alpha:.1f}  Recall@{k}={results[alpha]:.3f}")

    best_alpha = max(results, key=results.get)
    print(f"\nBest alpha: {best_alpha} (Recall@{k}={results[best_alpha]:.3f})")
    return results
```

Run this grid search on 50–100 labeled queries from your target domain. The curve shape tells you about your corpus: a flat curve means hybrid adds little over pure dense; a steep peak around 0.3 means your corpus is keyword-heavy.

---

## Common Misconceptions

**"Hybrid search always beats dense alone."** Not true. For long-form narrative content with rich semantic vocabulary, dense retrieval often wins outright. Always measure on your own corpus.

**"RRF scores are comparable across queries."** RRF scores are only meaningful for ranking within a single query's results — not for comparing relevance across different queries or for thresholding ("only include if score > 0.03").

**"You need BM25 if you have a cross-encoder reranker."** Not necessarily. A strong bi-encoder + cross-encoder pipeline can match or exceed hybrid + re-ranking for semantic corpora. Add BM25 when your corpus has high lexical specificity (codes, versions, proper nouns).

**"Alpha is a fixed hyperparameter."** In production, alpha can be *adaptive* — route technical queries (detected by presence of error codes, version strings) to lower alpha, and conceptual queries to higher alpha.

---

## Production Tips

- **Index sync strategy:** When a document is updated, update both the vector embedding and the BM25 index atomically — stale BM25 while the vector updates causes inconsistent rankings.
- **Latency budget:** For most vector databases, the hybrid query runs in a single round-trip. For self-assembled pipelines (separate BM25 + dense), run both in parallel with `asyncio.gather` to keep latency near the slower of the two, not their sum.
- **Tokenization consistency:** BM25 tokenization (lowercasing, stemming, stopwords) at index time must match tokenization at query time exactly. Inconsistency silently degrades recall for rare terms.
- **Monitor per-retriever quality:** Log both dense Recall@K and sparse Recall@K separately. If sparse recall drops after an index change, the BM25 rebuild may have failed silently.

---

## Key Takeaways

- Dense retrieval captures semantic meaning; sparse retrieval (BM25) captures lexical precision — neither alone covers all query types
- BM25 rewards rare, specific terms (error codes, IDs) through inverse document frequency; dense embeddings reward conceptual similarity
- **RRF** fuses by rank position — robust, no normalization needed, default choice for most implementations
- **Weighted fusion** requires score normalization but gives fine-grained control via the alpha parameter
- Most production vector databases (Weaviate, Pinecone, Elasticsearch) support hybrid natively — prefer that over assembling two separate indexes
- Tune alpha on a labeled eval set; start at 0.5 and measure Recall@5 per query category

---

## Related Papers

| Paper | Year | Key contribution |
|-------|------|-----------------|
| [Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) | 2009 | Introduced RRF; showed rank fusion beats score fusion for IR |
| [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://arxiv.org/abs/2107.05720) | 2021 | Learned sparse representations that bridge lexical and semantic retrieval |
| [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663) | 2021 | Benchmarks across 18 datasets — shows no single method dominates |
| [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906) | 2020 | Seminal dense retrieval paper; baseline for hybrid comparisons |

---

## Next Lesson

**[Lesson 08 — RAG Evaluation Metrics](08-RAG-Evaluation-Metrics.md):** Build a systematic eval pipeline that separates retrieval failures from generation failures, and learn Recall@K, MRR, faithfulness, and answer relevance metrics.
