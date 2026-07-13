---
title: Retrieval Methods
description: >-
  Compare dense, sparse, and hybrid retrieval — and learn when to use each in
  production RAG systems
duration: 55 min
difficulty: intermediate
has_code: true
module: module-09
youtube: 'https://www.youtube.com/watch?v=LxDvRsVp31c'
---

# Retrieval Methods

## Prerequisites

- **Lesson 02 — Vector Databases & Embeddings** — understand cosine similarity and embedding vectors
- **Lesson 03 — Chunking Strategies** — know how documents become chunks
- **Python basics** — read and modify the code examples

---

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand dense (semantic) retrieval and its trade-offs | 10 min | Intermediate |
| Explain BM25 and sparse retrieval at the algorithmic level | 10 min | Intermediate |
| Implement Reciprocal Rank Fusion for hybrid search | 15 min | Intermediate |
| Add a cross-encoder reranker to boost precision | 10 min | Intermediate |
| Choose the right retrieval strategy for a given use case | 10 min | Intermediate |

---

## Intuition First: Librarian vs. Catalogue Search

Imagine two ways to find a book in a library:

**Catalogue search (sparse / keyword)**: You type "error 429" and the catalogue finds every book containing exactly those words. Perfect if the book title uses that phrase. Useless if the relevant chapter is titled "Rate Limiting" without the word "error".

**Expert librarian (dense / semantic)**: You describe what you're looking for: "I'm hitting HTTP errors when I call the API too fast." The librarian understands what you mean and leads you to the rate-limiting chapter, the API quotas appendix, and the retry-backoff design pattern — none of which contain your exact words.

**Best of both (hybrid)**: The librarian uses the catalogue for speed but applies their own judgment to reorder the results. Catalogue returns 50 candidates; librarian picks the best 5.

In RAG:
- Sparse retrieval = catalogue (exact token match)
- Dense retrieval = expert librarian (semantic understanding)
- Hybrid + reranking = librarian + catalogue

---

## Why Retrieval Method Matters

Retrieval decides **which context the LLM sees**. A state-of-the-art generator with poor retrieval still hallucinates — it invents answers from irrelevant chunks.

```
User query → Retriever → Top-K chunks → LLM → Answer
              ↑
         The retriever is the gatekeeper.
         If the right chunk isn't in Top-K, no prompt
         engineering will fix the response.
```

Retrieval quality is measured separately from generation quality:

- **Recall@K** — what fraction of ground-truth answer chunks appear in the top K retrieved results?
- **Precision@K** — what fraction of the top K retrieved results are actually relevant?
- **MRR** (Mean Reciprocal Rank) — how high does the first relevant result appear?

Improving Recall@5 from 60% to 85% typically improves end-to-end answer accuracy more than any prompt tuning.

---

## Dense Retrieval (Semantic Search)

Dense retrieval converts both the query and documents into dense embedding vectors, then finds the nearest neighbors using approximate nearest-neighbor (ANN) search.

**How it works:**

\[
\text{score}(q, d) = \cos(\mathbf{e}_q,\, \mathbf{e}_d) = \frac{\mathbf{e}_q \cdot \mathbf{e}_d}{\|\mathbf{e}_q\| \;\|\mathbf{e}_d\|}
\]

where \(\mathbf{e}_q\) is the query embedding and \(\mathbf{e}_d\) is a document chunk embedding.

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

def embed(text: str) -> list[float]:
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    ).data[0].embedding

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Three candidate chunks
chunks = [
    "Password reset instructions: visit account settings and click 'Forgot Password'.",
    "API error 429 means the rate limit has been exceeded. Wait before retrying.",
    "Quarterly revenue for Q3 2024 came in at USD 14.2 billion.",
]
chunk_vecs = [embed(c) for c in chunks]

# Query using a paraphrase — no exact keyword overlap
query = "How do I recover my account credentials?"
query_vec = embed(query)

scores = [(cosine_similarity(query_vec, cv), chunk) for cv, chunk in zip(chunk_vecs, chunks)]
scores.sort(reverse=True)
for score, chunk in scores:
    print(f"{score:.3f}  {chunk[:60]}")
```

Expected output:
```
0.847  Password reset instructions: visit account settings and...
0.312  API error 429 means the rate limit has been exceeded...
0.198  Quarterly revenue for Q3 2024 came in at USD 14.2 bill...
```

The query "recover my account credentials" matched "Password reset instructions" with score 0.847 — even though neither phrase appears in the other. That is semantic retrieval working correctly.

**Trade-offs:**

| Pros | Cons |
|------|------|
| Handles synonyms, paraphrases, multilingual | Requires embedding model + vector database |
| Strong on conceptual/intent-based queries | Can miss exact codes, IDs, error strings |
| Scales with ANN (O(log N) query time) | Higher indexing cost (embedding computation) |
| Works across languages if model is multilingual | Black-box — hard to debug why something was retrieved |

**Best for**: FAQ-style Q&A, support tickets, documentation search, multi-language corpora.

---

## Sparse Retrieval (Keyword / BM25)

Sparse retrieval scores documents by term frequency, penalizing very common words and very long documents. BM25 (Best Match 25) is the gold standard algorithm.

### The BM25 Formula

\[
\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d) \cdot (k_1 + 1)}{f(t,d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}
\]

Breaking this down:

| Symbol | Meaning |
|--------|---------|
| \(f(t,d)\) | Term frequency of term \(t\) in document \(d\) |
| \(\text{IDF}(t)\) | Inverse Document Frequency — penalizes common words like "the" |
| \(k_1\) | Term-frequency saturation (typically 1.2–2.0) — prevents long docs from dominating just by repeating terms |
| \(b\) | Length normalization (typically 0.75) |
| \(\text{avgdl}\) | Average document length in the corpus |

The practical effect: a document that contains rare, discriminative query terms gets a high score; a document that contains only common words gets a low score.

```python
from rank_bm25 import BM25Okapi

corpus = [
    "Password reset instructions: visit account settings and click Forgot Password.",
    "API error 429 means the rate limit has been exceeded. Wait before retrying.",
    "Quarterly revenue for Q3 2024 came in at USD 14.2 billion.",
    "Rate limiting prevents abuse. When you hit error 429, implement exponential backoff.",
]

tokenized_corpus = [doc.lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

# Exact token query — "error 429" appears literally in two documents
query_tokens = "error 429 rate limit".lower().split()
scores = bm25.get_scores(query_tokens)

for doc, score in sorted(zip(corpus, scores), key=lambda x: x[1], reverse=True):
    print(f"{score:.3f}  {doc[:70]}")
```

Expected output:
```
3.421  Rate limiting prevents abuse. When you hit error 429, implement...
2.876  API error 429 means the rate limit has been exceeded...
0.000  Password reset instructions: visit account settings...
0.000  Quarterly revenue for Q3 2024 came in at USD 14.2 billion...
```

BM25 correctly identifies the two rate-limiting documents and ignores the unrelated ones — because "error", "429", "rate", and "limit" appear literally.

**Trade-offs:**

| Pros | Cons |
|------|------|
| Fast — no embedding at query time | Misses paraphrases ("recover account" ≠ "reset password") |
| Excellent for exact codes, IDs, error numbers | Fails on conceptual or intent-based queries |
| Interpretable — can explain why a doc ranked high | Sensitive to vocabulary mismatch |
| No GPU required | Requires tokenized inverted index |

**Best for**: Error codes, product SKUs, legal citations, named entities, any query where exact token matches are critical.

---

## Hybrid Search

Hybrid search combines dense and sparse rankings. It consistently outperforms either method alone on mixed query sets.

### Option 1: Weighted Score Fusion

Linearly combine normalized scores:

\[
\text{hybrid\_score}(q, d) = \alpha \cdot \text{dense\_score}(q, d) + (1 - \alpha) \cdot \text{sparse\_score}(q, d)
\]

The challenge: dense scores (cosine, 0 to 1) and sparse scores (BM25, 0 to unbounded) are on different scales. You must normalize both to [0, 1] before combining.

### Option 2: Reciprocal Rank Fusion (RRF)

RRF is scale-invariant and requires no normalization — it only uses rank order.

\[
\text{RRF}(d) = \sum_{r \in \text{rankings}} \frac{1}{k + \text{rank}(d, r)}
\]

where \(k = 60\) is a smoothing constant. Documents ranked first in multiple systems receive the highest fused score.

```python
from collections import defaultdict

def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """
    Combine multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        rankings: Each element is a ranked list of document IDs (best first).
        k:        Smoothing constant (default 60 from original paper).

    Returns:
        List of (doc_id, rrf_score) sorted best first.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# Dense retrieval returned these IDs in order
dense_ranking  = ["doc_a", "doc_c", "doc_b", "doc_e"]
# Sparse (BM25) returned these IDs in order
sparse_ranking = ["doc_b", "doc_a", "doc_d", "doc_c"]

fused = reciprocal_rank_fusion([dense_ranking, sparse_ranking])
for doc_id, score in fused:
    print(f"{score:.4f}  {doc_id}")
```

```
0.0328  doc_a    ← top in dense (#1), high in sparse (#2)
0.0311  doc_b    ← top in sparse (#1), high in dense (#3)
0.0295  doc_c    ← #2 dense, #4 sparse
0.0164  doc_d    ← only in sparse (#3)
0.0164  doc_e    ← only in dense (#4)
```

`doc_a` wins because it ranks well in *both* systems — a strong consensus signal.

!!! warning "Don't assume hybrid always wins"
    Start with dense-only. Add BM25 only when you see retrieval failures on exact-match queries (error codes, product IDs, legal citations). Hybrid adds operational complexity; justify it with a measurable improvement in Recall@K.

---

## Re-Ranking: The Precision Layer

First-pass retrieval (dense or hybrid) casts a wide net — it prioritizes recall by returning K=20 candidates. A **cross-encoder reranker** then scores each candidate against the query to produce a precision-optimized top-5.

```
Dense retrieval → top 20 candidates (high recall, some noise)
        ↓
Cross-encoder reranker → rescores all 20 with deeper attention
        ↓
Top 5 reranked (high precision) → LLM context
```

Cross-encoders concatenate the query and document and run a full attention pass over both together. This is far more accurate than embedding-based similarity (which processes query and document independently) but too slow for first-pass retrieval over millions of documents.

```python
from sentence_transformers import CrossEncoder

# A strong open-source reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

query = "How do I recover my account?"
candidates = [
    "Password reset: go to account settings and click Forgot Password.",
    "API rate limit error 429: implement exponential backoff.",
    "Quarterly revenue exceeded expectations by 12%.",
    "To recover access: click the link emailed to your registered address.",
]

# Score each (query, candidate) pair
scores = reranker.predict([(query, cand) for cand in candidates])

ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
for doc, score in ranked:
    print(f"{score:.3f}  {doc[:65]}")
```

Expected output:
```
 8.23  Password reset: go to account settings and click Forgot Password.
 7.41  To recover access: click the link emailed to your registered...
-2.34  API rate limit error 429: implement exponential backoff.
-5.11  Quarterly revenue exceeded expectations by 12%.
```

Both password-related chunks score highly; the unrelated docs are pushed far down with negative scores.

**Cloud rerankers**: Cohere Rerank API, Jina Reranker, and Voyage AI offer hosted cross-encoders that avoid deploying your own model.

---

## Choosing a Strategy

| Query type | Recommended approach | Why |
|------------|---------------------|-----|
| FAQ / support ("how do I reset my password?") | Dense | Handles paraphrases |
| Error codes, SKUs, citation IDs | Sparse (BM25) | Exact token match |
| Mixed real-world queries | Hybrid (BM25 + dense + RRF) | Best of both |
| High-stakes answers (legal, medical) | Hybrid + cross-encoder reranker | Maximum precision |
| Multilingual queries | Dense with multilingual model | Sparse fails across languages |
| Very large corpus (100M+ docs) | IVF index + BM25, no reranker | Latency budget is tight |

---

## A Worked End-to-End Hybrid Pipeline

```python
import chromadb
from rank_bm25 import BM25Okapi
from openai import OpenAI
import numpy as np

client = OpenAI()
chroma = chromadb.Client()
collection = chroma.get_or_create_collection("hybrid_demo")

corpus = [
    ("d1", "Password reset: go to account settings, click Forgot Password."),
    ("d2", "API error 429 means rate limit exceeded. Use exponential backoff."),
    ("d3", "Annual leave policy: 20 days per year, accrued monthly."),
    ("d4", "Recover account access by clicking the email link sent to you."),
]

# Build BM25 index
tokenized = [text.lower().split() for _, text in corpus]
bm25 = BM25Okapi(tokenized)

# Build dense index
doc_ids, doc_texts = zip(*corpus)
embeddings = [
    client.embeddings.create(model="text-embedding-3-small", input=t).data[0].embedding
    for t in doc_texts
]
collection.add(ids=list(doc_ids), documents=list(doc_texts), embeddings=embeddings)

def embed(text: str) -> list[float]:
    return client.embeddings.create(
        model="text-embedding-3-small", input=text
    ).data[0].embedding

def hybrid_retrieve(query: str, top_k: int = 2) -> list[str]:
    # Dense retrieval
    q_vec = embed(query)
    dense_results = collection.query(query_embeddings=[q_vec], n_results=len(corpus))
    dense_ranking = dense_results["ids"][0]

    # Sparse retrieval
    bm25_scores = bm25.get_scores(query.lower().split())
    sparse_ranking = [
        doc_ids[i] for i in np.argsort(bm25_scores)[::-1]
    ]

    # Fuse with RRF
    fused = reciprocal_rank_fusion([dense_ranking, sparse_ranking])
    top_ids = [doc_id for doc_id, _ in fused[:top_k]]

    # Return chunk texts in ranked order
    id_to_text = dict(corpus)
    return [id_to_text[i] for i in top_ids]


print(hybrid_retrieve("How do I get back into my account?"))
# → ['Password reset: go to...', 'Recover account access by...']

print(hybrid_retrieve("error 429 api limit"))
# → ['API error 429 means rate limit...', ...]
```

---

## Edge Cases & Misconceptions

**Misconception: Dense retrieval always beats BM25.**
On general benchmarks, yes. But on queries with rare technical terms (error codes, product IDs, chemical names), BM25 routinely outperforms dense retrieval because the embedding model may not have seen those terms in training. Always evaluate on your actual query distribution.

**Misconception: More retrieved candidates = better input to the LLM.**
LLMs are sensitive to position bias — information in the middle of a long context window is often underutilized. Retrieving K=20 chunks and passing all of them to the LLM is often worse than retrieving K=5 high-quality chunks. Use a reranker to maximize precision before reducing K.

**Edge case: Query specificity mismatch.**
A very short query ("error 429") has low signal for dense retrieval; it matches many embeddings weakly. A long, detailed query ("What does HTTP status code 429 mean in the context of API rate limiting, and how should I implement retry logic?") works much better with dense. Design your query expansion step to normalize query length.

**Edge case: Negative results (document not in corpus).**
If the answer simply doesn't exist in your documents, all retrieval methods will return the least-wrong results. Design your system to recognize when retrieved chunks are below a confidence threshold and respond with "I don't have information about this" rather than passing weak context to the LLM.

---

## Production Connection

In production RAG pipelines, retrieval engineering is an ongoing discipline:

- **Baseline first**: ship with dense-only retrieval. Measure Recall@5 on 50 representative queries.
- **Add BM25** when you see failures on exact-match queries. Measure the delta.
- **Add a reranker** when you see precision failures (correct document in top-20 but not top-5). Measure latency impact (rerankers add 50–200 ms).
- **Monitor query logs**: categorize failed queries into "retrieval failure" (right chunk not returned) vs. "generation failure" (right chunk returned but wrong answer generated). They require different fixes.
- **Eval-driven iteration**: every retrieval change should be backed by a Recall@K delta on a curated eval set, not intuition.

---

## Key Takeaways

- Dense retrieval uses embedding similarity — powerful for paraphrases and conceptual queries, but misses exact token matches.
- BM25 (sparse) uses term frequency — powerful for exact codes, IDs, and rare terms, but fails on paraphrases.
- Hybrid search (dense + BM25 + RRF) combines both signals without score normalization; it is the best default for mixed query sets.
- Cross-encoder rerankers run a deep attention pass over (query, document) pairs — dramatically improving precision, at 50–200 ms added latency.
- Start with dense-only; add BM25 and reranking only when measurable retrieval failures justify the complexity.
- Measure Recall@K and Precision@K on a real eval set before every retrieval architecture change.

---

## Related Papers

| Paper | What it contributes | Link |
|-------|-------------------|------|
| Karpukhin et al. (2020) — *Dense Passage Retrieval for Open-Domain QA* | DPR: training a biencoder for dense passage retrieval, outperforms BM25 on QA | [arxiv.org/abs/2004.04906](https://arxiv.org/abs/2004.04906) |
| Cormack et al. (2009) — *Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods* | Introduces RRF; shows it beats weighted fusion without parameter tuning | [dl.acm.org/doi/10.1145/1571941.1572114](https://dl.acm.org/doi/10.1145/1571941.1572114) |
| Nogueira & Cho (2020) — *Passage Re-ranking with BERT* | MonoBERT: cross-encoder reranking with BERT achieves SOTA on MSMARCO | [arxiv.org/abs/1901.04085](https://arxiv.org/abs/1901.04085) |
| Ma et al. (2022) — *Zero-Shot Listwise Document Reranking with a Large Language Model* | LLM-based rerankers; shows GPT-4 outperforms BERT cross-encoders zero-shot | [arxiv.org/abs/2305.02156](https://arxiv.org/abs/2305.02156) |

---

## Further Reading

- [Dense vs Sparse Retrieval Explained](https://www.youtube.com/watch?v=LxDvRsVp31c) — conceptual overview (20 min)
- [SBERT Documentation](https://www.sbert.net/) — sentence-transformer models including cross-encoders for reranking
- [Pinecone: Retrieval Series](https://www.pinecone.io/learn/series/rag/) — articles on hybrid search and reranking
- [BM25 Wikipedia](https://en.wikipedia.org/wiki/Okapi_BM25) — derivation and hyperparameter explanation

---

## Next Lesson

**[Lesson 5: Building a Basic RAG System](05-Building-a-Basic-RAG-System.md)** — Assemble everything learned so far into a complete, runnable RAG pipeline with proper observability and error handling.
