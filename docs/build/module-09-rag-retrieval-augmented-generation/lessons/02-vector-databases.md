---
title: Vector Databases & Embeddings
description: Master embeddings and vector similarity search - the foundation of RAG
duration: 55 min
difficulty: intermediate
has_code: true
module: module-09
youtube: 'https://www.youtube.com/watch?v=klTvEwg3oJ4'
---

# Vector Databases & Embeddings

## Prerequisites

- **Lesson 01 — Introduction to RAG** — you should know why we retrieve before generating
- **Basic linear algebra** — vectors, dot products (you don't need to derive anything; intuition is enough)
- **Python + NumPy** — comfortable reading array operations

---

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand what an embedding is geometrically | 10 min | Intermediate |
| Compute and interpret cosine similarity manually | 10 min | Intermediate |
| Choose between HNSW, IVF, and flat index strategies | 10 min | Intermediate |
| Build a working retrieval index with ChromaDB | 15 min | Intermediate |
| Compare embedding models by cost, dimension, and quality | 10 min | Intermediate |

---

## Intuition First: Meaning as Geometry

When you learn a language you build an intuitive sense for which words are related. "Dog" and "puppy" feel close; "dog" and "quarterly report" feel far. Embeddings formalize this intuition mathematically.

An **embedding model** is a neural network trained to map text to a point in high-dimensional space, such that *semantically similar text lands near each other*. The space has no physical meaning — dimension 742 doesn't mean anything named — but the distances and angles between points encode relationships learned from billions of text examples.

Think of it like a city map where every neighborhood (concept) is a location. "Machine learning" and "deep learning" live two blocks apart; "machine learning" and "medieval poetry" are across the continent. The vector database is the map; similarity search is asking "what's closest to this address?"

---

## What Are Embeddings?

An **embedding** is a fixed-length list of floating-point numbers — a vector — that represents a piece of text. OpenAI's `text-embedding-3-small` model produces 1,536 numbers for any input text, from a single word to an 8,000-token passage.

```
"The refund policy allows returns within 30 days."
    → [0.021, -0.145, 0.872, 0.003, ..., -0.034]
        ↑_________________________________↑
              1,536 floating-point numbers
```

The vector itself is meaningless in isolation. Its value comes from *comparison*: if two texts have similar vectors (high dot product or small angle between them), they are semantically related.

**The fundamental property:**

\[
\text{similar meaning} \;\Leftrightarrow\; \text{small angle between vectors}
\]

This is why we use cosine similarity — it measures the angle, ignoring vector magnitude.

---

## Cosine Similarity: A Numerical Walkthrough

Cosine similarity between two vectors \(\mathbf{a}\) and \(\mathbf{b}\) is:

\[
\cos(\theta) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \;\|\mathbf{b}\|}
\]

The result ranges from \(-1\) (opposite) to \(+1\) (identical direction).

Let's work through a toy 3-dimension example:

| Text | Vector |
|------|--------|
| "return policy refund" | [0.9, 0.3, 0.1] |
| "can I get my money back?" | [0.8, 0.4, 0.2] |
| "quarterly earnings report" | [0.1, 0.1, 0.9] |

Similarity between the first two (semantically related):

\[
\frac{(0.9)(0.8) + (0.3)(0.4) + (0.1)(0.2)}{\sqrt{0.9^2+0.3^2+0.1^2} \;\cdot\; \sqrt{0.8^2+0.4^2+0.2^2}}
= \frac{0.72 + 0.12 + 0.02}{\sqrt{0.91} \;\cdot\; \sqrt{0.84}}
\approx \frac{0.86}{0.874} \approx 0.984
\]

Similarity between "return policy" and "earnings report" (unrelated):

\[
\frac{(0.9)(0.1) + (0.3)(0.1) + (0.1)(0.9)}{\sqrt{0.91} \;\cdot\; \sqrt{0.83}}
= \frac{0.09 + 0.03 + 0.09}{0.869} \approx \frac{0.21}{0.869} \approx 0.24
\]

Score 0.98 → almost identical meaning. Score 0.24 → unrelated topics. This is how your retrieval step finds the right chunks.

```python
import numpy as np

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Toy 3-D example
refund   = [0.9, 0.3, 0.1]
money_back = [0.8, 0.4, 0.2]
earnings = [0.1, 0.1, 0.9]

print(cosine_similarity(refund, money_back))   # 0.984 — very similar
print(cosine_similarity(refund, earnings))     # 0.241 — unrelated
```

---

## Creating Real Embeddings

```python
from openai import OpenAI

client = OpenAI()

def embed(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",   # 1,536 dimensions
        input=text,
    )
    return response.data[0].embedding

# Observe real similarity
cat_vec   = embed("The cat sat on the mat")
dog_vec   = embed("A dog was lying on the carpet")
sales_vec = embed("Q3 revenue exceeded expectations by 12%")

print(f"cat–dog similarity:   {cosine_similarity(cat_vec, dog_vec):.3f}")
# Typical output: 0.870–0.920

print(f"cat–sales similarity: {cosine_similarity(cat_vec, sales_vec):.3f}")
# Typical output: 0.120–0.200
```

!!! note "Batching saves money and time"
    The OpenAI embeddings API accepts lists of strings. Always batch your documents rather than embedding one at a time: `client.embeddings.create(model=..., input=["text1", "text2", ...])`. A single batched call is 10–50× faster than N individual calls.

---

## Why Not Just Use Keyword Search?

Before vector databases, search meant keyword matching: find documents containing the exact words in the query. This fails for:

- **Paraphrases** — "reset my password" ≠ "recover account credentials" in keyword space; they are neighbors in embedding space.
- **Conceptual queries** — "fastest animal" should retrieve "cheetah can reach 70 mph" even though the word "fastest" doesn't appear.
- **Multilingual** — multilingual embedding models encode semantically equivalent phrases from different languages near each other.

Keyword search has advantages (speed, exact-match on codes and IDs) so modern RAG systems often combine both. That's hybrid search — covered in Lesson 07.

---

## Vector Databases: Purpose and Architecture

Storing 10 million embeddings in a NumPy array and scanning every vector on each query takes seconds — unusable in production. **Vector databases** solve this with approximate nearest-neighbor (ANN) index structures that return the top-K most similar vectors in milliseconds.

### What a vector database provides

1. **Ingestion** — accept vectors (with optional metadata like document ID, timestamp, source).
2. **Indexing** — build a data structure that enables sub-linear search.
3. **Query** — given a query vector, return the K nearest vectors by distance metric.
4. **Metadata filtering** — "search semantically, but only among documents tagged `category=legal`".
5. **Persistence** — durable storage that survives restarts.

### Index Strategies

**Flat (Exact)**  
Computes distance to every stored vector. Perfect recall, O(N) time. Acceptable for fewer than ~100K vectors; impractical at scale.

**HNSW (Hierarchical Navigable Small World)**  
Builds a multi-layer graph where each node connects to its nearest neighbors. Search traverses the graph starting from a few entry points, making locally greedy hops. Achieves ~95% recall with O(log N) complexity. The default in most vector databases.

\[
\text{Build time: } O(N \log N) \qquad \text{Query time: } O(\log N)
\]

**IVF (Inverted File Index)**  
Divides vectors into K clusters using k-means. At query time, only the nearest C clusters are searched. Good for very large corpora (tens of millions of vectors) where HNSW memory is prohibitive.

```
┌──────────────────────────────────────────────┐
│            HNSW Multi-Layer Graph            │
│                                              │
│  Layer 2 (coarse):   ●────────────●         │
│                       \          /           │
│  Layer 1 (medium):  ●──●────●──●──●         │
│                    / \  \  / \/ \/           │
│  Layer 0 (fine): ●──●──●──●──●──●──●──●    │
│                                              │
│  Query starts at Layer 2, greedy descends   │
└──────────────────────────────────────────────┘
```

---

## Vector Database Options

| Database | Type | Best for | Notes |
|----------|------|----------|-------|
| **ChromaDB** | Open-source | Local dev, < 500K docs | In-memory or local file, zero infra |
| **Qdrant** | Open-source | Production self-hosted | Rust core, fast, rich filtering |
| **Weaviate** | Open-source | Knowledge graph + vectors | GraphQL query language |
| **Pinecone** | Managed SaaS | Scale without ops burden | Serverless tier for small projects |
| **pgvector** | PostgreSQL ext | Already have Postgres | SQL filtering, no extra infra |
| **Milvus** | Open-source | Billion-scale | Complex ops; use Zilliz cloud variant |

**Decision heuristic:**

- Prototyping / local dev → **ChromaDB**
- Production, self-hosted, < 50M vectors → **Qdrant**
- Already on PostgreSQL → **pgvector**
- Fully managed with minimal ops → **Pinecone**

---

## ChromaDB Worked Example

```python
import chromadb
from openai import OpenAI

client = OpenAI()
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection(
    name="company_docs",
    metadata={"hnsw:space": "cosine"},  # use cosine distance
)

# ── Ingest documents ─────────────────────────────────────
docs = [
    ("policy-001", "Refunds available within 30 days. Items must be unused."),
    ("policy-002", "Shipping takes 5–7 business days. Expedited: USD 15."),
    ("policy-003", "Contact support@example.com for help."),
    ("hr-001",     "Annual leave: 20 days per year, accrued monthly."),
    ("hr-002",     "Remote work policy: up to 3 days per week with manager approval."),
]

ids, texts, embeddings, metadatas = [], [], [], []

for doc_id, text in docs:
    vec = client.embeddings.create(
        model="text-embedding-3-small", input=text
    ).data[0].embedding
    ids.append(doc_id)
    texts.append(text)
    embeddings.append(vec)
    metadatas.append({"category": doc_id.split("-")[0]})

collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)


# ── Retrieve ─────────────────────────────────────────────
def retrieve(query: str, n_results: int = 3, category: str | None = None):
    query_vec = client.embeddings.create(
        model="text-embedding-3-small", input=query
    ).data[0].embedding

    where = {"category": category} if category else None

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=n_results,
        where=where,
        include=["documents", "distances", "metadatas"],
    )
    return list(zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    ))


# Semantic search — no exact keywords needed
for doc, dist, meta in retrieve("Can I return a purchased item?"):
    print(f"[{meta['category']}] dist={dist:.3f}  {doc[:60]}")

# Filter by category
for doc, dist, meta in retrieve("days off work", category="hr"):
    print(f"[hr] dist={dist:.3f}  {doc[:60]}")
```

!!! warning "Distance ≠ Similarity (in some databases)"
    ChromaDB returns *distance* (lower = more similar) when using cosine space. Pinecone returns *score* (higher = more similar). Always check which convention your database uses to avoid sorting in the wrong direction.

---

## Pinecone Example

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="YOUR_KEY")

pc.create_index(
    name="company-docs",
    dimension=1536,                         # must match embedding model
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
)
index = pc.Index("company-docs")

# Upsert (id, vector, metadata)
index.upsert(vectors=[
    ("policy-001", embed("Refunds within 30 days..."), {"text": "Refunds...", "category": "policy"}),
    ("hr-001",     embed("Annual leave: 20 days..."),  {"text": "Annual...",  "category": "hr"}),
])

# Query
results = index.query(
    vector=embed("How many vacation days do I get?"),
    top_k=2,
    include_metadata=True,
    filter={"category": {"$eq": "hr"}},     # metadata filter
)

for match in results["matches"]:
    print(f"score={match['score']:.3f}  {match['metadata']['text'][:60]}")
```

---

## Embedding Model Comparison

| Model | Dimensions | Max tokens | Cost | When to use |
|-------|-----------|------------|------|-------------|
| `text-embedding-3-small` | 1,536 | 8,191 | USD 0.020 / 1M | Default for most RAG |
| `text-embedding-3-large` | 3,072 | 8,191 | USD 0.130 / 1M | Higher precision; worth cost at scale |
| Cohere `embed-v3-english` | 1,024 | 512 | USD 0.100 / 1M | Strong on short documents |
| `sentence-transformers/all-MiniLM-L6-v2` | 384 | 512 | Free (self-host) | Latency-critical, < 1M docs |
| `BAAI/bge-large-en-v1.5` | 1,024 | 512 | Free (self-host) | SOTA open model on MTEB |

**Choosing a model**:
1. Start with `text-embedding-3-small` — cheap, solid quality, 8K token input.
2. If you see retrieval failures on paraphrases → upgrade to `text-embedding-3-large`.
3. If latency or cost is critical → benchmark open models like `bge-large`; host on a GPU instance.

Always evaluate embedding models on your *own domain data* with a held-out query set. MTEB leaderboard rankings are on generic benchmarks; legal, medical, or code domains can invert the rankings.

---

## Edge Cases & Misconceptions

**Misconception: Longer texts produce better embeddings.**
Most embedding models cap input at 512–8,192 tokens and *average* or *pool* sub-token representations. A 10,000-word document collapsed into one vector loses the specificity of any individual fact buried in the middle. This is why we chunk — so each vector represents a focused, coherent unit.

**Misconception: Higher dimensions = better performance.**
Dimensions determine the *capacity* of the representation, not its quality. A `bge-large` model with 1,024 dimensions outperforms some 3,072-dimension models on several benchmarks. Quality depends on training data and fine-tuning, not raw dimension count.

**Misconception: Cosine similarity and dot product are interchangeable.**
Dot product is sensitive to vector magnitude; cosine normalizes for it. If your embedding model produces unit-norm vectors (which most do), they are equivalent. But if you ever fine-tune or concatenate embeddings, verify normalization before switching distance metrics.

**Edge case: Duplicate documents.**
If you index the same document twice (e.g., ingestion pipeline runs twice), you double-count its signal at retrieval time. Use deterministic IDs based on document hash to make upserts idempotent.

---

## Production Connection

In production you will encounter several decisions that don't arise in tutorials:

- **Incremental indexing** — batch-embed new documents nightly rather than re-indexing everything. Keep a `last_indexed_at` timestamp per document.
- **Multi-tenancy** — isolate customer data with metadata filters or separate collections/namespaces. Never let one tenant's data appear in another's search results.
- **Index freshness** — HNSW graphs degrade slightly when you delete many vectors (tombstone bloat). Periodically rebuild the index offline.
- **Dimensionality reduction** — if storage cost is a concern at hundreds of millions of vectors, use Matryoshka Representation Learning (MRL) models that allow truncating to fewer dimensions with minimal quality loss. OpenAI's `text-embedding-3-*` models support this via the `dimensions` parameter.

---

## Key Takeaways

- Embeddings map text to vectors where geometric closeness encodes semantic similarity — the core mechanism enabling semantic search.
- Cosine similarity measures the angle between vectors, ranging from 0 (orthogonal/unrelated) to 1 (identical direction/meaning).
- Vector databases use ANN index structures (HNSW, IVF) to search millions of vectors in milliseconds with ~95% recall.
- HNSW is the best default; IVF trades recall for memory efficiency at very large scale.
- Batch your embedding calls — sending arrays of texts is far more efficient than one-at-a-time API calls.
- Evaluate embedding models on your domain; generic benchmarks don't predict domain-specific retrieval quality.
- Metadata filtering lets you scope semantic search to a subset of documents (e.g., by date, category, or tenant).

---

## Related Papers

| Paper | What it contributes | Link |
|-------|-------------------|------|
| Karpukhin et al. (2020) — *Dense Passage Retrieval for Open-Domain QA* | Shows dense embeddings outperform BM25 on open-domain QA when trained end-to-end | [arxiv.org/abs/2004.04906](https://arxiv.org/abs/2004.04906) |
| Malkov & Yashunin (2018) — *Efficient and Robust Approximate Nearest Neighbor Search Using HNSW* | Introduces the HNSW algorithm used inside most vector databases | [arxiv.org/abs/1603.09320](https://arxiv.org/abs/1603.09320) |
| Muennighoff et al. (2022) — *MTEB: Massive Text Embedding Benchmark* | 56-task benchmark for comparing embedding models; the standard leaderboard | [arxiv.org/abs/2210.07316](https://arxiv.org/abs/2210.07316) |
| Kusupati et al. (2022) — *Matryoshka Representation Learning* | Trains embeddings that remain useful even when truncated to fewer dimensions | [arxiv.org/abs/2205.13147](https://arxiv.org/abs/2205.13147) |

---

## Choosing Between L2 Distance and Cosine Similarity

Most vector databases offer multiple distance metrics. The choice affects retrieval behavior:

**Cosine similarity** measures the angle between vectors, ignoring magnitude. Two vectors pointing in the same direction score 1.0 regardless of their lengths. This is the right default for text embeddings, where the embedding norm is not meaningful — only the direction encodes semantics.

**L2 (Euclidean) distance** measures the straight-line distance between vector tips. Sensitive to magnitude. If your embedding model does not produce unit-norm vectors, L2 can behave unpredictably — a semantically similar vector with a slightly different norm may rank lower than a semantically distant one with the same norm.

**Dot product** is equivalent to cosine when vectors are unit-normalized (which most embedding models guarantee). Some vector databases expose raw dot product for speed; only use it when you've verified your embedding model produces unit-norm outputs.

**Rule of thumb**: Use cosine for text embeddings. Verify which distance metric your chosen database uses by default — ChromaDB defaults to L2 unless you specify `metadata={"hnsw:space": "cosine"}`.

```python
# ChromaDB: explicitly set cosine similarity
collection = chroma.get_or_create_collection(
    name="my_docs",
    metadata={"hnsw:space": "cosine"},  # default is "l2" — always set this
)
```

---

## Further Reading

- [Jay Alammar — The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) — intuition for how neural networks learn word geometry
- [Vector Databases Simply Explained](https://www.youtube.com/watch?v=dN0lsF2cvm4) — Fireship quick overview (7 min)
- [Weaviate: What is a Vector Database?](https://weaviate.io/blog/what-is-a-vector-database) — architecture deep-dive with diagrams
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — current state-of-the-art embedding models

---

## Next Lesson

**[Lesson 3: Chunking Strategies](03-chunking-strategies.md)** — Learn how to split documents into retrieval-sized pieces. Chunk size and overlap are among the most impactful RAG tuning levers.
