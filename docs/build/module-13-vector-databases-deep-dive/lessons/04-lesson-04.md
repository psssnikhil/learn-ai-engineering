---
title: Working with Pinecone
description: >-
  Learn to build and query a production vector database using Pinecone's managed
  service
duration: 50 min
difficulty: intermediate
has_code: true
module: module-13
---
# Working with Pinecone

## Prerequisites

Before this lesson you should be comfortable with:

- **Module 13, Lessons 01–03** — embeddings, ANN indexes, HNSW/IVF concepts
- **Module 09, Lesson 02** — basic vector database operations
- **OpenAI API** — generating embeddings (Lesson 02)

This lesson covers Pinecone-specific production patterns: serverless vs pod indexes, namespace-based multi-tenancy, metadata filtering strategies, batch ingestion at scale, and cost modeling.

---

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Create and configure a Pinecone serverless index | 10 min | Intermediate |
| Implement upsert, query, delete with metadata filters | 15 min | Intermediate |
| Design namespace isolation for multi-tenant RAG | 10 min | Intermediate |
| Build a batch ingestion pipeline with error handling | 15 min | Intermediate |

---

## Intuition First: The Managed Warehouse

Self-hosting a vector database is like owning a warehouse: you buy land, hire staff, maintain forklifts, and worry about expansion. Pinecone is like renting space in a fully automated fulfillment center — you drop off boxes (vectors), specify what you need retrieved (queries), and the infrastructure handles shelving, scaling, and maintenance.

You pay for what you store and query. You don't configure HNSW `M` or `ef` directly — Pinecone tunes ANN parameters internally. Your engineering focus shifts to **schema design, metadata filters, namespace isolation, and ingestion pipelines**.

---

## Pinecone Architecture Overview

| Component | Purpose |
|-----------|---------|
| **Index** | Top-level container; defines dimension, metric, and hosting spec |
| **Namespace** | Logical partition within an index (multi-tenancy, environments) |
| **Vector** | ID + values + metadata |
| **Metadata filter** | Pre/post-filter on structured fields during search |

### Serverless vs Pod Indexes

| Feature | Serverless | Pod-based |
|---------|-----------|-----------|
| **Scaling** | Automatic | Manual pod sizing |
| **Cost model** | Pay per read/write/storage | Pay per pod hour |
| **Best for** | Variable traffic, prototyping → production | Sustained high QPS, predictable load |
| **Max vectors** | Millions (per index) | Billions (with scaling) |
| **Setup** | Instant | Pod provisioning (~minutes) |

For most RAG applications starting out, **serverless** is the right choice.

---

## Step-by-Step: Build a Knowledge Base Index

### Step 1: Create the Index

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="your-api-key")

# Only create if it doesn't exist
if "knowledge-base" not in pc.list_indexes().names():
    pc.create_index(
        name="knowledge-base",
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index("knowledge-base")
```

### Step 2: Prepare and Upsert Documents

```python
from openai import OpenAI

openai_client = OpenAI()

def get_embedding(text: str) -> list[float]:
    response = openai_client.embeddings.create(
        input=text, model="text-embedding-3-small"
    )
    return response.data[0].embedding

documents = [
    {
        "id": "doc-1",
        "text": "RAG systems combine retrieval with generation for accurate answers",
        "metadata": {"category": "rag", "source": "textbook", "year": 2024, "tenant_id": "acme"},
    },
    {
        "id": "doc-2",
        "text": "Fine-tuning adapts pre-trained models to specific domains",
        "metadata": {"category": "fine-tuning", "source": "blog", "year": 2024, "tenant_id": "acme"},
    },
    {
        "id": "doc-3",
        "text": "Vector databases enable semantic search over embeddings",
        "metadata": {"category": "vector-db", "source": "docs", "year": 2023, "tenant_id": "globex"},
    },
]

vectors = []
for doc in documents:
    vectors.append({
        "id": doc["id"],
        "values": get_embedding(doc["text"]),
        "metadata": {**doc["metadata"], "text": doc["text"]},
    })

index.upsert(vectors=vectors, namespace="production")
print(f"Upserted {len(vectors)} vectors")
```

### Step 3: Query with Filters

```python
query = "How do retrieval systems work?"
query_embedding = get_embedding(query)

results = index.query(
    vector=query_embedding,
    top_k=5,
    include_metadata=True,
    namespace="production",
    filter={"category": {"$eq": "rag"}},
)

for match in results["matches"]:
    print(f"Score: {match['score']:.3f} | {match['metadata']['text'][:80]}")
```

---

## Metadata Filtering

Metadata filters are Pinecone's superpower for production RAG. They let you combine semantic search with structured constraints.

### Filter Operators

| Operator | Meaning | Example |
|----------|---------|---------|
| `$eq` | Equals | `{"category": {"$eq": "rag"}}` |
| `$ne` | Not equals | `{"source": {"$ne": "blog"}}` |
| `$gt`, `$gte` | Greater than (or equal) | `{"year": {"$gte": 2024}}` |
| `$lt`, `$lte` | Less than (or equal) | `{"year": {"$lt": 2023}}` |
| `$in` | In list | `{"category": {"$in": ["rag", "vector-db"]}}` |
| `$nin` | Not in list | `{"category": {"$nin": ["archived"]}}` |

### Combining Filters

```python
results = index.query(
    vector=query_embedding,
    top_k=10,
    namespace="production",
    filter={
        "$and": [
            {"tenant_id": {"$eq": "acme"}},
            {"year": {"$gte": 2024}},
            {"category": {"$in": ["rag", "vector-db"]}},
        ]
    },
    include_metadata=True,
)
```

!!! warning "Over-aggressive filters shrink the candidate pool"
    If your filter reduces the searchable set to 50 vectors, ANN search quality degrades. Prefer namespace isolation for hard boundaries (tenant, environment) and metadata filters for soft constraints (category, date).

---

## Multi-Tenant Namespaces

Namespaces provide hard isolation without separate indexes:

```python
TENANTS = ["acme", "globex", "initech"]

for tenant in TENANTS:
    tenant_docs = load_documents_for_tenant(tenant)
    vectors = embed_and_format(tenant_docs)
    index.upsert(vectors=vectors, namespace=f"tenant-{tenant}")

# Query scoped to one tenant — no cross-tenant leakage possible
results = index.query(
    vector=get_embedding("refund policy"),
    top_k=5,
    namespace="tenant-acme",
    include_metadata=True,
)
```

| Isolation Pattern | Isolation Strength | Cost | When to Use |
|-------------------|-------------------|------|-------------|
| **Namespace per tenant** | Strong | Single index cost | 10–1000 tenants |
| **Separate indexes** | Strongest | Multiple index costs | Strict compliance, very large tenants |
| **Metadata filter only** | Moderate | Lowest | Dev/staging, low isolation needs |

Lesson 6 covers schema design patterns in depth; namespaces are Pinecone's recommended multi-tenant approach.

---

## Batch Ingestion Pipeline

Production ingestion must handle rate limits, retries, and progress tracking:

```python
import time
from typing import Callable

def batch_upsert(
    index,
    documents: list[dict],
    embed_fn: Callable,
    namespace: str = "",
    batch_size: int = 100,
    max_retries: int = 3,
):
    """Upsert documents with retry logic and progress tracking."""
    total = len(documents)
    upserted = 0

    for i in range(0, total, batch_size):
        batch = documents[i:i + batch_size]
        vectors = []
        for doc in batch:
            vectors.append({
                "id": doc["id"],
                "values": embed_fn(doc["text"]),
                "metadata": {**doc.get("metadata", {}), "text": doc["text"]},
            })

        for attempt in range(max_retries):
            try:
                index.upsert(vectors=vectors, namespace=namespace)
                upserted += len(vectors)
                print(f"Progress: {upserted}/{total} ({100*upserted/total:.0f}%)")
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                print(f"Retry {attempt+1} after {wait}s: {e}")
                time.sleep(wait)

    return upserted
```

### Delete and Update Patterns

```python
# Delete specific vectors
index.delete(ids=["doc-1", "doc-2"], namespace="production")

# Delete by metadata filter (e.g., re-index a document)
index.delete(
    filter={"source": {"$eq": "outdated-handbook.pdf"}},
    namespace="production",
)

# Document update = delete old chunks + upsert new ones
def update_document(index, doc_id: str, new_chunks: list[dict], namespace: str):
    index.delete(filter={"doc_id": {"$eq": doc_id}}, namespace=namespace)
    index.upsert(vectors=new_chunks, namespace=namespace)

# Index statistics
stats = index.describe_index_stats()
print(f"Total vectors: {stats.total_vector_count}")
print(f"Namespaces: {stats.namespaces}")
```

---

## Query Patterns for RAG

Production RAG systems use three query patterns beyond simple top-K search:

```python
# Pattern 1: Metadata-constrained RAG retrieval
def rag_retrieve(index, query: str, tenant_id: str, category: str = None, k: int = 5):
    filters = {"tenant_id": {"$eq": tenant_id}}
    if category:
        filters["category"] = {"$eq": category}
    return index.query(
        vector=get_embedding(query),
        top_k=k,
        filter=filters,
        namespace=f"tenant-{tenant_id}",
        include_metadata=True,
    )

# Pattern 2: Fetch by ID (for citation verification)
def fetch_chunks_by_id(index, chunk_ids: list[str], namespace: str):
    return index.fetch(ids=chunk_ids, namespace=namespace)

# Pattern 3: Scroll/list for re-indexing or audit
def list_all_doc_ids(index, namespace: str, prefix: str = "") -> list[str]:
    """Paginate through vectors for migration or audit."""
    all_ids = []
    pagination_token = None
    while True:
        result = index.list_paginated(
            prefix=prefix, namespace=namespace, pagination_token=pagination_token, limit=100
        )
        all_ids.extend(v.id for v in result.vectors)
        if not result.pagination.next:
            break
        pagination_token = result.pagination.next
    return all_ids
```

Use Pattern 1 for every RAG query. Pattern 2 supports citation verification ("does chunk X actually say that?"). Pattern 3 supports migrations and compliance audits.

---

## Cost Modeling

| Cost Component | Serverless Pricing (approx) | Example (1M vectors, 1536d) |
|----------------|---------------------------|----------------------------|
| **Storage** | ~$0.25/GB/month | ~$1.50/month (6 GB raw) |
| **Read units** | Per query | 10K queries/day ≈ $5-15/month |
| **Write units** | Per upsert | One-time ingest ≈ $2-5 |
| **Embedding API** | OpenAI $0.02/1M tokens | 1M chunks ≈ $10 one-time |

Compare against self-hosted Qdrant on a $50/month VPS for 1M vectors — Pinecone wins on zero-ops; self-hosted wins on cost at sustained scale.

---

## Failure Modes

**Dimension mismatch.** Index created with `dimension=1536` but embedding model returns 3072 floats. Every upsert and query fails. Validate at pipeline startup.

**Namespace omission.** Querying without a namespace when data was upserted into one returns zero results silently. Always pass namespace explicitly in multi-tenant setups.

**Metadata bloat.** Pinecone limits metadata to 40 KB per vector. Storing full document text is fine for chunks; storing entire PDFs in metadata is not.

**Stale index after model change.** Switching embedding models without re-indexing produces random similarity scores. Version your model and plan full re-index (Lesson 10).

**Rate limit during bulk ingest.** Default Pinecone limits may throttle large upserts. Use batch upsert with exponential backoff (shown above).

---

## Production Notes

- **Separate namespaces for dev/staging/prod.** Never query production data from a test script accidentally.
- **Store `text` in metadata** for RAG context injection — avoids a second database lookup.
- **Use source-based IDs** like `{filename}::chunk-{index:04d}` for idempotent re-ingestion.
- **Monitor `describe_index_stats()`** after bulk operations to verify vector counts.
- **Set up alerts** on query latency p95 > 200ms and zero-result rate > 5%.

---

## Key Takeaways

- Pinecone abstracts ANN tuning — focus on schema, namespaces, and ingestion quality.
- Namespaces provide strong multi-tenant isolation without separate indexes.
- Metadata filters combine semantic search with structured constraints; don't over-filter.
- Batch upsert with retry logic is essential for production ingestion pipelines.
- Cost scales with storage + queries; model against your expected QPS before committing.

---

## Resources

- [Pinecone Documentation](https://docs.pinecone.io/)
- [Pinecone: Metadata Filtering Guide](https://docs.pinecone.io/guides/data/filter-with-metadata)
- [Pinecone Examples (GitHub)](https://github.com/pinecone-io/examples)

---

## Next Lesson

**[Lesson 5: ChromaDB and Open-Source Vector Databases](05-lesson-05.md)** — Chroma, Qdrant, and Weaviate compared; build a complete document search app with hybrid-ready architecture.
