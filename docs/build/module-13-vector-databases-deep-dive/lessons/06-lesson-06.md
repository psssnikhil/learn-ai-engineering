---
title: Vector Database Schema Design
description: >-
  Learn how to design effective schemas, choose metadata strategies, and
  structure your vector collections for production applications
duration: 45 min
difficulty: intermediate
has_code: true
module: module-13
---
# Vector Database Schema Design

## Prerequisites

Before this lesson you should be comfortable with:

- **Module 13, Lessons 01–05** — embeddings, indexing, Pinecone namespaces, Chroma collections
- **Module 09, Lesson 05** — building a basic RAG system (chunking pipeline)
- **Basic data modeling** — keys, IDs, filtering concepts

Good schema design is the difference between a vector database that scales cleanly and one that requires painful full re-indexes every time requirements change. This lesson covers ID strategies, metadata design, multi-tenant sharding, access control, and schema evolution.

---

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Design vector record schemas for RAG and search use cases | 10 min | Intermediate |
| Choose ID strategies that support updates and deduplication | 10 min | Intermediate |
| Implement multi-tenant isolation with namespaces and sharding | 15 min | Intermediate |
| Plan schema evolution without downtime | 10 min | Intermediate |

---

## Intuition First: The Library Catalog Card

Every book in a library has three things: a **call number** (unique ID), a **subject classification** (metadata for finding it), and the **book itself** (the vector — a compressed representation of content).

Bad cataloging puts the entire book text on the card (metadata bloat), uses random numbers with no relation to location (UUID with no update path), or mixes fiction and legal documents in one drawer (no tenant isolation).

Good cataloging uses predictable call numbers (`2024-annual-report::chunk-0042`), stores only what you'll search by on the card (department, year, access level), and separates collections by tenant or domain.

---

## Schema Components

Every vector record has three parts:

```python
record = {
    "id": "annual-report-2024::chunk-0042",
    "values": [0.012, -0.087, ...],  # 1536 dimensions — fixed at index creation
    "metadata": {
        "text": "Revenue grew 23% year-over-year...",
        "source": "annual-report-2024.pdf",
        "page": 15,
        "chunk_index": 42,
        "category": "financial",
        "year": 2024,
        "tenant_id": "acme",
        "access_level": "internal",
        "embedding_model": "text-embedding-3-small",
        "content_hash": "a3f8b2c1e9d04f67",
        "indexed_at": "2024-03-15T10:30:00Z",
    }
}
```

| Field | Required? | Purpose |
|-------|-----------|---------|
| `id` | Yes | Upsert, delete, deduplication |
| `values` | Yes | ANN search |
| `metadata.text` | Strongly recommended | RAG context injection without second lookup |
| `metadata.source` | Recommended | Filtering, citation, re-index |
| `metadata.tenant_id` | If multi-tenant | Isolation |
| `metadata.content_hash` | Recommended | Change detection, skip re-embed |
| `metadata.embedding_model` | Recommended | Version tracking for re-index |

---

## ID Design Strategies

Your ID scheme determines how easily you update, delete, and deduplicate records.

```python
# Strategy 1: Source-based (best for document pipelines)
def make_doc_id(source: str, chunk_index: int) -> str:
    return f"{source}::chunk-{chunk_index:04d}"
# "annual-report-2024.pdf::chunk-0042"

# Strategy 2: Content-hash (automatic deduplication)
import hashlib
def make_content_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]

# Strategy 3: Composite (tenant + source + chunk)
def make_tenant_doc_id(tenant: str, source: str, chunk_index: int) -> str:
    return f"{tenant}/{source}::chunk-{chunk_index:04d}"
# "acme/handbook.pdf::chunk-0012"
```

| Strategy | Deduplication | Updatable | Human-readable | Multi-tenant |
|----------|--------------|-----------|----------------|--------------|
| Source-based | Yes (by source) | Yes (delete + re-upsert) | Yes | With prefix |
| Content-hash | Automatic | Replace only | No | Needs namespace |
| UUID | No | Hard (must track mapping) | No | Needs namespace |
| Composite | Yes | Yes | Yes | Built-in |

**Recommendation:** Use `{source}::chunk-{index:04d}` for RAG pipelines. Add tenant prefix or namespace for multi-tenancy.

---

## Metadata Design

### Store What You Filter On

```python
# Good: fields you'll query against
metadata = {
    "text": chunk_text,
    "source": "handbook.pdf",
    "page": 12,
    "category": "engineering",
    "department": "platform",
    "created_at": "2024-03-15",
    "access_level": "internal",
    "language": "en",
    "doc_version": 3,
}

# Bad: never filtered, wastes space
metadata = {
    "font_size": 12,
    "paragraph_count": 3,
    "author_middle_name": "James",
    "full_document_text": "...50KB of text...",
}
```

### Metadata Size Limits

| Database | Max Metadata per Vector | Practical Limit |
|----------|------------------------|-----------------|
| Pinecone | 40 KB | Keep under 10 KB |
| ChromaDB | No hard limit | ~1 MB practical |
| Qdrant | No hard limit | Payload indexed separately |
| Weaviate | No hard limit | Schema-defined properties |

Store chunk text (~500–2000 chars) in metadata for RAG. Store full documents in object storage (S3, GCS), not in vector metadata.

### Metadata Schema Template for RAG

```python
RAG_METADATA_SCHEMA = {
    # Required
    "text": str,           # Chunk text for LLM context
    "source": str,         # Original document identifier
    "chunk_index": int,    # Position in document

    # Filtering
    "tenant_id": str,      # Multi-tenant isolation
    "category": str,       # Topic/domain filter
    "access_level": str,   # "public", "internal", "confidential"
    "language": str,       # "en", "es", etc.

    # Operations
    "content_hash": str,   # SHA-256 prefix for change detection
    "embedding_model": str,# Model version for re-index planning
    "indexed_at": str,     # ISO timestamp
    "doc_version": int,    # Increment on document update
}
```

---

## Multi-Tenant Sharding Patterns

When multiple organizations share one vector database, isolation is non-negotiable. A query for Tenant A must never return Tenant B's data.

### Pattern 1: Namespace Isolation (Recommended)

```python
# Pinecone
index.upsert(vectors=tenant_vectors, namespace=f"tenant-{tenant_id}")
results = index.query(vector=query_vec, namespace=f"tenant-{tenant_id}", top_k=5)

# Qdrant — collection per tenant or partition key
client.upsert(collection_name=f"tenant_{tenant_id}", points=points)
```

**Pros:** Hard isolation, no filter leakage, simple mental model.
**Cons:** Many namespaces to manage; cross-tenant analytics requires aggregation.

### Pattern 2: Metadata Filter Isolation

```python
# Every record tagged with tenant_id
collection.add(
    documents=["User's private document"],
    metadatas=[{"tenant_id": "tenant-123", "access_level": "private"}],
    ids=["doc-1"],
)

results = collection.query(
    query_texts=["search query"],
    where={"tenant_id": "tenant-123"},
    n_results=5,
)
```

**Pros:** Single index, simpler ops.
**Cons:** Filter bugs can leak data; every query must include tenant filter.

### Pattern 3: Separate Collections / Indexes

```python
collection = client.get_or_create_collection(f"tenant-{tenant_id}")
```

**Pros:** Strongest isolation, independent scaling.
**Cons:** Highest cost; operational overhead per tenant.

### Multi-Tenant Decision Matrix

| Tenants | Vectors/Tenant | Recommended Pattern |
|---------|---------------|---------------------|
| 1–10 | Any | Separate collections |
| 10–1,000 | < 100K each | Namespaces |
| 1,000+ | < 10K each | Namespaces + metadata filter |
| 1,000+ | > 100K each | Separate indexes per large tenant |

!!! warning "Defense in depth for sensitive data"
    For healthcare, legal, or financial data: use namespace isolation AND metadata `tenant_id` filter AND application-level auth checks. Never rely on a single isolation layer.

---

## Access Control Patterns

```python
def search_with_access_control(index, query_vec, user, top_k=5):
    """Filter results by user's access level."""
    access_filter = {"access_level": {"$in": user.allowed_levels}}

    if user.tenant_id:
        access_filter["tenant_id"] = {"$eq": user.tenant_id}

    return index.query(
        vector=query_vec,
        top_k=top_k,
        filter=access_filter,
        namespace=f"tenant-{user.tenant_id}",
        include_metadata=True,
    )
```

| Access Level | Who Sees It | Filter |
|-------------|-------------|--------|
| `public` | All users | No filter needed |
| `internal` | Authenticated employees | `access_level in ["public", "internal"]` |
| `confidential` | Specific roles | `access_level eq "confidential"` + role check |
| `tenant-private` | Tenant members only | `tenant_id eq user.tenant_id` |

---

## Schema Evolution

Metadata schemas change. New fields get added; old fields become obsolete. Plan for this:

### Additive Changes (Safe)

Adding new optional metadata fields requires no re-index — just start including them on new upserts:

```python
# Old records lack "doc_version"; new records include it
metadata = {**existing_metadata, "doc_version": 2, "language": "en"}
index.upsert(vectors=[{"id": doc_id, "values": embedding, "metadata": metadata}])
```

### Breaking Changes (Require Re-Index)

| Change | Impact | Migration |
|--------|--------|-----------|
| Embedding model change | All vectors invalid | Full re-embed + re-upsert |
| Dimension change | Index incompatible | New index + migrate |
| Rename filter field | Old filters break | Dual-write both fields, then migrate |
| Change ID scheme | Orphan records | Delete old IDs, upsert new |

### Zero-Downtime Migration Pattern

```python
def migrate_to_new_embedding_model(old_index, new_index, documents, new_embed_fn):
    """Dual-write to new index while old index serves traffic."""
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        vectors = [{
            "id": doc["id"],
            "values": new_embed_fn(doc["text"]),
            "metadata": {**doc["metadata"], "embedding_model": "text-embedding-3-large"},
        } for doc in batch]
        new_index.upsert(vectors=vectors)

    # Validate recall on eval set before switching traffic
    # Then: update app config to point to new_index
    # Then: delete old_index after monitoring period
```

---

## Failure Modes

**Metadata schema drift.** Different ingestion pipelines write different field names (`category` vs `topic`). Queries filter on `category` and miss half the corpus. Enforce a schema contract.

**Oversized metadata.** Storing 50 KB chunks when Pinecone limits to 40 KB causes silent upsert failures. Validate metadata size at ingest.

**Missing tenant filter.** A single query without `tenant_id` filter in a metadata-isolated setup returns cross-tenant results. Use middleware that injects tenant filter automatically.

**Non-idempotent IDs.** Random UUIDs on every ingest create duplicate vectors for the same content. Use deterministic source-based IDs.

---

## Production Notes

- Document your metadata schema in a shared config (JSON Schema or Pydantic model) validated at ingest time.
- Store `embedding_model` and `content_hash` on every vector — you'll need them for re-index decisions.
- Use namespaces for tenant boundaries; metadata filters for category/date/access within a tenant.
- Plan embedding model migrations as a formal project: new index, eval validation, traffic switch, old index deletion.

---

## Key Takeaways

- Schema = ID + vector + metadata. Design all three deliberately upfront.
- Source-based IDs (`{file}::chunk-{N}`) enable idempotent updates and deduplication.
- Multi-tenant: namespaces for hard isolation, metadata filters for soft constraints, both for sensitive data.
- Schema evolution: additive changes are free; embedding model or dimension changes require full re-index.
- Validate metadata size and schema at ingest — failures at query time are harder to debug.

---

## Resources

- [Pinecone: Metadata Filtering](https://docs.pinecone.io/guides/data/filter-with-metadata)
- [Qdrant: Multi-Tenancy Guide](https://qdrant.tech/documentation/guides/multiple-partitions/)

---

## Next Lesson

**[Lesson 7: Scaling Vector Search](07-lesson-07.md)** — Sharding, replication, quantization at scale, and latency/throughput optimization for millions of vectors.
