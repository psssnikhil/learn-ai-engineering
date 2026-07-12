---
title: Vector Database Schema Design
description: >-
  Learn how to design effective schemas, choose metadata strategies, and
  structure your vector collections for production applications
duration: 35 min
difficulty: intermediate
has_code: false
---
# Vector Database Schema Design

## Learning Objectives

By the end of this lesson, you will be able to:
- Design vector database schemas for different use cases
- Choose effective metadata strategies for filtering and organization
- Handle multi-tenancy and access control in vector databases
- Plan for schema evolution as your application grows

---

## Schema Design Principles

Vector databases are schema-light compared to relational databases, but good design still matters. Your schema consists of three parts:

1. **Vector**: The embedding itself (fixed at index creation)
2. **ID**: Unique identifier for each record
3. **Metadata**: Structured key-value pairs for filtering

```python
# A well-designed vector record
{
    "id": "doc_20240315_annual-report_chunk-42",
    "values": [0.012, -0.087, ...],  # 1536 dimensions
    "metadata": {
        "source": "annual-report-2024.pdf",
        "page": 15,
        "chunk_index": 42,
        "category": "financial",
        "year": 2024,
        "department": "finance",
        "text": "Revenue grew 23% year-over-year..."
    }
}
```

---

## ID Design Strategies

Your ID scheme affects how you update, delete, and deduplicate records.

```python
# Strategy 1: Source-based IDs (best for document pipelines)
def make_doc_id(source: str, chunk_index: int) -> str:
    return f"{source}::chunk-{chunk_index:04d}"
# "annual-report.pdf::chunk-0042"

# Strategy 2: Content-hash IDs (deduplication)
import hashlib

def make_content_id(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]
# "a3f8b2c1e9d04f67"

# Strategy 3: UUID (simple, unique, no collisions)
import uuid

def make_uuid_id() -> str:
    return str(uuid.uuid4())
# "550e8400-e29b-41d4-a716-446655440000"
```

| Strategy | Deduplication | Updatable | Human-readable |
|----------|--------------|-----------|----------------|
| Source-based | Yes (natural) | Yes | Yes |
| Content-hash | Yes (automatic) | Replace only | No |
| UUID | No | No (must track externally) | No |

---

## Metadata Design

### What to Store as Metadata

Store metadata that you will **filter on** at query time. Do not store data you will never query.

```python
# Good metadata: useful for filtering and context
metadata = {
    "text": "The original chunk text",         # For displaying results
    "source": "handbook.pdf",                  # Filter by document
    "page": 12,                                # Reference back to source
    "category": "engineering",                 # Filter by topic
    "created_at": "2024-03-15",               # Filter by date
    "access_level": "internal",               # Access control
}

# Bad metadata: unlikely to be filtered on
metadata = {
    "font_size": 12,                          # Not useful for search
    "paragraph_count": 3,                     # Not useful for search
    "author_middle_name": "James",            # Too granular
    "full_document_text": "...(50KB)...",     # Too large
}
```

### Metadata Size Limits

| Database | Max Metadata per Vector |
|----------|----------------------|
| Pinecone | 40 KB |
| ChromaDB | No hard limit (practical ~1 MB) |
| Weaviate | No hard limit |
| Qdrant | No hard limit |

---

## Multi-Tenancy Patterns

When multiple users or organizations share the same vector database:

### Option 1: Namespace Isolation (Recommended)

```python
# Pinecone: use namespaces
index.upsert(vectors=user_vectors, namespace="tenant-123")
results = index.query(vector=query, namespace="tenant-123", top_k=5)
```

### Option 2: Metadata-Based Isolation

```python
# ChromaDB: filter by tenant
collection.add(
    documents=["User's document"],
    metadatas=[{"tenant_id": "tenant-123"}],
    ids=["doc-1"]
)

results = collection.query(
    query_texts=["search query"],
    where={"tenant_id": "tenant-123"},
    n_results=5,
)
```

### Option 3: Separate Collections

```python
# One collection per tenant
collection = client.get_or_create_collection(f"tenant-{tenant_id}")
```

| Pattern | Isolation | Cost | Complexity |
|---------|-----------|------|-----------|
| Namespace | Good | Low | Low |
| Metadata filter | Moderate | Low | Low |
| Separate collections | Strong | Higher | Medium |

---

## Key Takeaways

- Use source-based IDs for document pipelines to enable easy updates and deduplication
- Only store metadata you will actually filter on at query time
- Store the original text chunk in metadata so you can display results without a separate lookup
- Use namespaces or metadata filters for multi-tenancy -- separate collections for strict isolation
- Plan metadata schema early; changing it later requires re-indexing

## Resources

- [Pinecone: Metadata Filtering](https://docs.pinecone.io/guides/data/filter-with-metadata) -- Official guide on metadata best practices
- [YouTube: Vector DB Design Patterns](https://www.youtube.com/watch?v=J_0pMXEMZk0) -- Architecture patterns for production vector databases
- [Qdrant: Multi-Tenancy Guide](https://qdrant.tech/documentation/guides/multiple-partitions/) -- Multi-tenant patterns in Qdrant

---

Next: Scaling Vector Search
