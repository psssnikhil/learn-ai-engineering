---
title: Production Vector Database Patterns
description: >-
  Learn battle-tested patterns for running vector databases in production,
  including data pipelines, caching, and operational best practices
duration: 40 min
difficulty: advanced
has_code: false
---
# Production Vector Database Patterns

## Learning Objectives

By the end of this lesson, you will be able to:
- Design a production data ingestion pipeline for vector databases
- Implement caching strategies for vector search
- Handle updates and deletions in a vector database
- Apply operational best practices for reliability and cost control

---

## Data Ingestion Pipeline

A production pipeline continuously processes new documents into your vector database:

```
New Documents → Chunking → Embedding → Upsert → Vector DB
      ↑                                            │
      └──── Change detection (skip unchanged) ◄────┘
```

```python
import hashlib
from datetime import datetime

class VectorIngestionPipeline:
    def __init__(self, index, embed_fn, chunk_fn):
        self.index = index
        self.embed_fn = embed_fn
        self.chunk_fn = chunk_fn
        self.processed_hashes = set()  # Track what we've already indexed

    def ingest_document(self, doc_id: str, text: str, metadata: dict):
        """Process a single document into the vector database."""
        # Skip if content hasn't changed
        content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        if content_hash in self.processed_hashes:
            return 0

        # Delete old chunks for this document
        self.index.delete(filter={"doc_id": {"$eq": doc_id}})

        # Chunk the document
        chunks = self.chunk_fn(text)

        # Embed and upsert
        vectors = []
        for i, chunk in enumerate(chunks):
            embedding = self.embed_fn(chunk)
            vectors.append({
                "id": f"{doc_id}::chunk-{i:04d}",
                "values": embedding,
                "metadata": {
                    **metadata,
                    "doc_id": doc_id,
                    "chunk_index": i,
                    "text": chunk,
                    "content_hash": content_hash,
                    "indexed_at": datetime.utcnow().isoformat(),
                }
            })

        # Batch upsert
        for batch_start in range(0, len(vectors), 100):
            batch = vectors[batch_start:batch_start + 100]
            self.index.upsert(vectors=batch)

        self.processed_hashes.add(content_hash)
        return len(vectors)
```

---

## Caching Vector Search Results

Embedding API calls and vector searches both cost time and money. Cache aggressively:

```python
from functools import lru_cache
import json

class CachedVectorSearch:
    def __init__(self, index, embed_fn, cache_size=1000):
        self.index = index
        self.embed_fn = embed_fn
        self._cache = {}
        self.cache_size = cache_size

    def _cache_key(self, query: str, filters: dict, k: int) -> str:
        return hashlib.md5(
            f"{query}:{json.dumps(filters, sort_keys=True)}:{k}".encode()
        ).hexdigest()

    def search(self, query: str, k: int = 5, filters: dict = None):
        cache_key = self._cache_key(query, filters or {}, k)

        if cache_key in self._cache:
            return self._cache[cache_key]

        embedding = self.embed_fn(query)
        results = self.index.query(
            vector=embedding,
            top_k=k,
            filter=filters,
            include_metadata=True
        )

        if len(self._cache) >= self.cache_size:
            # Evict oldest entry
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        self._cache[cache_key] = results
        return results
```

---

## Handling Updates and Deletions

### Document Update Strategy

```python
def update_document(index, doc_id: str, new_text: str, embed_fn, chunk_fn):
    """Update a document: delete old chunks, insert new ones."""
    # Step 1: Delete all existing chunks for this document
    index.delete(filter={"doc_id": {"$eq": doc_id}})

    # Step 2: Re-chunk and re-embed the new content
    chunks = chunk_fn(new_text)
    vectors = []
    for i, chunk in enumerate(chunks):
        vectors.append({
            "id": f"{doc_id}::chunk-{i:04d}",
            "values": embed_fn(chunk),
            "metadata": {"doc_id": doc_id, "chunk_index": i, "text": chunk}
        })

    # Step 3: Upsert new chunks
    index.upsert(vectors=vectors)
```

### Soft Delete Pattern

```python
def soft_delete(index, doc_id: str):
    """Mark document as deleted without removing vectors."""
    # Fetch all chunks
    results = index.query(
        vector=[0] * 1536,  # dummy vector
        top_k=1000,
        filter={"doc_id": {"$eq": doc_id}},
    )
    # Update metadata to mark as deleted
    for match in results["matches"]:
        index.update(
            id=match["id"],
            set_metadata={"deleted": True}
        )
    # Exclude deleted docs from future queries
    # Always include: {"deleted": {"$ne": True}} in query filters
```

---

## Operational Best Practices

| Practice | Why |
|----------|-----|
| **Backup metadata externally** | Vector DBs are not your source of truth for metadata |
| **Monitor index size** | Costs scale with vector count and dimensions |
| **Set up alerts for latency** | Catch performance degradation early |
| **Use namespaces for environments** | Separate dev/staging/prod data |
| **Version your embedding model** | Changing models requires full re-index |
| **Track embedding costs** | OpenAI embeddings cost $0.02-$0.13 per 1M tokens |
| **Implement circuit breakers** | Graceful degradation when the vector DB is down |

---

## Cost Optimization

| Optimization | Savings | Trade-off |
|-------------|---------|-----------|
| Reduce dimensions (1536 → 256) | 6x memory, lower cost | Slight recall reduction |
| Use quantization | 4-32x memory | ~2-5% recall loss |
| Cache embeddings | Reduce API calls 50-80% | Cache staleness |
| Batch upserts | Fewer API calls | Slight ingestion delay |
| Delete unused vectors | Direct cost reduction | Need tracking system |

---

## Key Takeaways

- Build a pipeline that handles chunking, embedding, and upserting with change detection
- Cache both embeddings and search results to reduce costs and latency
- Use the delete-then-reinsert pattern for document updates
- Your vector database is not your source of truth -- always keep the original documents
- Monitor costs, latency, and search quality continuously in production

## Resources

- [YouTube: Vector DB in Production](https://www.youtube.com/watch?v=OATCgQtNX2o) -- Real-world deployment patterns
- [Pinecone: Best Practices](https://docs.pinecone.io/guides/getting-started/overview) -- Production deployment guide
- [LangChain: Vector Store Integration](https://python.langchain.com/docs/integrations/vectorstores/) -- Framework integration patterns

---

## Module Complete!

You have completed the Vector Databases Deep Dive module. You can now design, build, and operate vector search systems for production AI applications.
