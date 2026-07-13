---
title: ChromaDB and Open-Source Vector Databases
description: >-
  Build vector search applications using ChromaDB, Weaviate, and other
  open-source alternatives to managed services
duration: 50 min
difficulty: intermediate
has_code: true
module: module-13
---
# ChromaDB and Open-Source Vector Databases

## Prerequisites

Before this lesson you should be comfortable with:

- **Module 13, Lessons 01–04** — embeddings, indexing, Pinecone patterns
- **Module 09, Lesson 02** — basic Chroma usage
- **Python file I/O** — for the document search build below

Module 09 showed Chroma in 20 lines. This lesson builds a production-shaped document search app, compares Qdrant and Weaviate for hybrid search and sharding, and gives you a decision framework for open-source vs managed.

---

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Build a persistent ChromaDB collection with custom embeddings | 10 min | Intermediate |
| Implement filtered search and metadata strategies | 10 min | Intermediate |
| Compare Chroma, Qdrant, Weaviate, and Milvus for production | 10 min | Intermediate |
| Build a complete document search app from a directory of files | 20 min | Intermediate |

---

## Intuition First: Build vs Buy

Pinecone (Lesson 4) is like renting a fully managed apartment — move in, pay rent, maintenance included. Open-source vector databases are like buying a house — lower long-term cost, full control, but you handle plumbing (sharding), electrical (replication), and renovations (upgrades).

The open-source ecosystem has matured rapidly. Qdrant and Weaviate now match managed services on features like hybrid search, quantization, and multi-tenancy — at the cost of operational complexity.

Choose open-source when: you have infra expertise, need data residency, want hybrid search without vendor lock-in, or operate at scale where managed pricing exceeds self-host cost.

---

## ChromaDB: Quick Recap and Production Patterns

### Persistent Client with Custom Embeddings

```python
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

client = chromadb.PersistentClient(path="./chroma_db")

openai_ef = OpenAIEmbeddingFunction(
    api_key="your-api-key",
    model_name="text-embedding-3-small",
)

collection = client.get_or_create_collection(
    name="documents",
    embedding_function=openai_ef,
    metadata={"hnsw:space": "cosine"},
)

collection.add(
    documents=[
        "RAG combines retrieval with generation for grounded answers",
        "Vector databases store embeddings for similarity search",
        "Fine-tuning customizes models for specific tasks",
    ],
    metadatas=[
        {"category": "rag", "difficulty": "intermediate"},
        {"category": "vector-db", "difficulty": "beginner"},
        {"category": "training", "difficulty": "advanced"},
    ],
    ids=["doc-1", "doc-2", "doc-3"],
)
```

### Filtered Search

```python
results = collection.query(
    query_texts=["beginner AI concepts"],
    n_results=5,
    where={"difficulty": "beginner"},
)

results = collection.query(
    query_texts=["search techniques"],
    n_results=5,
    where={"$and": [
        {"category": {"$in": ["rag", "vector-db"]}},
        {"difficulty": {"$ne": "advanced"}},
    ]},
    where_document={"$contains": "retrieval"},
)
```

### Chroma Limitations for Production

| Limitation | Impact | Workaround |
|------------|--------|------------|
| No native hybrid search | Misses exact keyword matches | Add BM25 layer externally (Lesson 8) |
| Single-node only | No horizontal scaling | Migrate to Qdrant/Weaviate at scale |
| Limited multi-tenancy | Metadata filter only | Separate collections per tenant |
| HNSW params not exposed | Can't tune ef/M | Accept defaults or switch DB |

Chroma excels for prototyping and workloads under ~500K vectors. Plan migration before you hit scaling walls.

---

## Qdrant: High-Performance Self-Hosted Option

Qdrant (Rust) offers production features Chroma lacks: hybrid search, quantization, sharding, and gRPC API.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(url="http://localhost:6333")

client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

points = [
    PointStruct(
        id=i,
        vector=embedding,
        payload={"text": doc["text"], "category": doc["category"]},
    )
    for i, (doc, embedding) in enumerate(zip(documents, embeddings))
]
client.upsert(collection_name="documents", points=points)

results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    limit=5,
    query_filter={"must": [{"key": "category", "match": {"value": "rag"}}]},
)
```

### Qdrant Quantization (Memory Savings)

```python
from qdrant_client.models import ScalarQuantization, ScalarQuantizationConfig, ScalarType

client.update_collection(
    collection_name="documents",
    quantization_config=ScalarQuantization(
        scalar=ScalarQuantizationConfig(type=ScalarType.INT8, always_ram=True)
    ),
)
# 4× memory reduction with ~1% recall loss
```

---

## Weaviate: Hybrid Search Built-In

Weaviate combines BM25 keyword search and vector search natively — critical for production RAG (Lesson 8):

```python
import weaviate

client = weaviate.connect_to_local()
documents = client.collections.get("Documents")

results = documents.query.hybrid(
    query="Python TypeError debugging",
    alpha=0.5,
    limit=10,
)
```

---

## Open-Source Comparison Matrix

| Feature | ChromaDB | Qdrant | Weaviate | Milvus |
|---------|----------|--------|----------|--------|
| **Best for** | Prototyping | High-perf production | Hybrid search | Enterprise scale |
| **Language** | Python | Rust | Go | Go/C++ |
| **Hybrid search** | No | Yes (sparse vectors) | Yes (BM25) | Yes |
| **Quantization** | No | Scalar + PQ | PQ | Scalar + PQ |
| **Multi-tenancy** | Collections | Sharding + payload filter | Multi-tenancy module | Partitions |
| **Max scale** | ~1M vectors | Billions | Billions | Billions |
| **Setup complexity** | Very low | Low | Medium | High |
| **Cloud option** | Chroma Cloud | Qdrant Cloud | Weaviate Cloud | Zilliz Cloud |

---

## Step-by-Step Build: Document Search App

Build a complete semantic search tool over a directory of text files:

```python
"""Production-shaped document search with ChromaDB."""
import hashlib
import chromadb
from pathlib import Path
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CHUNK_SIZE = 500  # characters
CHUNK_OVERLAP = 50

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks

def content_id(source: str, chunk_idx: int) -> str:
    return f"{source}::chunk-{chunk_idx:04d}"

def build_index(docs_dir: str, db_path: str = "./search_db") -> chromadb.Collection:
    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection("file-search", embedding_function=ef)

    documents, metadatas, ids = [], [], []

    for filepath in sorted(Path(docs_dir).glob("**/*.txt")):
        text = filepath.read_text(encoding="utf-8")
        content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({
                "filename": filepath.name,
                "source_path": str(filepath),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "content_hash": content_hash,
            })
            ids.append(content_id(filepath.stem, i))

    if documents:
        # Upsert in batches of 100
        for i in range(0, len(documents), 100):
            collection.upsert(
                documents=documents[i:i+100],
                metadatas=metadatas[i:i+100],
                ids=ids[i:i+100],
            )

    print(f"Indexed {len(documents)} chunks from {docs_dir}")
    return collection

def search(collection, query: str, n: int = 5, filename_filter: str = None):
    where = {"filename": filename_filter} if filename_filter else None
    results = collection.query(query_texts=[query], n_results=n, where=where)

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = 1 - dist  # cosine distance → similarity
        print(f"\n--- {meta['filename']} (chunk {meta['chunk_index']}, score: {score:.3f}) ---")
        print(doc[:300] + "..." if len(doc) > 300 else doc)

# Usage
# collection = build_index("./my_documents")
# search(collection, "How do I configure HNSW parameters?")
```

### Incremental Re-Index

```python
def reindex_if_changed(collection, filepath: Path):
    """Only re-index files whose content hash changed."""
    text = filepath.read_text(encoding="utf-8")
    new_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

    existing = collection.get(where={"filename": filepath.name}, include=["metadatas"])
    if existing["metadatas"] and existing["metadatas"][0].get("content_hash") == new_hash:
        print(f"Skipping {filepath.name} — unchanged")
        return

    # Delete old chunks for this file
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    # Re-index
    chunks = chunk_text(text)
    collection.upsert(
        documents=chunks,
        metadatas=[{
            "filename": filepath.name,
            "chunk_index": i,
            "content_hash": new_hash,
        } for i in range(len(chunks))],
        ids=[content_id(filepath.stem, i) for i in range(len(chunks))],
    )
    print(f"Re-indexed {filepath.name} ({len(chunks)} chunks)")
```

---

## Milvus: Enterprise-Scale Option

For billion-vector deployments, Milvus offers distributed sharding, GPU-accelerated search, and DiskANN:

```python
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

connections.connect(uri="http://localhost:19530")

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=128),
]
schema = CollectionSchema(fields, description="Document embeddings")
collection = Collection("documents", schema)

index_params = {
    "index_type": "IVF_PQ",
    "metric_type": "COSINE",
    "params": {"nlist": 1024, "m": 48, "nbits": 8},
}
collection.create_index("embedding", index_params)
collection.load()

# Search
results = collection.search(
    data=[query_embedding],
    anns_field="embedding",
    param={"nprobe": 32},
    limit=10,
    output_fields=["text", "category"],
)
```

Milvus adds operational complexity but handles scale no single-node database can. Consider it when you exceed 50M vectors or need GPU acceleration.

---

## When to Choose Open-Source vs Managed

| Factor | Open-Source (Qdrant/Weaviate) | Managed (Pinecone) |
|--------|------------------------------|-------------------|
| **Team has DevOps capacity** | Yes → strong fit | No → strong fit |
| **Need hybrid search** | Weaviate/Qdrant native | Requires external BM25 |
| **Data residency requirements** | Full control | Check region availability |
| **Scale > 10M vectors** | Cost-effective self-hosted | Pod pricing may exceed VPS |
| **Time to first prototype** | Chroma: minutes | Pinecone: minutes |
| **Zero-ops requirement** | No | Yes |

---

## Failure Modes

**Chroma collection recreation.** Calling `create_collection` with different embedding functions silently produces inconsistent vectors. Always use `get_or_create_collection` with the same config.

**Path-based persistence.** Chroma's `PersistentClient(path=...)` stores data locally. Docker containers without volume mounts lose data on restart.

**Default embedding model drift.** Chroma's default embedding model may change between versions. Pin an explicit `embedding_function`.

**No backup strategy.** Open-source databases don't auto-backup. Snapshot the data directory (Chroma) or use Qdrant's snapshot API.

---

## Production Notes

- Start with Chroma for prototyping; migrate to Qdrant or Weaviate when you need hybrid search, quantization, or > 500K vectors.
- Pin embedding model and database version in your deployment config.
- Use content-hash-based incremental indexing to avoid re-embedding unchanged documents.
- Run Qdrant or Weaviate in Docker with persistent volumes from day one — not just Chroma in-memory.

---

## Key Takeaways

- Chroma is the fastest path to a working prototype; Qdrant and Weaviate are production-grade open-source alternatives.
- The document search app above is a reusable template: chunk → embed → upsert → query with metadata.
- Incremental re-indexing by content hash saves embedding API costs on unchanged files.
- Choose open-source when you need hybrid search, data control, or cost efficiency at scale.
- Plan your migration path before Chroma's single-node limits become a blocker.

---

## Resources

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Weaviate Hybrid Search](https://weaviate.io/developers/weaviate/search/hybrid)

---

## Next Lesson

**[Lesson 6: Vector Database Schema Design](06-lesson-06.md)** — ID strategies, metadata design, multi-tenant sharding patterns, and schema evolution without full re-index.
