---
title: 'Project 6: Semantic Search Engine'
description: >-
  Build a full-featured semantic search engine with hybrid search, faceted
  filtering, and a web interface
duration: 180 min
difficulty: advanced
has_code: true
module: module-17
---
# Project 6: Semantic Search Engine

## Project Overview

Build a semantic search engine that goes beyond keyword matching to understand the meaning of queries. This applies vector databases, embeddings, hybrid search, and web development skills.

**What you will build:**
- Document indexing with chunking and embedding
- Hybrid search combining BM25 and vector similarity
- Faceted filtering by metadata (date, category, source)
- Web search interface with results ranking

**Estimated time:** 4-6 hours

---

## Implementation

```python
# search_engine.py
import chromadb
from rank_bm25 import BM25Okapi
from openai import OpenAI

client = OpenAI()

class SemanticSearchEngine:
    def __init__(self, collection_name: str = "search-index"):
        self.chroma = chromadb.PersistentClient(path="./search_db")
        self.collection = self.chroma.get_or_create_collection(collection_name)
        self.documents = []
        self.bm25 = None

    def index_documents(self, documents: list[dict]):
        """Index documents with metadata.
        Each doc: {"id": str, "text": str, "title": str, "category": str, ...}
        """
        self.documents = documents
        texts = [d["text"] for d in documents]

        # Build BM25 index for keyword search
        tokenized = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(tokenized)

        # Add to vector database
        self.collection.upsert(
            documents=texts,
            metadatas=[{k: v for k, v in d.items() if k != "text"} for d in documents],
            ids=[d["id"] for d in documents],
        )

    def search(self, query: str, k: int = 10, alpha: float = 0.5,
               filters: dict = None) -> list[dict]:
        """Hybrid search combining BM25 and vector similarity."""
        # Vector search
        vector_results = self.collection.query(
            query_texts=[query],
            n_results=min(k * 2, len(self.documents)),
            where=filters,
        )
        vector_ids = vector_results["ids"][0]

        # BM25 keyword search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranked = sorted(range(len(bm25_scores)),
                            key=lambda i: bm25_scores[i], reverse=True)
        bm25_ids = [self.documents[i]["id"] for i in bm25_ranked[:k*2]]

        # Reciprocal Rank Fusion
        scores = {}
        rrf_k = 60
        for rank, doc_id in enumerate(vector_ids):
            scores[doc_id] = scores.get(doc_id, 0) + alpha / (rrf_k + rank + 1)
        for rank, doc_id in enumerate(bm25_ids):
            scores[doc_id] = scores.get(doc_id, 0) + (1 - alpha) / (rrf_k + rank + 1)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

        # Build result objects
        doc_map = {d["id"]: d for d in self.documents}
        return [
            {**doc_map[doc_id], "score": score}
            for doc_id, score in ranked
            if doc_id in doc_map
        ]

# Usage
engine = SemanticSearchEngine()
engine.index_documents([
    {"id": "1", "text": "RAG systems combine retrieval with generation...", "title": "RAG Overview", "category": "ai"},
    {"id": "2", "text": "Vector databases store embeddings for search...", "title": "Vector DBs", "category": "databases"},
    # ... more documents
])

results = engine.search("How do I search by meaning?", k=5, alpha=0.6)
for r in results:
    print(f"[{r['score']:.4f}] {r['title']}")
```

---

## Extensions and Challenges

- **Web UI**: Build a Flask/FastAPI frontend with search box and faceted filters
- **Autocomplete**: Add query suggestions based on indexed content
- **Analytics**: Track popular searches, zero-result queries, click-through rates
- **Re-ranking**: Add a cross-encoder re-ranker for better precision

## Resources

- [YouTube: Building Search Engines](https://www.youtube.com/watch?v=lYxGYXjfrNI) -- Hybrid search implementation
- [Sentence Transformers Cross-Encoders](https://www.sbert.net/docs/cross_encoder/usage/usage.html) -- Re-ranking for better precision
- [BM25 in Python](https://github.com/dorianbrown/rank_bm25) -- Keyword search implementation

---

Next: Project 7 -- LLM-Powered Data Extraction Pipeline
