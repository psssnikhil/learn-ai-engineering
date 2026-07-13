---
title: 'Project 5: Semantic Search Engine'
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

Build a semantic search engine that goes beyond keyword matching to understand the meaning of queries. Unlike a RAG chatbot that generates answers, this project focuses on **retrieval quality** — returning the right documents ranked by relevance, with faceted filtering and a polished search interface.

**Time estimate**: 10-15 hours
**Skills used**: Embeddings, Vector Databases, Hybrid Search, FastAPI, Web UI

---

## Prerequisites

Complete these modules before starting:

| Module | Topics Used |
|--------|------------|
| **Module 3: RAG Systems** | Chunking, embeddings, vector retrieval |
| **Module 4: Vector Databases** | ChromaDB indexing and querying |
| **Module 7: Prompt Engineering** | Query understanding (optional re-ranking) |
| **Module 13: LLMOps** | API deployment and monitoring |

**Environment setup:**

```bash
pip install chromadb openai rank-bm25 fastapi uvicorn pydantic python-dotenv pytest httpx jinja2
```

You also need an OpenAI API key for embeddings. ChromaDB can use its built-in embedding function, but we use OpenAI for consistency with production systems.

---

## What You'll Build

### Acceptance Criteria Checklist

- [ ] Index at least 50 documents with metadata (title, category, date, source)
- [ ] Hybrid search combining BM25 keyword scores and vector similarity
- [ ] Reciprocal Rank Fusion (RRF) to merge ranked lists
- [ ] Faceted filtering by category, date range, and source
- [ ] FastAPI backend with `/search`, `/index`, and `/facets` endpoints
- [ ] Simple web UI with search box, result cards, and filter sidebar
- [ ] Evaluation script measuring recall@5 and MRR on a labeled test set
- [ ] Average search latency under 500ms for 1,000-document index

---

## Architecture

```
[Web Search UI]
    |
    |  query + filters (category, date)
    v
[FastAPI Server]
    |-- POST /search   (query -> ranked results)
    |-- POST /index    (documents -> indexed)
    |-- GET  /facets   (available filter values)
    v
[Search Engine Core]
    |
    |-- Query preprocessing (lowercase, tokenize)
    |
    +-- Vector Search Path
    |       |-- Embed query (OpenAI text-embedding-3-small)
    |       |-- ChromaDB similarity search
    |       +-- Top-K vector results with scores
    |
    +-- Keyword Search Path
    |       |-- BM25Okapi over tokenized corpus
    |       +-- Top-K keyword results with scores
    |
    +-- Fusion Layer
    |       |-- Reciprocal Rank Fusion (alpha-weighted)
    |       +-- Apply metadata filters
    |
    +-- Optional Re-ranker
            +-- Cross-encoder or LLM re-rank top 20
    v
[Ranked Results + Metadata]
```

---

## Step 1: Document Indexing Pipeline

Start with a clean indexing layer that handles chunking, metadata, and dual-index construction.

```python
# src/indexer.py
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

@dataclass
class SearchDocument:
    id: str
    text: str
    title: str
    category: str
    source: str
    date: str = field(default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%d"))

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start = end - overlap
    return chunks

def load_corpus(directory: str) -> list[SearchDocument]:
    """Load all .txt and .md files from a directory."""
    documents = []
    for filename in sorted(os.listdir(directory)):
        if not filename.endswith((".txt", ".md")):
            continue
        filepath = os.path.join(directory, filename)
        with open(filepath, "r") as f:
            text = f.read()
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            documents.append(SearchDocument(
                id=f"{filename}__chunk_{i}",
                text=chunk,
                title=filename.replace(".txt", "").replace(".md", "").replace("_", " ").title(),
                category=_infer_category(filename),
                source=filename,
            ))
    return documents

def _infer_category(filename: str) -> str:
    """Simple category inference from filename prefix."""
    prefix = filename.split("_")[0].lower()
    categories = {"ai": "AI", "db": "Databases", "rag": "AI", "ml": "Machine Learning"}
    return categories.get(prefix, "General")
```

---

## Step 2: Build the Hybrid Search Engine

This is the core of the project — dual retrieval with RRF fusion.

```python
# src/search_engine.py
import chromadb
from rank_bm25 import BM25Okapi
from openai import OpenAI
from typing import Optional

client = OpenAI()

class SemanticSearchEngine:
    def __init__(self, collection_name: str = "search-index", persist_dir: str = "./search_db"):
        self.chroma = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.chroma.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.documents: list[dict] = []
        self.bm25: Optional[BM25Okapi] = None
        self._id_to_index: dict[str, int] = {}

    def _embed(self, texts: list[str]) -> list[list[float]]:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [item.embedding for item in response.data]

    def index_documents(self, documents: list[dict]) -> int:
        """Index documents with metadata. Each doc: {id, text, title, category, source, date}."""
        self.documents = documents
        texts = [d["text"] for d in documents]

        # Build BM25 keyword index
        tokenized = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(tokenized)
        self._id_to_index = {d["id"]: i for i, d in enumerate(documents)}

        # Index into ChromaDB with explicit embeddings
        embeddings = self._embed(texts)
        self.collection.upsert(
            documents=texts,
            embeddings=embeddings,
            metadatas=[{k: v for k, v in d.items() if k not in ("text", "id")} for d in documents],
            ids=[d["id"] for d in documents],
        )
        return len(documents)

    def search(
        self,
        query: str,
        k: int = 10,
        alpha: float = 0.6,
        filters: Optional[dict] = None,
    ) -> list[dict]:
        """Hybrid search: vector + BM25 fused with Reciprocal Rank Fusion."""
        if not self.documents or self.bm25 is None:
            return []

        # Apply metadata filters to candidate set
        candidate_ids = self._apply_filters(filters) if filters else None

        # Vector search
        query_embedding = self._embed([query])[0]
        n_candidates = min(k * 3, len(self.documents))
        vector_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_candidates,
            where=self._chroma_where(filters),
            include=["documents", "metadatas", "distances"],
        )
        vector_ids = vector_results["ids"][0]

        # BM25 keyword search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranked = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)

        if candidate_ids:
            bm25_ranked = [i for i in bm25_ranked if self.documents[i]["id"] in candidate_ids]

        bm25_ids = [self.documents[i]["id"] for i in bm25_ranked[:n_candidates]]

        # Reciprocal Rank Fusion
        scores: dict[str, float] = {}
        rrf_k = 60
        for rank, doc_id in enumerate(vector_ids):
            scores[doc_id] = scores.get(doc_id, 0) + alpha / (rrf_k + rank + 1)
        for rank, doc_id in enumerate(bm25_ids):
            scores[doc_id] = scores.get(doc_id, 0) + (1 - alpha) / (rrf_k + rank + 1)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        doc_map = {d["id"]: d for d in self.documents}
        return [
            {**doc_map[doc_id], "score": round(score, 4)}
            for doc_id, score in ranked
            if doc_id in doc_map
        ]

    def _apply_filters(self, filters: dict) -> set[str]:
        """Return document IDs matching metadata filters."""
        matching = set()
        for doc in self.documents:
            if filters.get("category") and doc.get("category") != filters["category"]:
                continue
            if filters.get("source") and doc.get("source") != filters["source"]:
                continue
            matching.add(doc["id"])
        return matching

    def _chroma_where(self, filters: Optional[dict]) -> Optional[dict]:
        if not filters:
            return None
        conditions = []
        if filters.get("category"):
            conditions.append({"category": {"$eq": filters["category"]}})
        if filters.get("source"):
            conditions.append({"source": {"$eq": filters["source"]}})
        if len(conditions) == 1:
            return conditions[0]
        if len(conditions) > 1:
            return {"$and": conditions}
        return None

    def get_facets(self) -> dict:
        """Return available facet values for the filter UI."""
        categories = sorted(set(d.get("category", "General") for d in self.documents))
        sources = sorted(set(d.get("source", "") for d in self.documents))
        return {"categories": categories, "sources": sources, "total_documents": len(self.documents)}
```

---

## Step 3: FastAPI Backend and Web UI

Expose the search engine via API and build a minimal search interface.

```python
# src/api.py
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
from src.search_engine import SemanticSearchEngine
from src.indexer import load_corpus

app = FastAPI(title="Semantic Search Engine")
engine = SemanticSearchEngine()

class SearchRequest(BaseModel):
    query: str
    k: int = 10
    alpha: float = 0.6
    category: Optional[str] = None
    source: Optional[str] = None

@app.on_event("startup")
def startup():
    docs = load_corpus("./data/corpus")
    engine.index_documents([d.__dict__ for d in docs])

@app.get("/health")
def health():
    return {"status": "ok", "documents": len(engine.documents)}

@app.post("/search")
def search(req: SearchRequest):
    filters = {}
    if req.category:
        filters["category"] = req.category
    if req.source:
        filters["source"] = req.source
    results = engine.search(req.query, k=req.k, alpha=req.alpha, filters=filters or None)
    return {"query": req.query, "results": results, "count": len(results)}

@app.get("/facets")
def facets():
    return engine.get_facets()

SEARCH_HTML = """
<!DOCTYPE html>
<html>
<head>
  <title>Semantic Search</title>
  <style>
    body { font-family: system-ui; max-width: 800px; margin: 40px auto; padding: 0 20px; }
    input { width: 100%; padding: 12px; font-size: 16px; border: 1px solid #ccc; border-radius: 8px; }
    .result { border-bottom: 1px solid #eee; padding: 16px 0; }
    .score { color: #666; font-size: 12px; }
    .meta { color: #999; font-size: 13px; }
    select { margin: 8px 8px 8px 0; padding: 8px; }
  </style>
</head>
<body>
  <h1>Semantic Search</h1>
  <input id="q" placeholder="Search by meaning..." onkeydown="if(event.key==='Enter')search()">
  <select id="category"><option value="">All categories</option></select>
  <button onclick="search()">Search</button>
  <div id="results"></div>
  <script>
    fetch('/facets').then(r=>r.json()).then(d=>{
      d.categories.forEach(c=>{
        document.getElementById('category').innerHTML += `<option value="${c}">${c}</option>`;
      });
    });
    async function search() {
      const q = document.getElementById('q').value;
      const category = document.getElementById('category').value;
      const res = await fetch('/search', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({query: q, category: category || null})
      });
      const data = await res.json();
      document.getElementById('results').innerHTML = data.results.map(r =>
        `<div class="result"><strong>${r.title}</strong>
         <div class="meta">${r.category} · ${r.source}</div>
         <p>${r.text.substring(0, 200)}...</p>
         <div class="score">Score: ${r.score}</div></div>`
      ).join('') || '<p>No results found.</p>';
    }
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def search_ui():
    return SEARCH_HTML
```

Run the server:

```bash
uvicorn src.api:app --reload --port 8000
```

---

## Step 4: Evaluation Suite

Measure retrieval quality with a labeled test set.

```python
# evaluation/search_eval.py
import json
import time
from src.search_engine import SemanticSearchEngine

# Format: {"query": "...", "relevant_ids": ["doc1__chunk_0", "doc2__chunk_1"]}
TEST_SET = [
    {"query": "How do vector databases work?", "relevant_ids": ["vector_databases__chunk_0"]},
    {"query": "What is reciprocal rank fusion?", "relevant_ids": ["hybrid_search__chunk_1"]},
    {"query": "embedding models for semantic search", "relevant_ids": ["embeddings_guide__chunk_0"]},
]

def recall_at_k(results: list[dict], relevant_ids: list[str], k: int) -> float:
    retrieved_ids = [r["id"] for r in results[:k]]
    hits = sum(1 for rid in relevant_ids if rid in retrieved_ids)
    return hits / len(relevant_ids) if relevant_ids else 0.0

def mrr(results: list[dict], relevant_ids: list[str]) -> float:
    for rank, r in enumerate(results, start=1):
        if r["id"] in relevant_ids:
            return 1.0 / rank
    return 0.0

def evaluate(engine: SemanticSearchEngine, test_set: list[dict]) -> dict:
    metrics = {"recall_at_5": [], "mrr": [], "latencies": []}
    for case in test_set:
        start = time.time()
        results = engine.search(case["query"], k=10)
        metrics["latencies"].append(time.time() - start)
        metrics["recall_at_5"].append(recall_at_k(results, case["relevant_ids"], 5))
        metrics["mrr"].append(mrr(results, case["relevant_ids"]))
    return {
        "recall_at_5": sum(metrics["recall_at_5"]) / len(test_set),
        "mrr": sum(metrics["mrr"]) / len(test_set),
        "avg_latency_ms": (sum(metrics["latencies"]) / len(test_set)) * 1000,
        "test_count": len(test_set),
    }

if __name__ == "__main__":
    engine = SemanticSearchEngine()
    # Load and index your corpus first
    print(json.dumps(evaluate(engine, TEST_SET), indent=2))
```

---

## Testing Your Build

### Unit Tests

```python
# tests/test_search.py
import pytest
from src.search_engine import SemanticSearchEngine

SAMPLE_DOCS = [
    {"id": "1", "text": "RAG systems combine retrieval with generation for accurate answers.", "title": "RAG Overview", "category": "AI", "source": "rag.txt", "date": "2025-01-01"},
    {"id": "2", "text": "Vector databases store embeddings for similarity search at scale.", "title": "Vector DBs", "category": "Databases", "source": "vectordb.txt", "date": "2025-01-02"},
    {"id": "3", "text": "BM25 is a probabilistic keyword ranking function used in search engines.", "title": "BM25", "category": "Search", "source": "bm25.txt", "date": "2025-01-03"},
]

@pytest.fixture
def engine():
    e = SemanticSearchEngine(collection_name="test-index", persist_dir="./test_db")
    e.index_documents(SAMPLE_DOCS)
    return e

def test_vector_search_finds_semantic_match(engine):
    results = engine.search("How do I search by meaning?", k=3, alpha=1.0)
    assert len(results) > 0
    assert results[0]["id"] in ("1", "2")

def test_keyword_search_finds_exact_match(engine):
    results = engine.search("BM25 probabilistic", k=3, alpha=0.0)
    assert results[0]["id"] == "3"

def test_facet_filter(engine):
    results = engine.search("search", k=10, filters={"category": "AI"})
    assert all(r["category"] == "AI" for r in results)

def test_empty_query_returns_empty(engine):
    results = engine.search("", k=5)
    assert isinstance(results, list)
```

Run tests:

```bash
pytest tests/test_search.py -v
```

### Manual Testing Checklist

- [ ] Search "vector similarity" returns vector database docs (semantic match)
- [ ] Search "BM25" returns BM25 doc (keyword match)
- [ ] Category filter restricts results correctly
- [ ] Empty query handled gracefully
- [ ] UI loads facets and displays results with scores
- [ ] Latency under 500ms for your corpus size

---

## Deployment Notes

### Production Considerations

| Concern | Development | Production |
|---------|-------------|------------|
| Vector DB | ChromaDB local | Pinecone, Weaviate, or Qdrant Cloud |
| Embeddings | OpenAI API | Cache embeddings; batch on index |
| Search latency | Single process | Add Redis cache for popular queries |
| Index updates | Restart server | Background reindexing job |

### Docker Deployment

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

```bash
OPENAI_API_KEY=sk-...
CHROMA_PERSIST_DIR=/data/search_db
SEARCH_DEFAULT_ALPHA=0.6
```

---

## Extensions and Challenges

- **Cross-encoder re-ranking**: Add `sentence-transformers` CrossEncoder to re-rank top 20 results for better precision
- **Autocomplete**: Build a trie or embedding-based suggestion index from document titles
- **Analytics dashboard**: Track popular queries, zero-result rate, and click-through position
- **Multi-language search**: Use multilingual embedding models (`text-embedding-3-small` handles many languages)
- **Incremental indexing**: Add documents without full reindex using ChromaDB upsert
- **Query expansion**: Use an LLM to expand short queries before search

## Resources

- [YouTube: Building Search Engines](https://www.youtube.com/watch?v=lYxGYXjfrNI) — Hybrid search implementation
- [Sentence Transformers Cross-Encoders](https://www.sbert.net/docs/cross_encoder/usage/usage.html) — Re-ranking for better precision
- [BM25 in Python](https://github.com/dorianbrown/rank_bm25) — Keyword search implementation

---

## Next Lesson

**Project 7: LLM-Powered Data Extraction Pipeline** — Build a pipeline that extracts structured data from unstructured documents like invoices, resumes, and contracts.
