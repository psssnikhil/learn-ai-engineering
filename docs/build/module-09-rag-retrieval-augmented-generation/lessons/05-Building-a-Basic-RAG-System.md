---
title: Building a Basic RAG System
description: >-
  Build an end-to-end RAG pipeline — ingest documents, embed chunks, retrieve
  context, and generate grounded answers
duration: 60 min
difficulty: intermediate
has_code: true
module: module-09
youtube: 'https://www.youtube.com/watch?v=sVcwVQRHIc8'
---

# Building a Basic RAG System

## Prerequisites

- **Lessons 01–04** — RAG concepts, embeddings, chunking, retrieval methods
- **Python intermediate** — classes, type hints, list comprehensions
- **OpenAI API key** set in environment: `export OPENAI_API_KEY=sk-...`
- Install: `pip install openai chromadb`

---

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Design a clean, testable RAG pipeline class | 15 min | Intermediate |
| Implement each stage: ingest, chunk, embed, store, retrieve, generate | 20 min | Intermediate |
| Add observability: logging, latency tracking, source citation | 10 min | Intermediate |
| Diagnose and fix the five most common RAG failures | 15 min | Intermediate |

---

## Intuition First: Assembling the Pipeline

All previous lessons covered individual components. Now you assemble them:

```
OFFLINE PIPELINE (runs once or on document updates)
────────────────────────────────────────────────────
Raw documents (PDFs, text, HTML)
        │
        ▼ ingest(): parse and normalize
Clean text with metadata
        │
        ▼ chunk(): split into retrieval-sized pieces
List of chunks (dict with text + metadata)
        │
        ▼ embed_and_store(): vectorize and persist
Vector store (ChromaDB on disk)


ONLINE PIPELINE (runs on every user query, ~300–600 ms)
────────────────────────────────────────────────────────
User query (string)
        │
        ▼ retrieve(): embed query, ANN search
Top-K chunks (with similarity scores + metadata)
        │
        ▼ generate(): build augmented prompt, call LLM
Grounded answer (string, with optional citations)
```

Each step is independently testable. You can verify retrieval quality before the LLM is ever called.

---

## Step 1: Document Ingestion

Good ingestion normalizes whitespace, preserves structure, and attaches metadata. Bad ingestion propagates noise into every downstream step.

```python
import re
from pathlib import Path
from datetime import datetime

def ingest_text_file(path: str, category: str = "general") -> dict:
    """
    Read a plain-text file and return a normalized document dict.
    """
    raw = Path(path).read_text(encoding="utf-8")

    # Normalize: collapse multiple blank lines, strip leading/trailing whitespace
    text = re.sub(r'\n{3,}', '\n\n', raw).strip()

    return {
        "doc_id":    Path(path).stem,
        "title":     Path(path).stem.replace("-", " ").replace("_", " ").title(),
        "text":      text,
        "category":  category,
        "source":    str(path),
        "char_count": len(text),
        "ingested_at": datetime.utcnow().isoformat(),
    }


def ingest_string(text: str, doc_id: str, title: str,
                  category: str = "general") -> dict:
    """
    Ingest an in-memory string — useful for tests and dynamic content.
    """
    normalized = re.sub(r'\n{3,}', '\n\n', text).strip()
    return {
        "doc_id":    doc_id,
        "title":     title,
        "text":      normalized,
        "category":  category,
        "source":    "in-memory",
        "char_count": len(normalized),
        "ingested_at": datetime.utcnow().isoformat(),
    }
```

---

## Step 2: Chunking with Metadata Propagation

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")

def _count_tokens(text: str) -> int:
    return len(_enc.encode(text))

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=_count_tokens,   # token-aware splitting
)

def chunk_document(doc: dict) -> list[dict]:
    """
    Split a document dict into chunk dicts.
    Each chunk inherits the document's metadata.
    """
    raw_chunks = _splitter.split_text(doc["text"])

    chunks = []
    for i, text in enumerate(raw_chunks):
        chunks.append({
            "chunk_id":    f"{doc['doc_id']}_chunk_{i:04d}",
            "doc_id":      doc["doc_id"],
            "title":       doc["title"],
            "category":    doc["category"],
            "source":      doc["source"],
            "text":        text,
            "chunk_index": i,
            "total_chunks": len(raw_chunks),
            "token_count": _count_tokens(text),
            "indexed_at":  datetime.utcnow().isoformat(),
        })

    return chunks
```

!!! note "Token count in metadata"
    Storing `token_count` per chunk lets you monitor chunk size distribution in production dashboards and flag ingestion anomalies (e.g., a PDF page that parsed as a single 8,000-token chunk because the parser failed).

---

## Step 3: Embedding and Storing

```python
import chromadb
from openai import OpenAI
import time

_openai = OpenAI()
_chroma = chromadb.PersistentClient(path="./rag_store")

def get_or_create_collection(name: str):
    return _chroma.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )

def embed_batch(texts: list[str], model: str = "text-embedding-3-small") -> list[list[float]]:
    """
    Embed a list of texts in one API call. Much faster than one-at-a-time.
    """
    response = _openai.embeddings.create(model=model, input=texts)
    return [item.embedding for item in response.data]

def index_chunks(chunks: list[dict], collection_name: str = "docs") -> int:
    """
    Embed and store a list of chunk dicts.
    Returns the number of chunks indexed.
    """
    collection = get_or_create_collection(collection_name)

    # Batch embed (up to 2048 texts per request for text-embedding-3-small)
    batch_size = 100
    indexed = 0

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [c["text"] for c in batch]
        embeddings = embed_batch(texts)

        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {k: v for k, v in c.items() if k not in ("text", "chunk_id")}
                for c in batch
            ],
        )
        indexed += len(batch)
        print(f"  Indexed {indexed}/{len(chunks)} chunks...")

    return indexed
```

---

## Step 4: Retrieval with Observability

```python
from dataclasses import dataclass

@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    score: float            # cosine similarity (higher = more relevant)
    doc_id: str
    title: str
    source: str
    chunk_index: int


def retrieve(
    query: str,
    collection_name: str = "docs",
    n_results: int = 4,
    category: str | None = None,
) -> list[RetrievalResult]:
    """
    Retrieve the most relevant chunks for a query.
    Returns structured RetrievalResult objects with scores and metadata.
    """
    collection = get_or_create_collection(collection_name)

    query_embedding = embed_batch([query])[0]

    where = {"category": category} if category else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "distances", "metadatas"],
    )

    retrieval_results = []
    for i, (doc, dist, meta) in enumerate(zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    )):
        # ChromaDB returns distance; convert to similarity for interpretability
        similarity = 1 - dist

        retrieval_results.append(RetrievalResult(
            chunk_id    = results["ids"][0][i],
            text        = doc,
            score       = similarity,
            doc_id      = meta.get("doc_id", ""),
            title       = meta.get("title", ""),
            source      = meta.get("source", ""),
            chunk_index = meta.get("chunk_index", 0),
        ))

    return retrieval_results
```

!!! note "Log retrieval results alongside every answer"
    In production, store the chunk IDs and scores next to every LLM response in your database. When a user reports a bad answer, you can immediately inspect what context the model saw — without guessing.

---

## Step 5: Generation with Grounding

```python
SYSTEM_PROMPT = """\
You are a helpful assistant. Answer the user's question using ONLY the
provided context passages. If the answer is not contained in the
context, respond with: "I don't have information about this in the
provided documents."

Do not add information from your training data. Cite the source
document name when relevant.
"""

def format_context(results: list[RetrievalResult]) -> str:
    """
    Format retrieved chunks into a labeled context block for the prompt.
    """
    parts = []
    for r in results:
        parts.append(
            f"[Source: {r.title} | chunk {r.chunk_index}]\n{r.text}"
        )
    return "\n\n---\n\n".join(parts)


def generate(
    query: str,
    results: list[RetrievalResult],
    model: str = "gpt-4o-mini",
) -> str:
    """
    Generate a grounded answer from retrieved chunks.
    """
    context = format_context(results)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context passages:\n\n{context}\n\nQuestion: {query}",
        },
    ]

    response = _openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,    # low temperature for factual retrieval tasks
    )

    return response.choices[0].message.content
```

---

## The Complete RAGSystem Class

```python
import time
import logging
from dataclasses import dataclass, field
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    answer: str
    sources: list[str]               # List of "title (chunk N)" citations
    retrieval_scores: list[float]    # Similarity scores for retrieved chunks
    latency_ms: dict[str, float]     # Breakdown: embed, retrieve, generate


class RAGSystem:
    """
    End-to-end RAG system with observability.
    """

    def __init__(
        self,
        collection_name: str = "docs",
        embedding_model: str = "text-embedding-3-small",
        generation_model: str = "gpt-4o-mini",
        n_results: int = 4,
    ):
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.n_results = n_results

    def add_document(self, text: str, doc_id: str, title: str,
                     category: str = "general") -> int:
        """
        Ingest, chunk, embed, and index a document string.
        Returns the number of chunks indexed.
        """
        logger.info(f"Ingesting document: {title!r}")
        doc = ingest_string(text, doc_id, title, category)
        chunks = chunk_document(doc)
        logger.info(f"  Created {len(chunks)} chunks")
        n = index_chunks(chunks, self.collection_name)
        logger.info(f"  Indexed {n} chunks into '{self.collection_name}'")
        return n

    def query(
        self,
        question: str,
        category: str | None = None,
        low_score_threshold: float = 0.30,
    ) -> RAGResponse:
        """
        Answer a question using RAG.
        Returns a structured RAGResponse with answer, citations, and latency.
        """
        latency: dict[str, float] = {}
        t0 = time.perf_counter()

        # Retrieve
        t_retrieve = time.perf_counter()
        results = retrieve(
            query=question,
            collection_name=self.collection_name,
            n_results=self.n_results,
            category=category,
        )
        latency["retrieve_ms"] = (time.perf_counter() - t_retrieve) * 1000

        # Low-confidence guard: if best chunk is below threshold, say so
        if results and results[0].score < low_score_threshold:
            logger.warning(
                f"Best retrieval score {results[0].score:.3f} below threshold "
                f"{low_score_threshold}. Answering from low-confidence context."
            )

        # Generate
        t_generate = time.perf_counter()
        answer = generate(question, results, model=self.generation_model)
        latency["generate_ms"] = (time.perf_counter() - t_generate) * 1000

        latency["total_ms"] = (time.perf_counter() - t0) * 1000

        return RAGResponse(
            answer=answer,
            sources=[f"{r.title} (chunk {r.chunk_index})" for r in results],
            retrieval_scores=[r.score for r in results],
            latency_ms=latency,
        )
```

---

## Running It End to End

```python
# ── Build the knowledge base ───────────────────────────
rag = RAGSystem(collection_name="company-kb")

documents = [
    (
        """
        Refund Policy
        Customers may return items within 30 days of purchase.
        Items must be unused, in original packaging, and accompanied
        by a receipt. Refunds are processed within 5–7 business days.
        Digital products are non-refundable once downloaded.
        """,
        "policy-refund",
        "Refund Policy",
        "customer-support",
    ),
    (
        """
        Shipping Policy
        Standard shipping takes 5–7 business days and is free for
        orders over USD 50. Expedited shipping (2–3 days) costs USD 15.
        International orders may face additional customs delays.
        """,
        "policy-shipping",
        "Shipping Policy",
        "customer-support",
    ),
    (
        """
        Annual Leave Policy
        Full-time employees receive 20 days of paid leave per year.
        Leave accrues at 1.67 days per month. Up to 5 unused days
        may be carried into the next calendar year.
        """,
        "hr-leave",
        "Annual Leave Policy",
        "hr",
    ),
]

for text, doc_id, title, category in documents:
    rag.add_document(text, doc_id, title, category)

# ── Query ──────────────────────────────────────────────
response = rag.query("Can I return a digital product I downloaded yesterday?")

print(f"Answer: {response.answer}\n")
print(f"Sources: {response.sources}")
print(f"Scores:  {[f'{s:.3f}' for s in response.retrieval_scores]}")
print(f"Latency: {response.latency_ms}")
```

Expected output:
```
Answer: No — digital products are non-refundable once downloaded,
        according to the Refund Policy.

Sources: ['Refund Policy (chunk 0)', 'Shipping Policy (chunk 0)', ...]
Scores:  ['0.843', '0.312', ...]
Latency: {'retrieve_ms': 45.2, 'generate_ms': 721.4, 'total_ms': 766.6}
```

---

## Diagnosing the Five Most Common RAG Failures

| Symptom | Root cause | Fix |
|---------|------------|-----|
| Hallucinated answer despite good system prompt | Grounding instruction too weak; model ignored context | Add "Do NOT use knowledge outside the provided context." Reduce temperature to 0.0–0.1. |
| Correct document indexed but not retrieved | Chunk too large — embedding averages over too many topics | Reduce chunk size to 300–400 tokens; re-index |
| Retrieved chunks don't contain the answer | Answer not in any indexed document | Audit corpus for coverage gaps; add missing documents |
| Accurate but vague answer ("it depends...") | Retrieved K=1 or 2 — not enough context passed | Increase `n_results` to 4–6; adjust chunk overlap |
| Stale answers after document update | Old chunks still in index | Re-index documents on change; use deterministic chunk IDs for upsert idempotency |

!!! warning "The retrieval ceiling is hard"
    If the answer doesn't exist in indexed documents, no generation quality improvement will produce it. A common mistake is tuning the prompt when the problem is actually corpus coverage. Always check retrieval quality first.

---

## Production Connection

Moving from this basic system to production requires several additions:

**Async ingestion pipeline**: Use a queue (Celery, Cloud Tasks) to process large document batches asynchronously. Never block the API response path on embedding computation.

**Incremental updates**: Compute a content hash for each document on ingestion. On subsequent runs, skip documents whose hash hasn't changed. Re-index only changed or new documents.

**Evaluation harness**: Maintain a 50–100 question eval set with ground-truth answers. Run retrieval and generation evaluation on every deployment. Alert if Recall@5 drops more than 3 percentage points.

**Latency monitoring**: Instrument each pipeline stage with your APM (Datadog, Honeycomb). Embedding latency spikes indicate API throttling; retrieval latency spikes indicate index rebuild or network issues.

**Cost tracking**: Log `input_tokens` and `output_tokens` from every LLM response. For high-volume systems, these numbers drive your unit economics and chunk-size decisions.

---

## Edge Cases & Misconceptions

**Misconception: Temperature 0 means the LLM always gives the same answer.**
Temperature 0 minimizes randomness but LLMs are not deterministic at zero temperature due to floating-point non-determinism and top-p sampling thresholds. Use `seed` parameter alongside `temperature=0` when reproducibility is critical.

**Misconception: Longer system prompts are more reliable.**
Past ~500 tokens, the system prompt competes with retrieved context for the model's attention. Keep the grounding instruction concise and clear; the context passages carry the facts.

**Edge case: Multi-document questions.**
"Compare our refund policy and shipping policy" retrieves chunks from two documents. Format the context with clear source labels (already done above) so the model can synthesize across sources without conflating them.

**Edge case: Re-indexing at scale.**
For 10 million chunks, re-indexing from scratch takes hours. Use an append-and-delete pattern: add new chunks with new IDs, then delete old chunk IDs in a separate pass. This keeps the index live during updates.

---

## Key Takeaways

- Structure your RAG pipeline as discrete, testable stages: ingest → chunk → embed → store → retrieve → generate.
- Add rich metadata to every chunk — source title, category, timestamp — to support citation, filtering, and freshness control.
- Batch embedding calls — always send arrays of texts, never one at a time.
- Log retrieval results (chunk IDs + scores) alongside every answer; debugging bad answers requires knowing what context the model saw.
- Add a low-score guard: if the best retrieved chunk is below a similarity threshold, the answer is likely unreliable.
- Low temperature (0.0–0.1) is appropriate for factual retrieval tasks; higher temperature for creative tasks.
- Before tuning the prompt, verify retrieval quality. Retrieval failure and generation failure have different symptoms and different fixes.

---

## Eval-Driven Development for RAG

The most common mistake when building a RAG system is developing by vibes — asking a few questions, reading the answers, and declaring it "working." This leads to fragile systems that fail unpredictably on real user queries.

A minimal eval harness that prevents this:

```python
from dataclasses import dataclass

@dataclass
class EvalCase:
    question: str
    expected_answer_keywords: list[str]  # Keywords that must appear in a correct answer
    expected_source_doc: str             # Which document should be retrieved

eval_set = [
    EvalCase(
        question="Can I return a digital product?",
        expected_answer_keywords=["non-refundable", "digital", "downloaded"],
        expected_source_doc="policy-refund",
    ),
    EvalCase(
        question="How long does shipping take?",
        expected_answer_keywords=["5", "7", "business days"],
        expected_source_doc="policy-shipping",
    ),
    EvalCase(
        question="How many vacation days do I get?",
        expected_answer_keywords=["20", "annual", "leave"],
        expected_source_doc="hr-leave",
    ),
]

def evaluate_rag(rag: RAGSystem, cases: list[EvalCase]) -> dict:
    results = {"retrieval_hits": 0, "answer_hits": 0, "total": len(cases)}

    for case in cases:
        response = rag.query(case.question)

        # Check retrieval: was the right document in the sources?
        retrieval_hit = any(
            case.expected_source_doc in src for src in response.sources
        )

        # Check answer: do expected keywords appear?
        answer_hit = all(
            kw.lower() in response.answer.lower()
            for kw in case.expected_answer_keywords
        )

        results["retrieval_hits"] += retrieval_hit
        results["answer_hits"]    += answer_hit

    results["recall_at_k"] = results["retrieval_hits"] / results["total"]
    results["answer_accuracy"] = results["answer_hits"] / results["total"]
    return results

# Run before every significant change
scores = evaluate_rag(rag, eval_set)
print(f"Recall@K: {scores['recall_at_k']:.0%}")
print(f"Answer accuracy: {scores['answer_accuracy']:.0%}")
```

Run this eval after every change to chunking strategy, embedding model, or retrieval parameters. A Recall@K drop of > 5% is a regression — revert and investigate before deploying.

---

## Related Papers

| Paper | What it contributes | Link |
|-------|-------------------|------|
| Lewis et al. (2020) — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* | Original RAG architecture with seq2seq generator + DPR retriever | [arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401) |
| Shi et al. (2023) — *REPLUG: Retrieval-Augmented Language Model Pre-Training* | Treating retrieval as an ensemble; updating retriever via LM feedback | [arxiv.org/abs/2301.12652](https://arxiv.org/abs/2301.12652) |
| Es et al. (2023) — *RAGAS: Automated Evaluation of Retrieval Augmented Generation* | Framework for evaluating RAG with faithfulness, answer relevancy, and context recall metrics | [arxiv.org/abs/2309.15217](https://arxiv.org/abs/2309.15217) |
| Asai et al. (2023) — *Self-RAG: Learning to Retrieve, Generate, and Critique Through Self-Reflection* | LLM that decides when to retrieve and critiques its own citations | [arxiv.org/abs/2310.11511](https://arxiv.org/abs/2310.11511) |

---

## Further Reading

- [Build a RAG System from Scratch](https://www.youtube.com/watch?v=sVcwVQRHIc8) — end-to-end walkthrough (30 min)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/) — official getting-started guide
- [RAGAS Documentation](https://docs.ragas.io/) — automated RAG evaluation framework
- [OpenAI Cookbook: Question Answering Using Embeddings](https://cookbook.openai.com/examples/question_answering_using_embeddings) — practical patterns

---

## Next Lesson

**[Lesson 6: Advanced RAG Techniques](06-Advanced-RAG-Techniques.md)** — Query rewriting, HyDE, parent-document retrieval, and other methods that push RAG quality beyond the basics.
