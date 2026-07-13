---
title: Chunking Strategies
description: Learn how to split documents effectively for optimal RAG performance
duration: 50 min
difficulty: intermediate
has_code: true
module: module-09
youtube: 'https://www.youtube.com/watch?v=8OJC21T2SL4'
---

# Chunking Strategies

## Prerequisites

- **Lesson 01 — Introduction to RAG** — understand the retrieval pipeline
- **Lesson 02 — Vector Databases & Embeddings** — know that each chunk becomes one vector
- **Python strings and lists** — comfortable with slicing and basic text processing

---

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Explain why chunking is necessary and how chunk size affects retrieval | 10 min | Intermediate |
| Implement fixed-size, sentence-based, and recursive chunking | 15 min | Intermediate |
| Understand semantic chunking and when it's worth the overhead | 10 min | Intermediate |
| Add metadata to chunks and design a chunking pipeline | 15 min | Intermediate |

---

## Intuition First: Why Chunks Must Be "Just Right"

Imagine you're filing a 200-page employee handbook into folders so a colleague can find specific policies later. You have two bad options:

1. **One giant folder** — "Here's the whole handbook, good luck finding the vacation policy." The folder is useless because you can't point to the relevant section.
2. **One sentence per folder** — "Here's a folder for every sentence." Now you have 3,000 folders and no sentence has enough context to answer "What happens if I take more than my annual leave allocation?"

The right answer is **one focused topic per folder** — a few paragraphs that are coherent on their own. That's a chunk.

In RAG terms: each chunk becomes one vector. At query time you retrieve the top-K vectors. If chunks are too large, every chunk is equally (un)related to the query; if they're too small, each chunk lacks context to be useful. The goal is chunks that are **semantically coherent** and **sized for your embedding model's sweet spot**.

---

## Why Chunking Is Necessary

Embedding models accept up to 512–8,192 tokens depending on the model. Even with 8K-token windows, collapsing an entire document into one vector produces an average representation — the signal for any specific fact is drowned out by the rest of the document.

```
10,000-token document → single 1,536-dim vector
                       → loses specificity

Same document → 20 chunks of ~500 tokens → 20 vectors
              → each vector represents one focused topic
              → semantic search can pinpoint the relevant chunk
```

The secondary reason is **cost and context window management**: you pay for every token sent to the LLM. Retrieving three 500-token chunks (1,500 tokens) is far cheaper than pasting an entire 50,000-token document.

---

## Strategy 1: Fixed-Size Chunking

The simplest approach: split every N characters or tokens with a fixed overlap.

```python
def chunk_fixed_size(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """
    Split text into fixed-character chunks with overlap.
    Overlap prevents losing context at chunk boundaries.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap          # step back by overlap amount
    return chunks


text = "The quick brown fox jumps over the lazy dog. " * 30
chunks = chunk_fixed_size(text, chunk_size=100, overlap=20)
print(f"Number of chunks: {len(chunks)}")
print(f"First chunk:\n{chunks[0]}")
print(f"Second chunk starts with overlap:\n{chunks[1][:30]}")
```

**Pros**: Simple, predictable sizes, easy to implement.  
**Cons**: Cuts mid-sentence, breaking semantic coherence. A policy that starts at character 498 may be split in half.

**When to use**: Quick prototypes, or when documents are already structured into short paragraphs (e.g., FAQ entries, database records).

---

## Strategy 2: Sentence-Based Chunking

Respect sentence boundaries by grouping N sentences per chunk. Avoids the mid-sentence split problem.

```python
import re

def chunk_by_sentences(
    text: str,
    sentences_per_chunk: int = 5,
    overlap_sentences: int = 1,
) -> list[str]:
    """
    Split text at sentence boundaries and group into chunks.
    overlap_sentences sentences of context carry over to the next chunk.
    """
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    i = 0
    while i < len(sentences):
        chunk_sentences = sentences[i : i + sentences_per_chunk]
        chunks.append(" ".join(chunk_sentences))
        i += sentences_per_chunk - overlap_sentences  # slide with overlap

    return chunks


text = """
Employees are entitled to 20 days of annual leave per calendar year.
Leave accrues at 1.67 days per month. Unused leave may be carried over
into the following year, up to a maximum of 5 days. Any carry-over
above 5 days is forfeited on January 1st. To request leave, employees
must submit a request via the HR portal at least 2 weeks in advance.
Emergency leave requests require manager approval and HR notification
within 24 hours.
""".strip()

for i, chunk in enumerate(chunk_by_sentences(text, sentences_per_chunk=3)):
    print(f"── Chunk {i} ──\n{chunk}\n")
```

**Pros**: Coherent sentences; much better for narrative text.  
**Cons**: Variable chunk sizes; very long sentences still create large chunks.

---

## Strategy 3: Recursive Character Chunking (LangChain)

The most practical general-purpose approach. It tries progressively smaller separators until chunks fit within the target size.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,          # target character count
    chunk_overlap=120,       # ~20% overlap
    separators=[
        "\n\n",     # try paragraph breaks first
        "\n",       # then line breaks
        ". ",       # then sentence ends
        " ",        # then word boundaries
        "",         # finally, hard split character-by-character
    ],
    length_function=len,     # measure in characters; swap for token counter
)

with open("employee_handbook.txt") as f:
    document = f.read()

chunks = splitter.split_text(document)
print(f"Total chunks: {len(chunks)}")
print(f"Average chunk size: {sum(len(c) for c in chunks) / len(chunks):.0f} chars")
```

The separator list is tried in order. If the document splits naturally at paragraph breaks within the size limit, great. If not, it falls back to line breaks, then sentences, and so on.

!!! note "Use a token-aware splitter in production"
    Embedding model limits are in *tokens*, not characters. A rough approximation is 4 characters per token, but code, languages with long words, and Unicode text can skew this significantly. Use `tiktoken` to count tokens accurately:

    ```python
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,           # tokens
        chunk_overlap=80,
        length_function=lambda t: len(enc.encode(t)),
    )
    ```

---

## Strategy 4: Semantic Chunking

Instead of splitting at fixed intervals or boundaries, detect where the *topic* changes and split there. Compute embeddings for each sentence, then split wherever adjacent sentences are dissimilar.

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

chunker = SemanticChunker(
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
    breakpoint_threshold_type="percentile",   # split at the 95th-percentile dissimilarity
    breakpoint_threshold_amount=95,
)

chunks = chunker.split_text(long_document)
```

Under the hood:
1. Split text into sentences.
2. Embed each sentence.
3. Compute cosine similarity between consecutive sentences.
4. Split at points where similarity drops below the threshold (i.e., topic change).

**Pros**: Chunks are topically coherent by construction; naturally handles variable-length sections.  
**Cons**: Expensive — embeds every sentence during ingestion; slower and more complex.

**When to use**: Long documents with clearly distinct topics (e.g., annual reports, textbooks, legal contracts). Worth the cost when retrieval precision is critical.

---

## Worked Example: Comparing Strategies on the Same Text

```python
document = """
Section 1: Refund Policy

Our refund policy allows customers to return items within 30 days of
purchase. Items must be in their original, unused condition with all
original packaging and tags attached. Refunds are processed within 5-7
business days after we receive the returned item.

Section 2: Shipping Policy

Standard shipping takes 5-7 business days and is free for orders over
USD 50. Expedited shipping (2-3 business days) costs USD 15. International
shipping is available to select countries and takes 10-21 business days.
Tracking information is emailed once the order ships.

Section 3: Contact Information

Our support team is available Monday through Friday, 9am to 6pm EST.
Email us at support@example.com or call 1-800-555-0199. For urgent
matters outside business hours, use the emergency contact form on our
website.
"""

# Fixed-size (may split within sections)
fixed = chunk_fixed_size(document, chunk_size=200, overlap=40)
print(f"Fixed-size: {len(fixed)} chunks")

# Recursive (respects paragraphs)
recursive = splitter.split_text(document)
print(f"Recursive: {len(recursive)} chunks")

# Sentence-based (respects sentences)
sentence = chunk_by_sentences(document, sentences_per_chunk=4)
print(f"Sentence-based: {len(sentence)} chunks")
```

On this document, recursive chunking will cleanly produce three chunks (one per section) because the paragraph boundaries fall within the size limit. Fixed-size chunking may awkwardly split the refund policy across two chunks.

---

## Metadata Enrichment

Every chunk should carry metadata that allows the system to cite sources, filter by category, and handle staleness.

```python
from datetime import datetime

def chunk_document_with_metadata(
    text: str,
    doc_id: str,
    title: str,
    source_url: str,
    category: str,
    splitter,
) -> list[dict]:
    """
    Chunk a document and attach rich metadata to each chunk.
    Returns a list of dicts ready for vector database ingestion.
    """
    raw_chunks = splitter.split_text(text)

    return [
        {
            "id": f"{doc_id}_chunk_{i}",
            "text": chunk,
            "metadata": {
                "doc_id":       doc_id,
                "title":        title,
                "source_url":   source_url,
                "category":     category,
                "chunk_index":  i,
                "total_chunks": len(raw_chunks),
                "char_count":   len(chunk),
                "indexed_at":   datetime.utcnow().isoformat(),
            },
        }
        for i, chunk in enumerate(raw_chunks)
    ]


chunks = chunk_document_with_metadata(
    text=document,
    doc_id="policy-v2",
    title="Customer Policies Q3 2024",
    source_url="https://internal.example.com/policies",
    category="customer-support",
    splitter=splitter,
)

for chunk in chunks:
    print(f"[{chunk['id']}] {chunk['text'][:80]}...")
    print(f"  metadata: {chunk['metadata']}\n")
```

Metadata enables:
- **Citation** — "Source: Customer Policies Q3 2024, chunk 2"
- **Filtering** — "Only search chunks with `category=customer-support`"
- **Freshness** — "Exclude chunks indexed before 2024-01-01"

---

## Chunk Size Guidelines

| Document type | Recommended chunk size | Overlap | Rationale |
|---------------|----------------------|---------|-----------|
| Short Q&A entries | 150–300 tokens | 10% | Each entry is self-contained |
| Policy / procedure docs | 300–500 tokens | 15–20% | Coherent policy = one chunk |
| Technical documentation | 400–600 tokens | 15% | One concept per chunk |
| Legal contracts | 500–800 tokens | 20% | Context across clauses |
| Narrative / book text | 600–1,000 tokens | 20% | Longer for narrative flow |

The overlap percentage (10–20%) prevents information from being stranded at a chunk boundary. A sentence that straddles the boundary appears in both the preceding and following chunk, ensuring it's always reachable.

!!! warning "Chunk size is a retrieval hyperparameter — always measure"
    A chunk size that works well for policy documents may fail for code documentation. Measure **Recall@5** (what percentage of ground-truth answer passages appear in the top-5 retrieved chunks) on a small eval set before deploying. Even a 5-question sample can catch gross misconfigurations.

---

## Edge Cases & Misconceptions

**Misconception: Larger chunks are better because they give the LLM more context.**
More context per chunk means the retriever must be more precise (retrieving few, large chunks). If the chunk is too large, the embedding vector averages over too many topics and the semantic signal for any specific fact is diluted. Precision vs. recall is always a trade-off.

**Misconception: Overlap is just wasted tokens.**
Without overlap, a sentence at the boundary of two chunks belongs to neither chunk's embedding — it's effectively invisible to retrieval. Overlap ensures every sentence appears in at least one chunk with enough surrounding context for the embedding to capture its meaning.

**Edge case: Tables and structured data.**
Plain text splitters destroy tables by splitting mid-row. Use document-aware parsers (Unstructured.io, `pdfplumber`, `docling`) that extract tables as structured data before chunking, then store them separately or serialize them to Markdown before embedding.

**Edge case: Code files.**
Splitting Python at character boundaries produces syntactically invalid fragments. Use tree-sitter or language-aware splitters that chunk at function or class boundaries.

```python
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

code_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=1000,
    chunk_overlap=100,
)
```

---

## Production Connection

In production systems the chunking pipeline is one of the highest-leverage places to iterate. Common patterns:

- **Two-pass chunking** — large chunks (800 tokens) for embedding, smaller sub-chunks (200 tokens) stored as searchable snippets. Retrieve the large chunk, display the sub-chunk. Sometimes called "parent-child" chunking.
- **Chunk-then-summarize** — for each chunk, generate a short LLM summary and embed *the summary* rather than the raw text. Improves retrieval when documents are verbose or noisy.
- **Ingestion observability** — log chunk count, average size, and empty-chunk rate. An empty-chunk rate above 5% signals a parser issue (e.g., blank pages in a PDF).
- **Re-chunking on model upgrade** — if you switch embedding models, re-index all chunks. Vectors from different models are not comparable; mixing them silently degrades retrieval.

---

## Key Takeaways

- Chunking converts long documents into retrieval-sized pieces; each chunk becomes one vector in the database.
- Fixed-size chunking is simple but cuts mid-sentence; recursive chunking respects natural boundaries and is the best general default.
- Semantic chunking splits at topic boundaries and produces the most coherent chunks, at the cost of embedding every sentence during ingestion.
- Always add metadata (doc ID, title, source, timestamp) to every chunk for citation and filtering.
- Overlap (10–20%) prevents information from being lost at chunk boundaries.
- Measure Recall@K on a small eval set before deploying; chunk size is a tunable hyperparameter.
- Handle tables, code, and structured data with format-aware parsers, not naive text splitters.

---

## Chunking Strategy Decision Guide

Use this guide to pick a starting strategy before benchmarking:

```
What type of documents are you chunking?
    │
    ├─ Short, self-contained entries (FAQ, product descriptions, records)
    │       → Fixed-size at 150–300 tokens, minimal overlap
    │
    ├─ Narrative prose (articles, book chapters, reports)
    │       → Recursive CharacterTextSplitter, 400–600 tokens, 15% overlap
    │
    ├─ Technical docs with clear section headers
    │       → Recursive splitting with double-newline as primary separator
    │
    ├─ Legal contracts or compliance documents
    │       → Semantic chunking or manual header-based splitting
    │
    ├─ Source code
    │       → Language-aware splitter (function/class boundaries)
    │
    └─ PDFs with tables, figures, and mixed content
            → Document parser (Unstructured.io, pdfplumber) first,
              then apply text splitting to prose sections only
```

Regardless of initial choice, always run a retrieval evaluation after indexing:
1. Create 10–20 test questions whose answers are in your corpus.
2. Measure what percentage of correct answer chunks appear in top-5 retrieved results (Recall@5).
3. If Recall@5 < 70%, adjust chunk size and re-evaluate.

A 10-question eval set built in an afternoon will save you days of debugging why the live system gives wrong answers. Track Recall@5 over time in a spreadsheet — it is the single most diagnostic number for chunking quality. Improvements in chunk strategy rarely hurt; they almost always help.

---

## Related Papers

| Paper | What it contributes | Link |
|-------|-------------------|------|
| Sarthi et al. (2024) — *RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval* | Hierarchical chunking via LLM summaries at multiple granularities | [arxiv.org/abs/2401.18059](https://arxiv.org/abs/2401.18059) |
| Edge et al. (2024) — *From Local to Global: A Graph RAG Approach* | Graph-based chunking that captures cross-document relationships | [arxiv.org/abs/2404.16130](https://arxiv.org/abs/2404.16130) |
| Chen et al. (2023) — *Dense X Retrieval: What Retrieval Granularity Should We Use?* | Empirical study of chunk granularity effects on open-domain QA | [arxiv.org/abs/2312.06648](https://arxiv.org/abs/2312.06648) |

---

## Further Reading

- [Pinecone: Chunking Strategies for LLM Applications](https://www.pinecone.io/learn/chunking-strategies/) — comprehensive illustrated guide
- [LangChain Text Splitters Reference](https://python.langchain.com/docs/modules/data_connection/document_transformers/) — all available splitter types with parameters
- [Unstructured.io](https://unstructured.io/) — document parsing library that handles PDFs, HTML, DOCX, tables, and more

---

## Next Lesson

**[Lesson 4: Retrieval Methods](04-Retrieval-Methods.md)** — Learn how dense, sparse, and hybrid retrieval differ — and when to reach for each method in production RAG systems.
