---
title: Advanced RAG Techniques
description: >-
  Level up RAG with query rewriting, re-ranking, contextual compression, and
  multi-step retrieval
duration: 55 min
difficulty: advanced
has_code: true
module: module-09
youtube: 'https://www.youtube.com/watch?v=TRjq7t2Ms5I'
---

## Prerequisites

- [Lesson 05 — Building a Basic RAG System](05-Building-a-Basic-RAG-System.md): end-to-end naive RAG pipeline
- [Lesson 04 — Retrieval Methods](04-Retrieval-Methods.md): dense vs sparse retrieval, cosine similarity
- [Lesson 03 — Chunking Strategies](03-chunking-strategies.md): why chunk size affects retrieval quality
- Comfortable reading Python; basic familiarity with the OpenAI SDK

---

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Diagnose when naive RAG fails and choose the right fix | 55 min | Advanced |
| Apply query transformation (HyDE, multi-query, step-back) | | |
| Use cross-encoders for precision re-ranking after vector search | | |
| Compress context to fit token budgets without losing signal | | |
| Chain retrieval steps for multi-hop and comparative questions | | |

---

## Intuition First

Think of naive RAG as a library patron who walks to the card catalog, finds three books by title, and reads the first page of each before answering your question. That works for simple lookups. It fails when:

- Your question uses shorthand the catalog doesn't index ("that new policy")
- The answer requires comparing chapters from two different books
- The relevant sentence is buried on page 200 of a 300-page book

Advanced RAG adds *pre-search* and *post-search* stages — a research librarian who rewrites your question, checks multiple indexes, skims full chapters, and discards noise before handing you a summary.

The stages map to four techniques:

```
User query
    │
    ▼
[Query Transformation]  — fix the question before embedding
    │
    ▼
[Vector Search]         — fast, high-recall candidates
    │
    ▼
[Re-ranking]            — slow, high-precision scoring
    │
    ▼
[Contextual Compression] — strip noise, fit token budget
    │
    ▼
LLM generation
```

Each stage costs latency. Add them surgically when eval scores prove you need them — not preemptively.

---

## When Basic RAG Falls Short

Naive RAG — embed query, fetch top-K, generate — breaks down in four patterns:

| Failure pattern | Example | Root cause |
|----------------|---------|------------|
| **Vocabulary mismatch** | User says "reset credentials"; doc says "change password" | Embedding distance penalizes different surface forms |
| **Multi-part questions** | "Compare plan A vs plan B pricing" | Single query vector can't represent two intents |
| **Ambiguous shorthand** | "the new policy" | No context to disambiguate; wrong chunks retrieved |
| **Long-document precision** | Legal clause buried in 40-page contract | Top-K chunks are noisy; LLM hallucinates in the noise |

!!! tip "Diagnose before adding complexity"
    Always evaluate retrieval and generation separately first. If Recall@5 is above 0.80, the problem is likely generation — adding re-ranking won't help. Run [Lesson 08 — RAG Evaluation Metrics](08-RAG-Evaluation-Metrics.md) before choosing a fix.

---

## Query Transformation

### Why Queries Are Often the Problem

Embeddings capture semantic similarity, but user queries are short, ambiguous, and written for humans — not for vector spaces. A 4-word query like "update user email" has a very different embedding distribution from the documentation paragraph that says "To modify the email address associated with your account, navigate to…"

Query transformation bridges that gap by rewriting the query into a form whose embedding will retrieve better candidates.

### HyDE — Hypothetical Document Embeddings

**Idea:** Ask the LLM to write a plausible answer document, then embed *that* for search instead of the original query.

**Why it works:** A hypothetical answer is longer, richer in vocabulary, and structurally similar to real documentation — so it lands in a closer neighborhood of the embedding space.

**Worked example — step by step:**

Query: `"What is the API rate limit?"`

HyDE hypothetical:
> *"The API enforces a rate limit of 100 requests per minute for free-tier accounts and 1,000 requests per minute for Pro accounts. Exceeding this limit returns HTTP 429 with a Retry-After header indicating how many seconds to wait before retrying."*

The hypothetical embeds as a dense vector `h`. We then search for chunks with high cosine similarity to `h`. The chunk `"Free tier: 100 req/min; Pro: 1,000 req/min. HTTP 429 on excess."` now scores highly because it shares specific vocabulary (`429`, `req/min`, `Retry-After`) with `h`, even though the original query `"API rate limit"` might have matched only moderately.

```python
def hyde_query(user_query: str, client) -> str:
    """Generate a hypothetical answer document for embedding."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Write a short, specific documentation paragraph that "
                    "would directly answer the following question. "
                    "Be concrete — include numbers, API names, or steps if applicable. "
                    "Do not say you don't know."
                ),
            },
            {"role": "user", "content": user_query},
        ],
        max_tokens=200,
    )
    return response.choices[0].message.content
```

!!! note "HyDE limitations"
    HyDE can introduce hallucinated specifics into the hypothetical (wrong version numbers, nonexistent endpoints). These hallucinations can *improve* retrieval by adding vocabulary, but if they are so wrong that no real chunk matches them, retrieval quality drops. Monitor Recall@5 with and without HyDE on your eval set.

### Multi-Query Expansion

Generate several distinct search queries from one user question, run each, merge and deduplicate results before ranking.

```python
def multi_query(user_query: str, client, n: int = 3) -> list[str]:
    """Expand one question into N search queries."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    f"Generate {n} different search queries that together cover "
                    "all aspects of the user's question. "
                    "Write one query per line, no bullets or numbering."
                ),
            },
            {"role": "user", "content": user_query},
        ],
    )
    return [
        q.strip()
        for q in response.choices[0].message.content.split("\n")
        if q.strip()
    ]

def retrieve_multi_query(
    user_query: str,
    vector_store,
    client,
    k: int = 5,
) -> list[str]:
    queries = multi_query(user_query, client)
    seen_ids: set[str] = set()
    chunks: list[str] = []

    for q in queries:
        results = vector_store.similarity_search(q, k=k)
        for chunk_id, text in results:
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                chunks.append(text)

    return chunks  # feed to re-ranker next
```

### Step-Back Prompting

Ask the LLM to first identify the *general principle* behind a specific question, then retrieve documents about that principle. Useful for physics, law, medicine where specific cases are explained by general rules.

```
Specific: "Why does my JWT expire after 15 minutes?"
Step-back: "What are the security trade-offs of JWT expiration policies?"
```

Retrieve on the step-back query, then answer the specific question with that context.

---

## Re-Ranking with Cross-Encoders

### Bi-Encoder vs Cross-Encoder

Vector search uses **bi-encoders**: query and document are embedded independently, then compared with cosine similarity. This is fast (O(1) lookup after indexing) but approximate — the model never sees query and document together.

**Cross-encoders** take `(query, document)` as a joint input and output a single relevance score. Because they attend over both simultaneously, they are far more accurate — but cost O(N) full forward passes for N candidates.

The standard two-stage architecture:

```
Stage 1: Vector search → top 50 candidates   (milliseconds, bi-encoder)
Stage 2: Cross-encoder re-rank → top 5       (100–500 ms, cross-encoder)
Stage 3: LLM generation with top-5 context
```

This gives you the *recall* of vector search and the *precision* of a cross-encoder, at a cost that scales with your candidate pool size, not your full corpus.

```python
from sentence_transformers import CrossEncoder

# ms-marco models are trained on passage retrieval — good general-purpose choice
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, documents: list[str], top_k: int = 5) -> list[str]:
    """Return top_k documents sorted by cross-encoder relevance score."""
    pairs = [[query, doc] for doc in documents]
    scores = reranker.predict(pairs)  # shape: (N,)
    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]

def full_retrieve_and_rerank(
    query: str,
    vector_store,
    top_candidates: int = 50,
    final_k: int = 5,
) -> list[str]:
    # Stage 1: fast recall
    candidates = vector_store.similarity_search(query, k=top_candidates)
    # Stage 2: accurate precision
    return rerank(query, [text for _, text in candidates], top_k=final_k)
```

**Score intuition:** Cross-encoder scores are raw logits — not 0-to-1 probabilities. A score of 8.2 for the top chunk and −3.1 for the fifth is typical. What matters is *rank*, not absolute value.

!!! tip "Cohere and Voyage reranking APIs"
    If you don't want to self-host, [Cohere Rerank](https://docs.cohere.com/docs/rerank) and [Voyage AI reranking](https://docs.voyageai.com/docs/reranker) expose cross-encoder-quality scoring as an API. Latency is comparable to self-hosted for batches under 50 documents.

---

## Contextual Compression

When retrieved chunks are long or only partially relevant, you're wasting tokens in the LLM context window on noise. Contextual compression extracts only the relevant sentences before sending context to the generator.

### Why This Matters

Suppose each chunk is 500 tokens and you retrieve 5 chunks — that's 2,500 tokens of context, potentially with only 200 tokens of signal. At gpt-4o rates, the noise costs you money and can degrade answer quality by distracting the model.

```python
def compress_context(query: str, chunks: list[str], client) -> str:
    """Extract only the sentences relevant to the query from each chunk."""
    joined = "\n\n".join(f"[Passage {i}]\n{c}" for i, c in enumerate(chunks))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise text extractor. "
                    "Given passages and a question, output ONLY the sentences "
                    "directly relevant to answering the question. "
                    "Preserve exact wording. Omit irrelevant sentences entirely. "
                    "Do not add commentary."
                ),
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nPassages:\n{joined}",
            },
        ],
    )
    return response.choices[0].message.content
```

**When compression helps most:**
- Chunking by document section (sections can be 800–2,000 tokens)
- Tables or lists where one cell is relevant but the rest is noise
- Legal / regulatory docs with boilerplate surrounding the operative clause

**When compression hurts:** If chunks are already tightly scoped (200-token sentence-window chunks), compression adds an LLM call without much gain.

---

## Multi-Step Retrieval

For complex questions, decompose into sub-queries and retrieve context for each separately before combining.

### Query Decomposition

```python
def decompose_query(user_query: str, client) -> list[str]:
    """Break a complex query into atomic sub-questions."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Break the following question into 2–4 simpler sub-questions "
                    "that can each be answered with a single retrieval step. "
                    "Return one sub-question per line."
                ),
            },
            {"role": "user", "content": user_query},
        ],
    )
    return [q.strip() for q in response.choices[0].message.content.split("\n") if q.strip()]

def multi_step_rag(
    user_query: str,
    vector_store,
    client,
    k: int = 3,
) -> str:
    sub_queries = decompose_query(user_query, client)
    all_context: list[str] = []

    for sub_q in sub_queries:
        chunks = vector_store.similarity_search(sub_q, k=k)
        all_context.extend(text for _, text in chunks)

    # Deduplicate
    seen: set[str] = set()
    unique_context = [c for c in all_context if not (c in seen or seen.add(c))]

    # Generate final answer with merged context
    context_str = "\n\n".join(f"[Source]\n{c}" for c in unique_context)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Answer the question using only the provided sources."},
            {"role": "user", "content": f"Question: {user_query}\n\n{context_str}"},
        ],
    )
    return response.choices[0].message.content
```

**Worked example — "Which plan has better support AND lower price?"**

```
Sub-query 1: "support features and SLA by plan"
  → Retrieved: "Pro plan: 24/7 chat support, 4-hour response SLA"
              "Free plan: email only, 3-day response"

Sub-query 2: "pricing by plan"
  → Retrieved: "Pro: $49/month; Free: $0"

Merged context → LLM synthesizes: "Free is cheaper ($0 vs $49) but Pro has
better support (24/7 chat + 4h SLA vs email-only). Choose based on your
support need."
```

!!! warning "Latency and cost multiply with steps"
    Each decomposition adds one LLM call. Each sub-query adds one retrieval call. For a 3-sub-query pipeline, expect 3–5× the cost of naive RAG. Gate this behind an intent classifier or use it only when single-pass Recall@5 is below threshold.

---

## Technique Selection Guide

| Technique | Latency added | Best when |
|-----------|--------------|-----------|
| HyDE | Medium (1 LLM call) | Short, ambiguous, or vocabulary-mismatched queries |
| Multi-query | Medium (1 LLM call + N searches) | Questions with multiple valid phrasings |
| Step-back | Medium | General principle needed before specific answer |
| Re-ranking | Medium (cross-encoder over K docs) | High precision required; corpus has many near-misses |
| Compression | Medium (1 LLM call) | Long chunks, tight context budget |
| Multi-step | High (1+ LLM calls, N searches) | Multi-hop, comparative, or multi-entity questions |

---

## End-to-End Advanced Pipeline

Putting it all together — a pipeline that applies query transformation, re-ranking, and contextual compression in sequence:

```python
def advanced_rag_pipeline(
    user_query: str,
    vector_store,
    client,
    top_candidates: int = 50,
    final_k: int = 5,
) -> str:
    # Step 1: Query transformation (HyDE)
    hypothetical = hyde_query(user_query, client)

    # Step 2: Vector search on the hypothetical
    raw_chunks = vector_store.similarity_search(hypothetical, k=top_candidates)
    docs = [text for _, text in raw_chunks]

    # Step 3: Re-rank with cross-encoder
    reranked = rerank(user_query, docs, top_k=final_k)

    # Step 4: Contextual compression
    compressed = compress_context(user_query, reranked, client)

    # Step 5: Generate
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Answer using only the provided context."},
            {"role": "user", "content": f"Question: {user_query}\n\nContext:\n{compressed}"},
        ],
    )
    return response.choices[0].message.content
```

This four-stage pipeline handles vocabulary mismatch (HyDE), precision (re-ranking), and token efficiency (compression). Add multi-step retrieval on top when questions require multi-hop reasoning.

---

## Common Misconceptions

**"HyDE always improves retrieval."** False. On short, well-formed queries ("what is OAuth 2.0"), the original embedding often outperforms a hypothetical. Always A/B on your eval set.

**"Re-ranking replaces vector search."** Re-ranking is too slow to replace vector search over a full corpus. It refines a *candidate set* produced by vector search — you need both stages.

**"More context is better for the LLM."** Counterintuitive but supported by research: irrelevant context in a long prompt can increase hallucination rates. Compression and tight retrieval often outperform throwing all top-50 chunks at the model.

**"Multi-step retrieval handles all multi-hop questions."** Only if decomposition produces the right sub-queries. Test your decomposition quality independently — a sub-query that misses the point produces useless context even if retrieval is perfect.

---

## Production Tips

- **Measure before adding:** Run baseline Recall@5 and faithfulness scores (Lesson 08) before adding any of these stages. Complexity is a liability.
- **Cache HyDE outputs:** Hypotheticals for common queries can be pre-cached since they're deterministic given fixed temperature=0.
- **Use async for multi-query:** Fire all N sub-queries in parallel using `asyncio.gather` — the total latency equals the slowest single search, not N × search latency.
- **Cross-encoder model selection:** `ms-marco-MiniLM-L-6-v2` is fast and accurate for general text. For domain-specific corpora (legal, medical), fine-tune on your own annotated query-document pairs.
- **Compression LLM costs:** Running `gpt-4o-mini` for compression on every query adds ~0.5–1¢ per request. At 10k daily queries, that's $50–100/day. Evaluate whether faithfulness improvement justifies the cost.

---

## Key Takeaways

- Advanced RAG adds **pre-search** (query transformation) and **post-search** (re-ranking, compression) stages around vector retrieval
- **HyDE** closes vocabulary gaps by embedding a hypothetical answer instead of the raw query
- **Cross-encoder re-ranking** improves precision over a vector-retrieved candidate pool — always use a two-stage architecture
- **Contextual compression** reduces token waste when chunks are larger than the relevant excerpt
- **Multi-step retrieval** decomposes complex questions — at the cost of multiplied latency and LLM calls
- Add complexity only when baseline eval metrics prove you need it

---

## Related Papers

| Paper | Year | Key contribution |
|-------|------|-----------------|
| [Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)](https://arxiv.org/abs/2212.10496) | 2022 | Hypothetical document embeddings for zero-shot dense retrieval |
| [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909) | 2020 | Foundational paper on jointly training retrieval and generation |
| [Improving Language Models by Retrieving from Trillions of Tokens](https://arxiv.org/abs/2112.04426) | 2021 | Retro model — retrieval at scale |
| [Benchmarking Large Language Models in Retrieval-Augmented Generation](https://arxiv.org/abs/2309.01431) | 2023 | Systematic evaluation of RAG failure modes across LLMs |

---

## Next Lesson

**[Lesson 07 — Hybrid Search](07-Hybrid-Search.md):** Combine dense embeddings with sparse BM25 to cover both semantic and exact-match queries, and learn when hybrid outperforms either method alone.
