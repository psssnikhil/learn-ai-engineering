---
title: Introduction to RAG Systems
description: 'Understand what RAG is, why it''s essential, and how it solves LLM limitations'
duration: 50 min
difficulty: beginner
has_code: true
module: module-09
youtube: 'https://www.youtube.com/watch?v=T-D1OfcDW1M'
---

# Introduction to RAG Systems

## Prerequisites

Before this lesson you should be comfortable with:

- **Prompting basics** — how to write a system prompt and user message (Module 08)
- **What an LLM is** — transformer model trained to predict tokens (Module 01–02)
- **Python functions** — enough to read and run the code examples below

You do not need prior knowledge of databases or information retrieval.

---

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Explain the knowledge-cutoff problem and why fine-tuning alone doesn't solve it | 10 min | Beginner |
| Describe RAG's three-stage pipeline at the whiteboard level | 15 min | Beginner |
| Contrast RAG with fine-tuning and in-context stuffing | 10 min | Beginner |
| Run a minimal RAG loop in Python | 15 min | Beginner |

---

## Intuition First: The Library Analogy

Imagine you hired a brilliant analyst who has read millions of books but was locked in a room with no new material after 2023. Ask them about the Q3 2024 earnings call — they will confidently make something up, because they have no access to that document.

Now give that analyst a library card and a research assistant. Instead of recalling from memory, they:

1. Hand the question to the assistant ("find documents about Q3 2024 earnings").
2. Read the retrieved pages.
3. Draft an answer citing what they just read.

That is RAG. The LLM is still the analyst (the reasoner). The vector database is the library. The retrieval step is the research assistant that fetches the right pages before the analyst starts writing.

The crucial insight: **the LLM does not need to memorize facts it can look up**. Its job is to *reason over context*, not *store context* in its weights.

---

## The Problem with Pure LLMs

Large language models are trained on a snapshot of the web up to a cutoff date. That training produces impressive general knowledge, but it creates four structural limitations:

**1. Knowledge cutoff**
Training data has a hard stop. Events after the cutoff simply don't exist inside the model. Asking GPT-4 about something that happened last week returns either a refusal or a hallucination.

**2. Hallucinations on private data**
Your internal documents, customer records, and proprietary databases were never in the training set. The model cannot know them — but it will still generate plausible-sounding answers if you ask.

**3. Hallucinations as a general failure mode**
Even on topics covered in training, LLMs sometimes generate false statements that sound confident. When the model doesn't have crisp facts, it interpolates from patterns. That interpolation can be wrong.

**4. Fine-tuning is expensive and static**
You could bake new knowledge into the model weights via fine-tuning, but retraining costs thousands of dollars and takes days. Worse, knowledge baked into weights is still frozen — next month you run the same problem again.

```
Without RAG — asking about private or recent information:

User: "What were the key points from our board meeting last Tuesday?"
LLM:  "I'm sorry, I don't have access to information about your specific
       organization's internal meetings..."

Or worse:
LLM:  "Your board discussed a 12% revenue increase and approved the expansion
       into Southeast Asia." ← completely fabricated
```

---

## RAG to the Rescue

**Retrieval-Augmented Generation (RAG)** solves these problems by injecting relevant documents into the LLM prompt *at inference time*, before the model generates its response.

The formula:

\[
\text{RAG}(q) = \text{LLM}\!\left(\,q \;\|\; \text{Retrieve}(q, \mathcal{D})\,\right)
\]

where \(q\) is the user query, \(\mathcal{D}\) is your document corpus, and \(\|\) means concatenation into the prompt. The model sees the question *and* the evidence before it writes a single token.

```
With RAG — same question, different pipeline:

User query: "What were the key points from our board meeting last Tuesday?"
        ↓
Retrieve top-3 chunks from meeting-notes corpus (semantic search)
        ↓
Context injected into prompt:
  "[Chunk 1] Board meeting 2024-07-09: Revenue up 8% YoY. CFO raised
   concerns about APAC margins..."
   "[Chunk 2] Action items: CEO to present expansion plan by Q3 end..."
        ↓
LLM reads context + question → generates grounded answer
        ↓
"Based on the July 9th board notes, the key points were: (1) 8% YoY
 revenue growth, (2) CFO concerns about APAC margins, (3) CEO deadline..."
```

The LLM didn't memorize this. It read it — just like your analyst with the library card.

---

## The Three-Stage Pipeline

Every RAG system has two offline stages and one online stage:

```
OFFLINE (build once, update as docs change)
─────────────────────────────────────────────
Documents
    │
    ▼
[1. INGEST]  Parse PDFs, HTML, databases → clean text
    │
    ▼
[2. CHUNK]   Split text into 300-800 token pieces
    │
    ▼
[3. EMBED]   Convert each chunk → dense vector (1536 dims)
    │
    ▼
[4. INDEX]   Store vectors in a vector database (Chroma, Pinecone, pgvector)


ONLINE (runs on every user query, ~200–400 ms added latency)
─────────────────────────────────────────────────────────────
User query
    │
    ▼
[5. EMBED QUERY]   Same embedding model → query vector
    │
    ▼
[6. RETRIEVE]      ANN search → top-K most similar chunks
    │
    ▼
[7. AUGMENT]       Insert chunks into prompt as context
    │
    ▼
[8. GENERATE]      LLM reads context → writes grounded response
    │
    ▼
Answer (with optional citations)
```

The offline pipeline runs once (or on updates). The online pipeline runs in milliseconds for every query.

---

## Worked Example: Step by Step

Let's trace a concrete query through the pipeline manually.

**Setup**: You have three document chunks indexed:

| Chunk ID | Text |
|----------|------|
| C1 | "Refunds are available within 30 days of purchase. Items must be unused and in original packaging." |
| C2 | "Our standard shipping time is 5-7 business days. Expedited shipping costs USD 15." |
| C3 | "Contact support at support@example.com or call 1-800-555-0199 during business hours." |

**Query**: "Can I return something I bought 3 weeks ago?"

**Step 1 — Embed query**

The query is converted to a 1536-dimension vector, e.g.  
`query_vec = [0.12, -0.34, 0.87, ..., 0.05]`

**Step 2 — ANN search**

The vector database computes cosine similarity between `query_vec` and all stored chunk vectors:

| Chunk | Cosine Similarity |
|-------|-----------------|
| C1 | 0.91 ← closest |
| C3 | 0.52 |
| C2 | 0.38 |

Top-1 retrieved: C1.

**Step 3 — Augmented prompt**

```
System: Answer using only the provided context. If the answer
        isn't in the context, say "I don't have that information."

Context:
[C1] Refunds are available within 30 days of purchase.
     Items must be unused and in original packaging.

User: Can I return something I bought 3 weeks ago?
```

**Step 4 — Generation**

```
LLM: Yes — 3 weeks is 21 days, which is within the 30-day return
     window. Make sure the item is unused and in its original
     packaging. You can initiate the return through your account
     portal.
```

The LLM correctly computed 21 < 30 and cited the policy. It did not hallucinate a policy that doesn't exist.

---

## Minimal Implementation

```python
from openai import OpenAI
import numpy as np

client = OpenAI()

# --- OFFLINE: Build a tiny knowledge base ---

docs = [
    "Refunds are available within 30 days of purchase. Items must be unused.",
    "Standard shipping takes 5-7 business days. Expedited costs USD 15.",
    "Contact support@example.com or call 1-800-555-0199.",
]

def embed(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding

# Pre-compute embeddings for every document chunk
doc_embeddings = [embed(doc) for doc in docs]


# --- ONLINE: Handle a user query ---

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def retrieve(query: str, top_k: int = 2) -> list[str]:
    query_vec = embed(query)
    scored = [
        (cosine_similarity(query_vec, doc_vec), doc)
        for doc_vec, doc in zip(doc_embeddings, docs)
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]

def rag_answer(question: str) -> str:
    chunks = retrieve(question)
    context = "\n\n".join(chunks)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using ONLY the provided context. "
                    "If the answer isn't in the context, say "
                    "'I don't have that information.'"
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    return response.choices[0].message.content


# Test it
print(rag_answer("Can I return something I bought 3 weeks ago?"))
# → "Yes, 3 weeks (21 days) is within the 30-day return window..."

print(rag_answer("What is the CEO's salary?"))
# → "I don't have that information."
```

!!! note "Why not just stuff all documents into the prompt?"
    If you have 500 documents you can't fit them all. Even with 200K-token context windows, costs scale linearly with tokens — and attention quality degrades on very long contexts. Retrieval gives you precision: only the relevant 2-5 chunks reach the LLM.

---

## RAG vs Fine-Tuning vs In-Context Stuffing

| Strategy | How it works | Cost | Update speed | Best for |
|----------|-------------|------|--------------|----------|
| **Pure LLM** | Model's training weights | Zero extra | Months (retrain) | General knowledge |
| **In-context stuffing** | Paste entire document corpus in prompt | High (many tokens) | Instant | Small corpuses, < 20 pages |
| **Fine-tuning** | Bake new data into weights | Very high (GPU hours) | Days–weeks | Consistent style/format changes |
| **RAG** | Retrieve relevant chunks at query time | Low (embedding + retrieval) | Instant (add docs) | Large knowledge bases, fresh data |

**Why not fine-tune instead of RAG?**
Fine-tuning teaches the model style and behavior patterns. It does not reliably teach facts — studies show that facts learned via fine-tuning are fragile and hard to update. RAG separates *knowledge* (the document store) from *reasoning* (the LLM), making each independently updatable.

**Why not stuff the whole database into the prompt?**
Token cost is the limiting factor. At 1 million tokens × USD 0.15/1M = USD 0.15 *per query* for GPT-4o. For thousands of daily queries over a large corpus this is prohibitive. Retrieval narrows to the relevant 2-5 chunks for a fraction of the cost.

---

## Edge Cases & Common Misconceptions

**Misconception 1: RAG eliminates hallucinations.**
RAG reduces hallucinations by grounding answers in retrieved text — but the LLM can still misread, misquote, or extrapolate beyond the context. Grounding instruction ("answer only from context") helps but doesn't fully prevent drift.

**Misconception 2: More retrieved chunks = better answers.**
Retrieving 20 chunks floods the context with noise. Studies show retrieval precision peaks at K=3–5. Beyond that, off-topic chunks confuse the model.

**Misconception 3: Any embedding model works equally well.**
Embedding quality is the foundation. A weak embedding model that cannot capture the semantic relationship between "return policy" and "Can I get a refund?" will fail at retrieval entirely. Benchmarking on your domain data matters.

**Misconception 4: You should always use RAG.**
For a document that fits in a single prompt (< 50 pages), just paste it in. RAG's overhead — latency, infrastructure, embedding cost — is only justified when your corpus is large or frequently updated.

!!! warning "The retrieval ceiling"
    If the correct answer is not in the indexed documents, no amount of generation quality will produce it. RAG raises the ceiling on accuracy; it does not conjure missing information. Always audit your corpus for coverage gaps.

---

## Production Connection

In production, RAG is the dominant architecture for enterprise knowledge assistants, customer support bots, legal research tools, and internal wikis. Key engineering decisions you will face:

- **Embedding model selection** — OpenAI `text-embedding-3-small` for cost efficiency; `text-embedding-3-large` or Cohere for higher precision on technical domains.
- **Vector database** — Chroma or SQLite-vec for < 100K documents; Pinecone, Qdrant, or Weaviate for millions of vectors.
- **Chunk size tuning** — 300 tokens for precise factual retrieval; 800 tokens for narrative documents. Always measure Recall@5 on a held-out eval set.
- **Reranking** — add a cross-encoder reranker (e.g., Cohere Rerank) after first-pass retrieval to improve precision.
- **Latency budget** — embedding a query takes ~20 ms; ANN search ~5–50 ms; total added overhead before the LLM call is typically 50–200 ms.

A well-tuned RAG system routinely achieves answer accuracy of 85–95% on factual Q&A benchmarks, compared to 40–60% for the same LLM without retrieval on domain-specific questions.

---

## Key Takeaways

- RAG solves the knowledge-cutoff and private-data problems by retrieving relevant documents at inference time rather than baking knowledge into weights.
- The pipeline has two offline stages (chunk + embed + index) and one online stage (embed query → retrieve → generate).
- Grounding the LLM in retrieved context dramatically reduces hallucination rates on factual questions.
- RAG is cheaper and faster to update than fine-tuning; knowledge lives in the document store, not in model weights.
- Retrieval quality is the critical bottleneck — a great LLM with poor retrieval still fails; measure Recall@K on your own data.
- Use K=3–5 retrieved chunks; more chunks add noise faster than signal.
- RAG is not a silver bullet — if documents don't contain the answer, the system cannot conjure it.

---

## Related Papers

| Paper | What it contributes | Link |
|-------|-------------------|------|
| Lewis et al. (2020) — *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* | Introduces the RAG framework and shows it outperforms pure LMs on open-domain QA | [arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401) |
| Guu et al. (2020) — *REALM: Retrieval-Enhanced Language Model Pre-Training* | Pre-trains the retriever jointly with the LM end-to-end | [arxiv.org/abs/2002.08909](https://arxiv.org/abs/2002.08909) |
| Izacard & Grave (2021) — *Leveraging Passage Retrieval with Generative Models for Open Domain QA* | FiD (Fusion-in-Decoder): encodes each passage separately, fuses in decoder — large accuracy gains | [arxiv.org/abs/2007.01282](https://arxiv.org/abs/2007.01282) |
| Shi et al. (2023) — *REPLUG: Retrieval-Augmented Language Model Pre-Training* | Shows retrieval can improve any black-box LLM via ensemble scoring | [arxiv.org/abs/2301.12652](https://arxiv.org/abs/2301.12652) |

---

## RAG Failure Modes: A Diagnostic Map

Understanding *why* a RAG system gives a wrong answer is the first step to fixing it. Most failures fall into one of three categories:

**Retrieval failure** — the correct document exists but was not returned in top-K.
- Symptom: LLM says "I don't have information" but the answer is in your corpus.
- Root cause: embedding model doesn't capture the query-document relationship; chunk too large; embedding model not domain-tuned.
- Fix: measure Recall@5; try hybrid search; reduce chunk size.

**Context failure** — the correct document was retrieved, but the answer spans multiple chunks and no single chunk contains the complete answer.
- Symptom: LLM gives a partial or hedged answer; "the document says X but not Y."
- Root cause: relevant information was split across a chunk boundary.
- Fix: increase overlap; use parent-document retrieval; increase K.

**Generation failure** — the correct context was retrieved and passed to the LLM, but the LLM still gives a wrong answer.
- Symptom: LLM contradicts the retrieved context or adds information from training.
- Root cause: weak grounding instruction; temperature too high; model ignores context under pressure.
- Fix: strengthen system prompt grounding instruction; lower temperature to 0.0; use a better model.

Keep these three categories in mind: they have different symptoms and different fixes. The most common mistake is tweaking the prompt when the problem is actually in retrieval.

---

## Further Reading

- [RAG from Scratch](https://www.youtube.com/watch?v=wd7TZ4w1mSw) — LangChain's full video tutorial series walking through every component
- [What is Retrieval Augmented Generation?](https://www.youtube.com/watch?v=T-D1OfcDW1M) — IBM Technology clear explainer (8 min)
- [Pinecone: What is RAG?](https://www.pinecone.io/learn/retrieval-augmented-generation/) — Pinecone's illustrated guide with architecture diagrams
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/) — Hands-on official walkthrough

---

## Next Lesson

**[Lesson 2: Vector Databases & Embeddings](02-vector-databases.md)** — Learn how embeddings encode meaning as geometry, and how vector databases search millions of vectors in milliseconds.
