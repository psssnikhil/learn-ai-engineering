---
title: Agentic RAG
description: >-
  Combine RAG with AI agents that decide when to retrieve, which tools to use,
  and how to refine answers iteratively
duration: 60 min
difficulty: advanced
has_code: true
module: module-09
youtube: 'https://www.youtube.com/watch?v=u5Vcrwpzoz8'
---

## Prerequisites

- [Lesson 05 — Building a Basic RAG System](05-Building-a-Basic-RAG-System.md): standard retrieve-then-generate pipeline
- [Lesson 06 — Advanced RAG Techniques](06-Advanced-RAG-Techniques.md): multi-step retrieval and query rewriting
- Basic familiarity with function calling / tool use in LLM APIs (OpenAI tools format)
- Comfortable with Python classes and async concepts

---

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand the architectural difference between pipeline RAG and agentic RAG | 60 min | Advanced |
| Design an agent that controls its own retrieval loop | | |
| Implement Self-RAG: the model reflects on whether context is sufficient | | |
| Understand Corrective RAG (CRAG): graceful fallback to web search | | |
| Know when agentic RAG is worth the added complexity | | |

---

## Intuition First

A standard RAG pipeline is like a researcher who always: (1) searches the library catalog, (2) reads the top 5 results, (3) writes an answer. The order is fixed regardless of what the question is or what the search turned up.

An **agentic RAG** system is like a researcher who thinks: *"My first search returned tangentially related results — let me refine the query. Now I have three documents, but they conflict on the date — let me fetch the primary source. Wait, this question is about recent regulations; I need to check the current government website, not just the internal knowledge base."*

The core difference is **control flow**: in pipeline RAG, the engineer determines the sequence at design time. In agentic RAG, the LLM determines the sequence at runtime, using retrieval as a *tool* rather than a *fixed step*.

This matters when:
- The user's question requires multiple retrieval steps to gather all needed facts
- Retrieval quality is uncertain (corpora are incomplete or outdated)
- Questions have conditional structure ("If plan A is unavailable, what does plan B offer?")

---

## Pipeline RAG vs Agentic RAG

**Pipeline RAG** — fixed sequence, always the same:

```
Query → [Embed] → [Search top-K] → [Generate] → Done
```

Every query goes through this sequence whether it needs one step or ten. There's no reflection, no retry, no tool selection.

**Agentic RAG** — LLM controls the sequence:

```
Query → Agent observes query → Decides:
         ├─ Search internal KB? → Tool call → observe results
         ├─ Search is insufficient? → Rewrite query → Search again
         ├─ Need live data? → Call web search tool
         ├─ Need calculation? → Call calculator tool
         ├─ Enough context? → Generate final answer
         └─ Answer uncertain? → Reflect → Retrieve more
```

The LLM's decisions replace the engineer's hard-coded sequence.

| Dimension | Pipeline RAG | Agentic RAG |
|-----------|-------------|-------------|
| Latency | Low (1–2 API calls) | Variable (3–15 API calls) |
| Cost | Predictable | Variable |
| Handles ambiguous queries | Poorly | Well |
| Handles multi-hop questions | Limited | Strong |
| Debuggability | Easy (fixed path) | Harder (dynamic path) |
| Production complexity | Low | High |

---

## Core Agent Loop

The agent loop is the foundation of agentic RAG. The model receives a goal, has access to retrieval tools, and runs until it decides it has enough information.

```python
import json
from typing import Any

def agentic_rag(
    query: str,
    tools: dict[str, dict],
    tool_handlers: dict[str, callable],
    client,
    model: str = "gpt-4o",
    max_steps: int = 8,
) -> str:
    """
    Run an agentic RAG loop.
    
    The agent decides when to retrieve, which queries to use,
    and when it has enough context to answer.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant with access to a knowledge base. "
                "Use the search_docs tool to find relevant information before answering. "
                "If your first search doesn't return useful results, reformulate your "
                "query and search again. Only provide a final answer when you have "
                "sufficient evidence from the retrieved documents. "
                "Always cite which documents informed your answer."
            ),
        },
        {"role": "user", "content": query},
    ]

    tool_schemas = [
        {"type": "function", "function": spec}
        for spec in tools.values()
    ]

    for step in range(max_steps):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tool_schemas,
        )
        message = response.choices[0].message

        if not message.tool_calls:
            # No tool calls → model has decided it has enough context
            return message.content

        # Execute each tool call
        messages.append(message)
        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            if fn_name not in tool_handlers:
                result = json.dumps({"error": f"Unknown tool: {fn_name}"})
            else:
                result = tool_handlers[fn_name](**fn_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result if isinstance(result, str) else json.dumps(result),
            })

    return "Reached step limit without a complete answer. Partial findings above."
```

---

## RAG as an Agent Tool

Expose your retrieval system as a function the agent can call with different queries and filters:

```python
RETRIEVAL_TOOLS = {
    "search_docs": {
        "name": "search_docs",
        "description": (
            "Search the internal knowledge base for relevant documents. "
            "Use for questions about company policies, product features, "
            "API documentation, or internal procedures. "
            "Returns the most relevant document excerpts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Specific search query. Be precise — shorter, focused queries work better than long questions.",
                },
                "filters": {
                    "type": "object",
                    "description": "Optional metadata filters.",
                    "properties": {
                        "category": {"type": "string", "description": "Document category, e.g. 'billing', 'api', 'security'"},
                        "updated_after": {"type": "string", "description": "ISO date string — only return documents updated after this date"},
                    },
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return (1–10). Default 5.",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["query"],
        },
    },
    "get_document": {
        "name": "get_document",
        "description": "Fetch the full text of a specific document by its ID. Use when a search result looks promising but you need more context than the excerpt provides.",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "Document ID from a previous search_docs result.",
                },
            },
            "required": ["doc_id"],
        },
    },
}


def make_tool_handlers(vector_store, document_store):
    """Build tool handler functions backed by real retrieval."""

    def search_docs(query: str, filters: dict | None = None, top_k: int = 5) -> str:
        results = vector_store.similarity_search(
            query, k=top_k, filter=filters or {}
        )
        if not results:
            return json.dumps({"results": [], "message": "No relevant documents found. Try a different query."})
        return json.dumps({
            "results": [
                {"doc_id": doc_id, "excerpt": text[:500], "score": score}
                for doc_id, text, score in results
            ]
        })

    def get_document(doc_id: str) -> str:
        doc = document_store.get(doc_id)
        if not doc:
            return json.dumps({"error": f"Document {doc_id} not found."})
        return json.dumps({"doc_id": doc_id, "content": doc["text"]})

    return {"search_docs": search_docs, "get_document": get_document}
```

**Trace of a multi-step agentic retrieval:**

```
User: "What's the refund policy for annual plans, and how does it differ from monthly?"

Step 1: search_docs("refund policy annual plan") → 3 results about billing
Step 2: search_docs("refund policy monthly subscription") → 2 results
Step 3: get_document("billing_faq_003") → Full FAQ text with exact refund terms
Step 4: [No tool call] → Final answer comparing annual (pro-rated within 30 days)
         vs monthly (no refunds, cancel next cycle)
```

Three tool calls; a pipeline RAG with a single search would have missed the monthly policy document entirely.

---

## Self-RAG: Reflection Tokens

**Self-RAG** (Asai et al., 2023) introduces *reflection tokens* — special tokens the model emits to evaluate its own retrieval process:

```
Retrieve? [Yes/No] → If Yes: Search → 
IsRel? [Relevant/Irrelevant] for each chunk → 
IsSup? [Fully/Partially/No support] → 
IsUse? [Useful 1-5]
```

You can approximate Self-RAG without fine-tuning by adding explicit reflection prompts:

```python
def self_rag_loop(
    query: str,
    retriever,
    client,
    max_rounds: int = 3,
) -> str:
    """Retrieve → reflect → optionally re-retrieve."""
    
    for round_num in range(max_rounds):
        chunks = retriever.retrieve(query, k=5)
        context = "\n\n".join(chunks)

        # Reflect: is this context sufficient?
        reflection = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Question: {query}\n\n"
                        f"Retrieved context:\n{context}\n\n"
                        "Assess the retrieved context:\n"
                        "1. Is any retrieved passage directly relevant to the question? (yes/no)\n"
                        "2. Is there enough information to answer completely? (yes/no)\n"
                        "3. If not, what specific information is missing?\n"
                        "Reply as JSON: {\"relevant\": bool, \"sufficient\": bool, \"missing\": str}"
                    ),
                }
            ],
            response_format={"type": "json_object"},
        )
        assessment = json.loads(reflection.choices[0].message.content)

        if assessment["sufficient"]:
            # Context is good — generate the final answer
            answer_resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Answer using only the context provided."},
                    {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"},
                ],
            )
            return answer_resp.choices[0].message.content

        # Refine query based on what's missing
        if assessment.get("missing"):
            query = f"{query} {assessment['missing']}"

    # Fallback: best effort with what we have
    answer_resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Answer as best you can with the context. Note any gaps."},
            {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"},
        ],
    )
    return answer_resp.choices[0].message.content
```

!!! warning "Self-RAG latency and cost"
    Each reflection round adds one LLM call. With 3 rounds maximum, you're looking at 4–7 API calls per query versus 1–2 for naive RAG. Self-RAG pays off when the cost of a wrong or hallucinated answer is higher than the cost of extra API calls — healthcare, legal, compliance, financial advice.

---

## Corrective RAG (CRAG)

CRAG (Yan et al., 2024) adds a *relevance evaluator* that triggers a web search fallback when the knowledge base returns poor results.

```
Retrieve from vector DB
    ↓
Grade each chunk: [CORRECT / AMBIGUOUS / INCORRECT]
    ↓
All INCORRECT → Web search fallback → Re-retrieve from web → Refine → Generate
Some CORRECT  → Filter correct chunks → Generate  
All CORRECT   → Generate directly
```

```python
def corrective_rag(
    query: str,
    kb_retriever,
    web_search_fn,
    client,
) -> str:
    """RAG with automatic fallback to web search when KB retrieval is poor."""
    
    # Step 1: Retrieve from knowledge base
    kb_chunks = kb_retriever.retrieve(query, k=5)

    # Step 2: Grade each chunk
    graded_chunks = []
    for chunk in kb_chunks:
        grade_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Query: {query}\n\nDocument excerpt: {chunk}\n\n"
                        "Is this excerpt directly relevant and useful for answering the query? "
                        "Reply: CORRECT, AMBIGUOUS, or INCORRECT"
                    ),
                }
            ],
        )
        grade = grade_resp.choices[0].message.content.strip().upper()
        graded_chunks.append((chunk, grade))

    correct = [c for c, g in graded_chunks if g == "CORRECT"]
    ambiguous = [c for c, g in graded_chunks if g == "AMBIGUOUS"]

    # Step 3: Decide context source
    if not correct and not ambiguous:
        # All chunks irrelevant → fall back to web
        web_results = web_search_fn(query, max_results=5)
        context = "\n\n".join(web_results)
        source_note = "[Source: web search]"
    elif correct:
        context = "\n\n".join(correct[:3])
        source_note = "[Source: knowledge base]"
    else:
        # Ambiguous — use KB but flag uncertainty
        context = "\n\n".join(ambiguous[:3])
        source_note = "[Source: knowledge base — low confidence]"

    # Step 4: Generate
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"Answer using the context. {source_note}",
            },
            {
                "role": "user",
                "content": f"Question: {query}\n\nContext:\n{context}",
            },
        ],
    )
    return response.choices[0].message.content
```

---

## LangGraph: Agentic RAG as a State Machine

For production-grade agentic RAG, [LangGraph](https://langchain-ai.github.io/langgraph/) models the retrieval loop as an explicit state graph — each node is a processing step, each edge is a conditional transition:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    question: str
    retrieved_chunks: list[str]
    answer: str
    retrieval_sufficient: bool
    steps: int

def retrieve_node(state: AgentState) -> AgentState:
    chunks = retriever.retrieve(state["question"], k=5)
    return {**state, "retrieved_chunks": chunks}

def grade_retrieval(state: AgentState) -> AgentState:
    # Simplified grading
    is_sufficient = len(state["retrieved_chunks"]) >= 3
    return {**state, "retrieval_sufficient": is_sufficient, "steps": state["steps"] + 1}

def generate_node(state: AgentState) -> AgentState:
    context = "\n\n".join(state["retrieved_chunks"])
    answer = llm_generate(state["question"], context)
    return {**state, "answer": answer}

def rewrite_query_node(state: AgentState) -> AgentState:
    # Rewrite question for better retrieval
    new_question = llm_rewrite(state["question"])
    return {**state, "question": new_question}

def should_continue(state: AgentState) -> str:
    if state["retrieval_sufficient"] or state["steps"] >= 3:
        return "generate"
    return "rewrite"

# Build graph
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("grade", grade_retrieval)
workflow.add_node("rewrite", rewrite_query_node)
workflow.add_node("generate", generate_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_conditional_edges("grade", should_continue, {"generate": "generate", "rewrite": "rewrite"})
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("generate", END)

graph = workflow.compile()
result = graph.invoke({"question": "What is the API rate limit?", "steps": 0})
```

The graph representation makes agentic RAG debuggable — you can inspect state at every node, replay failed runs, and add human-in-the-loop approval at any edge.

---

## When to Use Agentic RAG

| Use case | Pipeline RAG | Agentic RAG |
|----------|-------------|-------------|
| Simple FAQ chatbot | Preferred — fast, cheap, predictable | Overkill |
| Multi-document comparison | Limited | Preferred |
| Research assistant | Limited | Preferred |
| Low-latency customer support | Preferred | Too slow |
| High-stakes compliance Q&A | Risky alone | Preferred (with Self-RAG reflection) |
| Outdated KB, need web fallback | Fails silently | CRAG pattern |
| Real-time data requirements | Fails | Preferred (with web/API tools) |

**Decision heuristic:** If your baseline RAG pipeline achieves Recall@5 > 0.80 and users are satisfied, you don't need agentic RAG. Add it when you have documented evidence that single-step retrieval misses multi-hop or conditional questions.

---

## Common Misconceptions

**"Agentic RAG is always better."** It's almost always more expensive, slower, and harder to debug. Use it only when pipeline RAG fails measurably on your eval set.

**"More retrieval steps improve accuracy."** Adding more steps without a convergence criterion creates infinite retrieval loops that waste money and don't improve answers. Always set a `max_steps` budget.

**"Self-RAG requires a fine-tuned model."** The paper describes a fine-tuned model, but the *pattern* (retrieve → reflect → re-retrieve) works with any instruction-following model using reflection prompts. Results may differ from the paper's benchmarks.

**"The agent controls hallucination."** The agent controls *retrieval* — it can fetch better context. But if the LLM ignores context or the context is wrong, the agent doesn't fix hallucination on its own. Combine with faithfulness checks (Lesson 08) for high-stakes applications.

---

## Production Tips

- **Step budget is mandatory:** Even a prototype needs `max_steps=5`. Without it, a retrieval loop can run for minutes burning API credits.
- **Trace every step:** Log which tool was called, what query was used, how many results came back, and the reflection outcome. Debugging agentic systems without traces is nearly impossible.
- **Cache common intermediate retrievals:** Many agentic queries start with similar first-step searches. Cache retrieval results (not final answers) with a short TTL (5–15 minutes) to avoid redundant searches.
- **Start with one tool:** Begin with just `search_docs`. Add `get_document`, `search_web`, and calculation tools only after proving the single-tool loop handles your eval set adequately.
- **Test for infinite loops:** Add a test case where the knowledge base has no relevant documents. The agent should reach `max_steps` gracefully and return a partial answer, not run forever.

---

## Key Takeaways

- Agentic RAG gives the LLM control over the retrieval loop — it decides when, what, and how often to search
- **Self-RAG** adds explicit reflection: the model evaluates whether retrieved context is sufficient before generating
- **CRAG** adds a web search fallback triggered when the KB returns irrelevant results — keeps the system useful even for out-of-scope queries
- **LangGraph** models the retrieval loop as a state machine — making complex agentic flows debuggable and resumable
- Agentic RAG is worth the complexity for multi-hop, comparative, or conditional questions; stick with pipeline RAG for simple FAQ use cases
- Always set a `max_steps` budget and log every tool call — debuggability is the primary operational challenge

---

## Related Papers

| Paper | Year | Key contribution |
|-------|------|-----------------|
| [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511) | 2023 | Fine-tuned model with reflection tokens for retrieve-then-critique |
| [Corrective Retrieval Augmented Generation](https://arxiv.org/abs/2401.15884) | 2024 | Web search fallback triggered by retrieval relevance scoring |
| [Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG](https://arxiv.org/abs/2501.09136) | 2025 | Taxonomy and survey of agentic RAG architectures |
| [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) | 2022 | Foundational ReAct pattern underlying agentic tool use |

---

## Next Lesson

**[Lesson 10 — RAG in Production](10-RAG-in-Production.md):** Deploy, monitor, and maintain RAG systems at scale — index versioning, caching strategies, observability pipelines, and cost control.
