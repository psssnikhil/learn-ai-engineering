---
title: Agentic RAG
description: >-
  Combine RAG with AI agents that decide when to retrieve, which tools to use,
  and how to refine answers iteratively
duration: 45 min
difficulty: advanced
has_code: false
module: module-09
youtube: 'https://www.youtube.com/watch?v=u5Vcrwpzoz8'
---
## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand agentic vs pipeline RAG | 45 min | Advanced |
| Design agents that control retrieval | | |
| Implement iterative retrieve-refine loops | | |
| Know when agentic RAG is worth the complexity | | |

---

## Pipeline RAG vs Agentic RAG

**Pipeline RAG** follows a fixed sequence every time:

```
Query → Retrieve → Generate → Done
```

**Agentic RAG** lets an LLM agent decide the steps:

```
Query → Agent decides:
         ├─ Need more context? → Retrieve again
         ├─ Need calculation? → Call tool
         ├─ Need clarification? → Ask user
         └─ Ready to answer → Generate
```

> **Tip:** Agentic RAG shines on multi-hop questions. For simple FAQ lookup, a fixed pipeline is faster and cheaper.

---

## Core Agent Loop

```python
def agentic_rag(query: str, tools: dict, client, max_steps: int = 5) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant. Use tools to find information "
                "before answering. If you need more context, search again."
            ),
        },
        {"role": "user", "content": query},
    ]

    for _ in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=[{"type": "function", "function": spec} for spec in tools.values()],
        )
        message = response.choices[0].message

        if not message.tool_calls:
            return message.content  # Final answer

        messages.append(message)
        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            result = execute_tool(fn_name, tool_call.function.arguments)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    return "Could not find a complete answer within the step limit."
```

---

## RAG as an Agent Tool

Expose retrieval as a tool the agent can call multiple times:

```python
tools = {
    "search_docs": {
        "name": "search_docs",
        "description": "Search the knowledge base for relevant documents",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "filters": {
                    "type": "object",
                    "description": "Optional metadata filters (department, date)",
                },
            },
            "required": ["query"],
        },
    },
    "get_document": {
        "name": "get_document",
        "description": "Fetch the full text of a document by ID",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string"},
            },
            "required": ["doc_id"],
        },
    },
}
```

The agent might:
1. Search broadly → get chunk IDs
2. Fetch full document for the most promising ID
3. Search again with refined query based on what it learned
4. Synthesize final answer

---

## Self-RAG: Reflect on Retrieval Quality

**Self-RAG** adds reflection tokens — the model evaluates whether retrieved context is useful before generating.

```
Retrieve → Is context relevant? (yes/no)
         → Is context sufficient? (yes/no)
         → If no: rewrite query and retrieve again
         → Generate → Is answer supported? (yes/no)
         → If no: retrieve more or refuse to answer
```

> **Warning:** Self-RAG adds 2–4 extra LLM calls per query. Use it when hallucination cost is high (healthcare, legal, finance).

---

## Corrective RAG (CRAG)

When retrieval quality is low, CRAG triggers a **web search fallback**:

```
Retrieve from vector DB
  ↓
Grade relevance of each chunk
  ↓
All irrelevant? → Web search → Re-retrieve
Some relevant? → Filter + generate
All relevant? → Generate directly
```

---

## When to Use Agentic RAG

| Use case | Pipeline RAG | Agentic RAG |
|----------|-------------|-------------|
| FAQ chatbot | Preferred | Overkill |
| Research assistant | Limited | Preferred |
| Multi-document comparison | Limited | Preferred |
| Low-latency support | Preferred | Too slow |
| High-stakes compliance Q&A | Risky alone | Preferred (with reflection) |

---

## Architecture Pattern

```
┌─────────────────────────────────────────┐
│              Agent (LLM)                │
│  ┌─────────┐ ┌──────────┐ ┌──────────┐ │
│  │ Retrieve│ │ Calculate│ │  Search  │ │
│  │  Tool   │ │   Tool   │ │   Web    │ │
│  └────┬────┘ └──────────┘ └──────────┘ │
└───────┼─────────────────────────────────┘
        ↓
   Vector DB + Document Store
```

---

## Recommended Videos

- [Agentic RAG Explained](https://www.youtube.com/watch?v=u5Vcrwpzoz8)
- [Self-RAG Paper Walkthrough](https://www.youtube.com/watch?v=AhGyMoEAM3A)

---

## Additional Resources

- [LangGraph Agentic RAG](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_agentic_rag/)
- [Self-RAG Paper](https://arxiv.org/abs/2310.11511)
- [Corrective RAG Paper](https://arxiv.org/abs/2401.15884)
