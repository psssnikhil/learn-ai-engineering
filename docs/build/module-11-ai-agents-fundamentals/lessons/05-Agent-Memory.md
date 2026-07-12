---
title: Agent Memory Systems
description: >-
  Learn how AI agents use short-term, long-term, and episodic memory to maintain
  context and improve over time
duration: 35 min
difficulty: advanced
has_code: false
module: module-11
objectives:
  - Explain the three types of agent memory
  - Implement a conversation buffer memory
  - Build a vector-based long-term memory system
  - Design a memory retrieval strategy for an agent
---
# Agent Memory Systems

## Learning Objectives

By the end of this lesson, you will be able to:
- Distinguish between short-term, long-term, and episodic memory
- Implement conversation memory with sliding windows and summaries
- Build vector-store-backed long-term memory
- Design memory retrieval strategies that balance relevance and recency

---

## Why Agents Need Memory

Without memory, every agent interaction starts from zero. The agent cannot:
- Remember what the user said 5 messages ago
- Learn from past mistakes
- Build up context over a long task
- Recall facts from previous sessions

Memory transforms a stateless LLM into a persistent, context-aware agent.

---

## 1. Short-Term Memory (Working Memory)

Short-term memory holds the current conversation context. It is the most common form of agent memory -- simply the list of messages passed to the LLM.

### The Problem: Context Window Limits

LLMs have finite context windows (4K-200K tokens). A long conversation will eventually exceed the limit.

### Solution 1: Sliding Window

Keep only the most recent N messages:

```python
class SlidingWindowMemory:
    def __init__(self, max_messages: int = 20):
        self.messages = []
        self.max_messages = max_messages

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def get_messages(self) -> list[dict]:
        return list(self.messages)

    def clear(self):
        self.messages = []
```

### Solution 2: Summary Memory

Periodically summarize older messages to compress the context:

```python
from openai import OpenAI

client = OpenAI()

class SummaryMemory:
    def __init__(self, max_messages: int = 10):
        self.summary = ""
        self.recent_messages = []
        self.max_messages = max_messages

    def add(self, role: str, content: str):
        self.recent_messages.append({"role": role, "content": content})

        if len(self.recent_messages) > self.max_messages:
            self._compress()

    def _compress(self):
        """Summarize older messages and keep only recent ones."""
        old_messages = self.recent_messages[:-5]
        old_text = "
".join(
            f"{m['role']}: {m['content']}" for m in old_messages
        )

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{
                "role": "user",
                "content": f"Summarize this conversation concisely, preserving key facts and decisions:

{old_text}"
            }],
            max_tokens=300,
        )

        new_summary = response.choices[0].message.content
        if self.summary:
            self.summary = f"{self.summary}
{new_summary}"
        else:
            self.summary = new_summary

        self.recent_messages = self.recent_messages[-5:]

    def get_messages(self) -> list[dict]:
        messages = []
        if self.summary:
            messages.append({
                "role": "system",
                "content": f"Conversation summary so far:
{self.summary}"
            })
        messages.extend(self.recent_messages)
        return messages
```

---

## 2. Long-Term Memory (Semantic Memory)

Long-term memory persists across sessions. It stores facts, preferences, and knowledge the agent has learned. The most effective approach uses a vector database for semantic retrieval.

```python
import chromadb
import uuid

class LongTermMemory:
    def __init__(self, collection_name: str = "agent_memory"):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def store(self, text: str, metadata: dict = None):
        """Store a memory with automatic embedding."""
        self.collection.add(
            ids=[str(uuid.uuid4())],
            documents=[text],
            metadatas=[metadata or {}],
        )

    def recall(self, query: str, n_results: int = 5) -> list[str]:
        """Retrieve the most relevant memories for a query."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
        )
        return results["documents"][0] if results["documents"] else []

    def recall_with_metadata(self, query: str, n_results: int = 5) -> list[dict]:
        """Retrieve memories with their metadata."""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
        )
        memories = []
        for i, doc in enumerate(results["documents"][0]):
            memories.append({
                "text": doc,
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return memories

# Usage
memory = LongTermMemory()
memory.store("User prefers Python over JavaScript", {"type": "preference"})
memory.store("User is building a RAG application", {"type": "context"})
memory.store("User's deadline is March 15th", {"type": "fact"})

relevant = memory.recall("What language should I write the code in?")
# Returns: ["User prefers Python over JavaScript"]
```

---

## 3. Episodic Memory

Episodic memory stores complete interaction sequences -- past tasks the agent completed, including what worked and what failed. This enables learning from experience.

```python
from datetime import datetime

class EpisodicMemory:
    def __init__(self):
        self.episodes = []

    def record_episode(self, task: str, steps: list[dict],
                       outcome: str, success: bool):
        """Record a complete task episode."""
        self.episodes.append({
            "task": task,
            "steps": steps,
            "outcome": outcome,
            "success": success,
            "timestamp": datetime.now().isoformat(),
        })

    def recall_similar(self, task: str, n: int = 3) -> list[dict]:
        """Find past episodes similar to the current task.

        In production, use vector similarity. Here we use simple keyword overlap.
        """
        task_words = set(task.lower().split())
        scored = []
        for ep in self.episodes:
            ep_words = set(ep["task"].lower().split())
            overlap = len(task_words & ep_words) / max(len(task_words), 1)
            scored.append((overlap, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ep for _, ep in scored[:n]]

    def get_successful_strategies(self, task: str) -> list[dict]:
        """Retrieve only successful past episodes."""
        similar = self.recall_similar(task)
        return [ep for ep in similar if ep["success"]]
```

---

## 4. Combining Memory Types in an Agent

A production agent uses all three memory types together:

```python
class MemoryEnabledAgent:
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.short_term = SummaryMemory(max_messages=10)
        self.long_term = LongTermMemory()
        self.episodic = EpisodicMemory()

    def run(self, user_input: str) -> str:
        # 1. Retrieve relevant long-term memories
        memories = self.long_term.recall(user_input, n_results=3)
        memory_context = "
".join(f"- {m}" for m in memories)

        # 2. Retrieve relevant past episodes
        past_tasks = self.episodic.get_successful_strategies(user_input)
        episode_context = ""
        if past_tasks:
            episode_context = f"
Relevant past experience:
"
            for ep in past_tasks[:2]:
                episode_context += f"- Task: {ep['task']} -> {ep['outcome']}
"

        # 3. Build messages with all memory types
        messages = [
            {"role": "system", "content": (
                f"{self.system_prompt}

"
                f"Known facts about the user:
{memory_context}
"
                f"{episode_context}"
            )}
        ]
        messages.extend(self.short_term.get_messages())
        messages.append({"role": "user", "content": user_input})

        # 4. Call the LLM
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
        )
        reply = response.choices[0].message.content

        # 5. Update short-term memory
        self.short_term.add("user", user_input)
        self.short_term.add("assistant", reply)

        return reply
```

---

## Memory Design Decisions

| Decision | Options | Trade-off |
|----------|---------|-----------|
| **When to store** | Every turn vs. key facts only | Noise vs. completeness |
| **How to retrieve** | Recency vs. relevance vs. both | Speed vs. accuracy |
| **What to forget** | Never vs. TTL vs. importance score | Storage vs. relevance |
| **Where to store** | In-memory vs. vector DB vs. SQL | Speed vs. persistence |

---

## Key Takeaways

- **Short-term memory** (sliding window or summary) handles the current conversation
- **Long-term memory** (vector store) persists facts and preferences across sessions
- **Episodic memory** records complete task experiences for learning
- Combine all three types for agents that feel context-aware and capable
- Always consider what to forget -- not all information is worth storing

## Resources

- [LangChain Memory documentation](https://python.langchain.com/docs/concepts/memory/) -- Built-in memory types
- [Letta (formerly MemGPT)](https://github.com/letta-ai/letta) -- Research on LLM self-managed memory
- [Zep: Long-term memory for AI assistants](https://www.getzep.com/) -- Production memory infrastructure

---

Next: Planning & Reasoning
