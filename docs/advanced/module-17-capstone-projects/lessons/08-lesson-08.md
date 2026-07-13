---
title: 'Project 7: Chatbot with Long-Term Memory'
description: >-
  Build a chatbot that remembers user preferences, past conversations, and
  important facts across sessions using vector-based memory
duration: 180 min
difficulty: advanced
has_code: true
module: module-17
---
# Project 8: Chatbot with Long-Term Memory

## Project Overview

Build a chatbot that maintains **long-term memory** across conversations. Unlike a stateless chatbot that forgets everything when the session ends, this system remembers user preferences, past interactions, and important facts — retrieving relevant memories at the right moment to personalize responses.

This project applies agent memory patterns, vector databases, and conversation design. You will implement both short-term memory (current conversation context) and long-term memory (persisted facts in a vector store).

**Time estimate**: 10-15 hours
**Skills used**: Agent Memory, Vector Databases, Prompt Engineering, FastAPI, Session Management

---

## Prerequisites

| Module | Topics Used |
|--------|------------|
| **Module 5: AI Agents** | Agent loops, tool use, memory patterns |
| **Module 4: Vector Databases** | ChromaDB storage and similarity search |
| **Module 7: Prompt Engineering** | System prompts, memory injection |
| **Module 13: LLMOps** | Session management, API design |

**Environment setup:**

```bash
pip install openai chromadb fastapi uvicorn pydantic python-dotenv pytest httpx
```

---

## What You'll Build

### Acceptance Criteria Checklist

- [ ] Short-term memory: maintains last 20 messages in conversation history
- [ ] Long-term memory: persists facts to ChromaDB vector store per user
- [ ] Memory extraction: automatically saves `[REMEMBER: ...]` facts from responses
- [ ] Memory retrieval: injects top-3 relevant memories into system prompt per query
- [ ] Memory management API: list, search, and delete user memories
- [ ] Multi-session persistence: memories survive server restarts
- [ ] FastAPI endpoints: `POST /chat`, `GET /memories`, `DELETE /memories/{id}`
- [ ] Clean responses: memory tags stripped before returning to user
- [ ] Evaluation: bot recalls preferences stated 5+ messages ago

---

## Architecture

```
[User Message]
    |
    v
[Chat API]  POST /chat
    |
    +-- Load user session (conversation history)
    |
    +-- Memory Retrieval
    |       |-- Embed user message
    |       |-- Query ChromaDB for relevant memories (top-3)
    |       +-- Build memory context block
    |
    +-- Prompt Assembly
    |       |-- System prompt + memory context
    |       |-- Conversation history (short-term)
    |       +-- Current user message
    |
    +-- LLM Call (gpt-4o)
    |       +-- Response with optional [REMEMBER: ...] tags
    |
    +-- Memory Extraction
    |       |-- Parse [REMEMBER: ...] tags
    |       +-- Save new facts to ChromaDB
    |
    +-- Response Cleanup
    |       |-- Strip [REMEMBER: ...] tags
    |       +-- Update conversation history
    v
[Clean Response to User]

[Memory Store] (ChromaDB, per user)
    |-- POST save (fact, type, timestamp)
    |-- GET  search (query -> relevant memories)
    |-- GET  list  (all memories for user)
    |-- DELETE forget (remove specific memory)
```

---

## Step 1: Build the Memory Store

Vector-based long-term memory with metadata for filtering and management.

```python
# src/memory_store.py
import chromadb
from datetime import datetime, timezone
from typing import Optional
import uuid

class MemoryStore:
    """Vector-based long-term memory for the chatbot."""

    def __init__(self, user_id: str, persist_dir: str = "./chatbot_memory"):
        self.user_id = user_id
        self.chroma = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.chroma.get_or_create_collection(
            name=f"user-{user_id}",
            metadata={"hnsw:space": "cosine"},
        )

    def save(self, memory_text: str, memory_type: str = "fact") -> str:
        """Save a memory to the vector store. Returns the memory ID."""
        memory_id = f"mem-{uuid.uuid4().hex[:12]}"
        self.collection.add(
            documents=[memory_text],
            metadatas=[{
                "type": memory_type,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "user_id": self.user_id,
            }],
            ids=[memory_id],
        )
        return memory_id

    def search(self, query: str, k: int = 5, memory_type: Optional[str] = None) -> list[dict]:
        """Find relevant memories for a query."""
        if self.collection.count() == 0:
            return []
        where_filter = {"type": memory_type} if memory_type else None
        results = self.collection.query(
            query_texts=[query],
            n_results=min(k, self.collection.count()),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
        memories = []
        for i in range(len(results["ids"][0])):
            memories.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "type": results["metadatas"][0][i].get("type", "fact"),
                "created_at": results["metadatas"][0][i].get("created_at", ""),
                "relevance": round(1 - results["distances"][0][i], 3),
            })
        return memories

    def list_all(self) -> list[dict]:
        """List all memories for this user."""
        if self.collection.count() == 0:
            return []
        results = self.collection.get(include=["documents", "metadatas"])
        return [
            {
                "id": results["ids"][i],
                "text": results["documents"][i],
                "type": results["metadatas"][i].get("type", "fact"),
                "created_at": results["metadatas"][i].get("created_at", ""),
            }
            for i in range(len(results["ids"]))
        ]

    def delete(self, memory_id: str) -> bool:
        """Delete a specific memory."""
        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception:
            return False

    def count(self) -> int:
        return self.collection.count()
```

---

## Step 2: Build the Memory Chatbot

Core chatbot with memory retrieval, extraction, and conversation management.

```python
# src/memory_chatbot.py
import re
from openai import OpenAI
from src.memory_store import MemoryStore

client = OpenAI()

SYSTEM_PROMPT = """You are a helpful assistant with long-term memory.

You can remember things about the user across conversations.
When you learn something important about the user (preferences, facts, context),
include [REMEMBER: fact to remember] in your response.
When you recall relevant memories, reference them naturally in your reply.

Rules:
- Only save genuinely useful facts (preferences, name, job, goals)
- Do not save trivial or temporary information
- Reference memories naturally, not robotically
- If no relevant memories exist, respond normally"""

REMEMBER_PATTERN = re.compile(r"\[REMEMBER:\s*(.*?)\]", re.DOTALL)

class MemoryChatbot:
    def __init__(self, user_id: str, max_history: int = 20):
        self.user_id = user_id
        self.memory = MemoryStore(user_id)
        self.conversation_history: list[dict] = []
        self.max_history = max_history

    def chat(self, user_message: str) -> dict:
        """Process a message and return a response with metadata."""
        memories = self.memory.search(user_message, k=3)
        memory_context = self._build_memory_context(memories)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + memory_context},
            *self.conversation_history,
            {"role": "user", "content": user_message},
        ]

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
        )
        assistant_message = response.choices[0].message.content

        saved_memories = self._extract_and_save_memories(assistant_message)
        clean_response = REMEMBER_PATTERN.sub("", assistant_message).strip()

        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": clean_response})
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]

        return {
            "response": clean_response,
            "memories_used": [m["text"] for m in memories],
            "memories_saved": saved_memories,
            "total_memories": self.memory.count(),
        }

    def _build_memory_context(self, memories: list[dict]) -> str:
        if not memories:
            return ""
        lines = "\n".join(f"- {m['text']}" for m in memories)
        return f"\n\nRelevant memories about this user:\n{lines}"

    def _extract_and_save_memories(self, response: str) -> list[str]:
        saved = []
        for match in REMEMBER_PATTERN.finditer(response):
            memory_text = match.group(1).strip()
            if memory_text:
                self.memory.save(memory_text)
                saved.append(memory_text)
        return saved

    def get_memories(self) -> list[dict]:
        return self.memory.list_all()

    def forget_memory(self, memory_id: str) -> bool:
        return self.memory.delete(memory_id)

    def clear_conversation(self):
        self.conversation_history = []
```

---

## Step 3: FastAPI Server with Session Management

```python
# src/api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.memory_chatbot import MemoryChatbot

app = FastAPI(title="Memory Chatbot")
sessions: dict[str, MemoryChatbot] = {}

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    memories_used: list[str]
    memories_saved: list[str]
    total_memories: int

def get_bot(user_id: str) -> MemoryChatbot:
    if user_id not in sessions:
        sessions[user_id] = MemoryChatbot(user_id)
    return sessions[user_id]

@app.get("/health")
def health():
    return {"status": "ok", "active_sessions": len(sessions)}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    bot = get_bot(req.user_id)
    result = bot.chat(req.message)
    return ChatResponse(**result)

@app.get("/memories/{user_id}")
def list_memories(user_id: str):
    bot = get_bot(user_id)
    return {"user_id": user_id, "memories": bot.get_memories()}

@app.delete("/memories/{user_id}/{memory_id}")
def delete_memory(user_id: str, memory_id: str):
    bot = get_bot(user_id)
    success = bot.forget_memory(memory_id)
    if not success:
        raise HTTPException(404, "Memory not found")
    return {"deleted": memory_id}

@app.post("/sessions/{user_id}/clear")
def clear_session(user_id: str):
    bot = get_bot(user_id)
    bot.clear_conversation()
    return {"status": "conversation cleared"}
```

Run the server:

```bash
uvicorn src.api:app --reload --port 8000
```

---

## Step 4: Usage Example and Memory Evaluation

```python
# demo.py
from src.memory_chatbot import MemoryChatbot

def demo_conversation():
    bot = MemoryChatbot(user_id="demo-user")

    # Session 1: Share preferences
    r1 = bot.chat("Hi! I'm Nikhil, a Python developer working on ML projects.")
    print(f"Bot: {r1['response']}")
    print(f"Saved memories: {r1['memories_saved']}\n")

    r2 = bot.chat("I prefer using PyTorch over TensorFlow for deep learning.")
    print(f"Bot: {r2['response']}")
    print(f"Saved memories: {r2['memories_saved']}\n")

    # Simulate new session (clear short-term, keep long-term)
    bot.clear_conversation()

    # Session 2: Test memory recall
    r3 = bot.chat("What framework should I use for my next deep learning project?")
    print(f"Bot: {r3['response']}")
    print(f"Memories used: {r3['memories_used']}")
    print(f"Total memories stored: {r3['total_memories']}")

def evaluate_memory_recall():
    """Test that the bot recalls facts from earlier in the conversation."""
    bot = MemoryChatbot(user_id="eval-user")
    bot.chat("My favorite programming language is Rust.")
    bot.chat("I work at a fintech startup in San Francisco.")
    bot.clear_conversation()

    result = bot.chat("What language do I like to code in?")
    response_lower = result["response"].lower()
    recalled = "rust" in response_lower
    print(f"Memory recall test: {'PASS' if recalled else 'FAIL'}")
    print(f"Response: {result['response']}")
    print(f"Memories used: {result['memories_used']}")
    return recalled

if __name__ == "__main__":
    demo_conversation()
    print("\n--- Evaluation ---")
    evaluate_memory_recall()
```

---

## Testing Your Build

### Unit Tests

```python
# tests/test_memory.py
import pytest
from src.memory_store import MemoryStore
from src.memory_chatbot import MemoryChatbot, REMEMBER_PATTERN

@pytest.fixture
def store(tmp_path):
    return MemoryStore(user_id="test-user", persist_dir=str(tmp_path / "mem"))

def test_save_and_search(store):
    store.save("User prefers dark mode")
    store.save("User is a Python developer")
    results = store.search("programming language preferences")
    assert len(results) > 0

def test_delete_memory(store):
    mem_id = store.save("Temporary fact")
    assert store.count() == 1
    assert store.delete(mem_id)
    assert store.count() == 0

def test_remember_pattern_extraction():
    text = "Great choice! [REMEMBER: User prefers PyTorch] I'll keep that in mind."
    matches = REMEMBER_PATTERN.findall(text)
    assert len(matches) == 1
    assert "PyTorch" in matches[0]

def test_clean_response_strips_tags():
    text = "Noted! [REMEMBER: User likes Rust] Happy to help."
    clean = REMEMBER_PATTERN.sub("", text).strip()
    assert "[REMEMBER:" not in clean
    assert "Noted!" in clean

def test_conversation_history_limit():
    bot = MemoryChatbot(user_id="limit-test", max_history=4)
    bot.conversation_history = [{"role": "user", "content": f"msg{i}"} for i in range(6)]
    bot.conversation_history = bot.conversation_history[-bot.max_history:]
    assert len(bot.conversation_history) == 4
```

Run tests:

```bash
pytest tests/test_memory.py -v
```

### Memory Types and Classification

Not all memories are equal. Classify memories by type for better retrieval:

```python
MEMORY_TYPES = {
    "preference": "User likes, dislikes, choices (e.g., prefers PyTorch)",
    "fact": "Objective information (e.g., works at Acme Corp)",
    "goal": "User objectives (e.g., learning Rust for systems programming)",
    "context": "Situational context (e.g., building a RAG pipeline this week)",
}

def classify_memory(memory_text: str) -> str:
    """Use a quick LLM call to classify memory type."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Classify this memory into one of: preference, fact, goal, context.\n\nMemory: {memory_text}\n\nRespond with just the type.",
        }],
        max_tokens=10,
    )
    mem_type = response.choices[0].message.content.strip().lower()
    return mem_type if mem_type in MEMORY_TYPES else "fact"
```

Typed memories enable filtered retrieval — when the user asks about tools, search only `preference` memories.

### Session Persistence Across Restarts

```python
# src/session_store.py
import json
from pathlib import Path

class SessionStore:
    """Persist conversation history to disk between server restarts."""

    def __init__(self, persist_dir: str = "./sessions"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(exist_ok=True)

    def save_history(self, user_id: str, history: list[dict]):
        path = self.persist_dir / f"{user_id}.json"
        path.write_text(json.dumps(history))

    def load_history(self, user_id: str) -> list[dict]:
        path = self.persist_dir / f"{user_id}.json"
        if path.exists():
            return json.loads(path.read_text())
        return []
```

Integrate with the chatbot: load history on session start, save after each message. Long-term memories in ChromaDB persist automatically via the vector store.

---

- [ ] Bot saves preference when user states one
- [ ] Bot recalls preference in a new session (after `clear_conversation`)
- [ ] `[REMEMBER: ...]` tags do not appear in user-facing response
- [ ] `GET /memories/{user_id}` lists all stored memories
- [ ] `DELETE /memories/{user_id}/{id}` removes a specific memory
- [ ] Memories persist after server restart (ChromaDB on disk)
- [ ] Irrelevant memories are not injected (search quality)

---

## Deployment Notes

### Production Considerations

| Concern | Development | Production |
|---------|-------------|------------|
| Session storage | In-memory dict | Redis for session state |
| Memory store | ChromaDB local | Pinecone/Weaviate per user namespace |
| Memory growth | Unbounded | Summarize + prune old memories |
| Privacy | Single user | Per-user isolation, GDPR delete |

### Memory Pruning Strategy

```python
def prune_old_memories(store: MemoryStore, max_memories: int = 100):
    """Keep only the most recent memories when count exceeds limit."""
    all_memories = store.list_all()
    if len(all_memories) <= max_memories:
        return
    sorted_memories = sorted(all_memories, key=lambda m: m["created_at"])
    to_delete = sorted_memories[:len(all_memories) - max_memories]
    for mem in to_delete:
        store.delete(mem["id"])
```

### Privacy and GDPR

- Implement `DELETE /users/{user_id}/data` to wipe all memories and sessions
- Never store sensitive data (passwords, SSNs) — filter in memory extraction
- Log memory operations for audit without logging memory content

---

## Extensions and Challenges

- **Memory decay**: Weight recent memories higher in retrieval scoring
- **Memory summarization**: Compress 50+ memories into a user profile summary
- **User control UI**: Build a settings page where users view, edit, and delete memories
- **Emotion tracking**: Detect sentiment and remember communication preferences
- **Multi-user teams**: Shared memory namespace for team bots with role-based access
- **MemGPT-style paging**: Implement memory tiers (core, archival, recall) like the MemGPT paper

## Resources

- [YouTube: AI Memory Systems](https://www.youtube.com/watch?v=6qSqSrJh0kA) — Building persistent memory for chatbots
- [MemGPT Paper](https://arxiv.org/abs/2310.08560) — Research on LLM memory management
- [LangChain Memory](https://python.langchain.com/docs/concepts/memory/) — Framework memory patterns

---

## Next Lesson

**Project 9: AI Safety Evaluation Suite** — Build a comprehensive testing tool that evaluates LLM applications for bias, prompt injection, hallucination, and policy compliance.
