---
title: 'Project 8: Chatbot with Long-Term Memory'
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

Build a chatbot that maintains long-term memory across conversations. It remembers user preferences, past interactions, and important facts. This applies agent memory, vector databases, and conversation design skills.

**What you will build:**
- Short-term memory (current conversation context)
- Long-term memory (persisted to vector database)
- Memory retrieval during conversations
- Memory management (save, search, forget)

**Estimated time:** 4-6 hours

---

## Implementation

```python
# memory_chatbot.py
import chromadb
import json
from datetime import datetime
from openai import OpenAI

client = OpenAI()

class MemoryStore:
    """Vector-based long-term memory for the chatbot."""

    def __init__(self, user_id: str):
        self.chroma = chromadb.PersistentClient(path="./chatbot_memory")
        self.collection = self.chroma.get_or_create_collection(f"user-{user_id}")

    def save(self, memory_text: str, memory_type: str = "fact"):
        """Save a memory to the vector store."""
        memory_id = f"mem-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        self.collection.add(
            documents=[memory_text],
            metadatas=[{
                "type": memory_type,
                "created_at": datetime.utcnow().isoformat(),
            }],
            ids=[memory_id],
        )

    def search(self, query: str, k: int = 5) -> list[str]:
        """Find relevant memories for a query."""
        if self.collection.count() == 0:
            return []
        results = self.collection.query(
            query_texts=[query],
            n_results=min(k, self.collection.count()),
        )
        return results["documents"][0] if results["documents"] else []

class MemoryChatbot:
    def __init__(self, user_id: str):
        self.memory = MemoryStore(user_id)
        self.conversation_history = []
        self.system_prompt = """You are a helpful assistant with long-term memory.

You can remember things about the user across conversations.
When you learn something important about the user (preferences, facts, context), 
include [REMEMBER: fact to remember] in your response.
When you recall relevant memories, reference them naturally."""

    def chat(self, user_message: str) -> str:
        """Process a message and return a response."""
        # Retrieve relevant memories
        memories = self.memory.search(user_message, k=3)
        memory_context = ""
        if memories:
            memory_context = "

Relevant memories about this user:
" + "
".join(
                f"- {m}" for m in memories
            )

        # Build messages
        messages = [
            {"role": "system", "content": self.system_prompt + memory_context}
        ] + self.conversation_history + [
            {"role": "user", "content": user_message}
        ]

        # Get response
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
        )
        assistant_message = response.choices[0].message.content

        # Extract and save memories
        self._extract_memories(assistant_message)

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": assistant_message})

        # Keep conversation history manageable
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

        # Clean memory tags from response
        clean_response = assistant_message
        while "[REMEMBER:" in clean_response:
            start = clean_response.index("[REMEMBER:")
            end = clean_response.index("]", start) + 1
            memory_text = clean_response[start+10:end-1].strip()
            clean_response = clean_response[:start] + clean_response[end:]

        return clean_response.strip()

    def _extract_memories(self, response: str):
        """Extract [REMEMBER: ...] tags and save to memory."""
        while "[REMEMBER:" in response:
            start = response.index("[REMEMBER:")
            end = response.index("]", start)
            memory_text = response[start+10:end].strip()
            self.memory.save(memory_text)
            response = response[end+1:]

# Usage
bot = MemoryChatbot(user_id="user-123")
print(bot.chat("Hi! I'm a Python developer working on ML projects."))
print(bot.chat("I prefer using PyTorch over TensorFlow."))
# Later session...
print(bot.chat("What framework should I use for my next project?"))
# Bot remembers: user prefers PyTorch
```

---

## Extensions and Challenges

- **Memory decay**: Reduce relevance of old memories over time
- **Memory summarization**: Compress old memories into summaries
- **User control**: Let users view, edit, and delete their memories
- **Emotion tracking**: Detect and remember user sentiment patterns
- **Multi-user**: Support shared memories for team bots

## Resources

- [YouTube: AI Memory Systems](https://www.youtube.com/watch?v=6qSqSrJh0kA) -- Building persistent memory for chatbots
- [MemGPT Paper](https://arxiv.org/abs/2310.08560) -- Research on LLM memory management
- [LangChain Memory](https://python.langchain.com/docs/concepts/memory/) -- Framework memory patterns

---

Next: Project 9 -- AI Safety Evaluation Suite
