---
title: Your First AI Application
description: Set up your environment and build your first AI-powered chatbot
duration: 30 min
difficulty: beginner
has_code: true
module: module-01
youtube: 'https://www.youtube.com/watch?v=T9aRN5JkmL8'
objectives:
  - Successfully call OpenAI API
  - Build a working chatbot
  - Understand API structure
---
# Your First AI Application

![First AI App](https://images.unsplash.com/photo-1516116216624-53e697fedbea?w=800)

## What We'll Build

A simple but powerful chatbot that can:
- Answer questions
- Remember conversation history
- Stream responses in real-time

## Prerequisites

### 1. Get an OpenAI API Key

1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up or log in
3. Navigate to API Keys section
4. Create new key
5. **IMPORTANT**: Copy it now (you won't see it again!)

### 2. Install Required Packages

```bash
pip install openai python-dotenv
```

### 3. Create `.env` file

```
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
```

## Understanding the OpenAI API

The OpenAI API follows a simple request-response pattern:

```
YOU → [API Request] → OPENAI → [AI Response] → YOU
```

### Key Concepts

#### 1. Messages
Conversations are lists of messages with roles:
- **system**: Instructions for the AI (personality, behavior)
- **user**: The human's message
- **assistant**: The AI's response

#### 2. Models
Different models have different capabilities:
- `gpt-4o`: Most capable, latest model
- `gpt-4o-mini`: Fast and cheap, great for most tasks
- `gpt-3.5-turbo`: Older, cheaper

#### 3. Temperature
Controls randomness (0.0 - 2.0):
- **0.0**: Deterministic, same output every time
- **0.7**: Balanced (default)
- **1.5+**: Creative, varied

## Code Exercise: Build Your First Chatbot

Complete the code below to create a working chatbot!

---

## 📹 Recommended Videos

- [Build Your First AI App with OpenAI API](https://www.youtube.com/watch?v=T9aRN5JkmL8) — Step-by-step tutorial
- [OpenAI API Crash Course](https://www.youtube.com/watch?v=qYSWDk4-NHI) — Comprehensive API walkthrough
- [Python + ChatGPT API Tutorial](https://www.youtube.com/watch?v=c-g6epk3fFE) — Building chatbots with Python

---

## 📚 Additional Resources

### Articles & Blogs:
- [OpenAI API Quickstart](https://platform.openai.com/docs/quickstart) — Official getting started guide
- [Building LLM Applications](https://www.pinecone.io/learn/series/langchain/) — Pinecone's practical guide
- [Best Practices for API Key Safety](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety) — OpenAI security guide

### Tools:
- [OpenAI Playground](https://platform.openai.com/playground) — Test prompts interactively
- [python-dotenv](https://pypi.org/project/python-dotenv/) — Environment variable management
