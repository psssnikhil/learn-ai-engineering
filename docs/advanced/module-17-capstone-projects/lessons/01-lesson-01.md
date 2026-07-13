---
title: 'Capstone Overview: Building Production AI Applications'
description: >-
  Overview of the capstone module — project selection, requirements,
  architecture patterns, evaluation rubric, and portfolio presentation
duration: 30 min
difficulty: advanced
has_code: false
module: module-17
---
# Capstone Overview: Building Production AI Applications

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand capstone project requirements and evaluation criteria | 30 min | Advanced |
| Choose the right project for your goals | | |
| Plan your architecture and development timeline | | |
| Prepare your project for portfolio presentation | | |

---

## Prerequisites

Before starting any capstone project, you should have completed (or be actively working through) these course modules:

| Module | Why It Matters for Capstones |
|--------|---------------------------|
| **Module 3: RAG Systems** | Required for Projects 1, 5, 6, 7 |
| **Module 5: AI Agents** | Required for Projects 2, 3, 8 |
| **Module 12: Multi-Agent Systems** | Required for Project 3 |
| **Module 13: LLMOps & Deployment** | Required for all production deployments |
| **Module 15: Fine-Tuning** | Recommended for Project 5 |
| **Module 16: AI Safety & Ethics** | Required for Projects 8, 9 |

**Technical prerequisites:**
- Python 3.10+ with virtual environment management
- Git and GitHub for version control
- OpenAI API key (or equivalent LLM provider)
- Basic familiarity with FastAPI or Flask
- Comfort reading API documentation and debugging errors

```bash
# Recommended starter environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install openai chromadb fastapi uvicorn pydantic python-dotenv pytest
```

---

## What You'll Build

This module is not a single project — it is a **portfolio of production AI applications**. By the end, you will have built at least two complete systems that demonstrate you can ship real AI products, not just notebooks.

### Module Completion Checklist

Use this checklist to confirm your capstone work is portfolio-ready:

- [ ] **At least 2 capstone projects** completed end-to-end
- [ ] **Working application** with API or web UI (not just scripts)
- [ ] **Architecture diagram** in README showing data flow
- [ ] **Evaluation metrics** with concrete numbers (accuracy, latency, cost)
- [ ] **Error handling** with retries, timeouts, and user-friendly messages
- [ ] **Tests** covering core functionality (unit + integration)
- [ ] **Documentation** with setup instructions anyone can follow
- [ ] **Demo artifact** — GIF, video, or live deployment link
- [ ] **Reflection section** describing technical challenges and solutions

### Project Catalog

| # | Project | Core Skills | Difficulty | Time |
|---|---------|-------------|------------|------|
| 1 | RAG Knowledge Assistant | RAG, Vector DB, Embeddings | Medium | 10-15h |
| 2 | Autonomous Coding Agent | Agents, Tool Use, ReAct | Hard | 15-20h |
| 3 | Multi-Agent Research System | Multi-Agent, MCP, Orchestration | Hard | 15-20h |
| 4 | AI-Powered Content Platform | RAG, Fine-Tuning, Full-Stack | Medium | 12-18h |
| 5 | Semantic Search Engine | Hybrid Search, Facets, Web UI | Medium | 10-15h |
| 6 | Data Extraction Pipeline | Structured Output, OCR, Validation | Medium | 12-16h |
| 7 | Memory Chatbot | Agent Memory, Vector DB, Sessions | Medium | 10-15h |
| 8 | AI Safety Evaluation Suite | Red Teaming, Judge Models, CI | Medium | 10-14h |
| 9 | Real-Time AI Dashboard | LLMOps, Monitoring, Deployment | Medium | 10-15h |
| 10 | Open-Ended Capstone | Your choice | Varies | 15-25h |

### Choosing Your Projects

Pick projects that tell a coherent story about your skills:

| Career Goal | Recommended Pair |
|-------------|-----------------|
| **AI Engineer (RAG focus)** | Project 1 + Project 5 |
| **Agent Developer** | Project 2 + Project 3 |
| **Full-Stack AI** | Project 4 + Project 1 |
| **AI Safety / Trust** | Project 8 + Project 7 |
| **Data / Document AI** | Project 6 + Project 1 |

---

## Architecture

Every capstone should follow a consistent production architecture. Adapt the layers to your project, but keep the separation of concerns:

```
                    [User Interface]
                    (Web UI / CLI / API Client)
                           |
                           v
                    [API Layer]
                    (FastAPI / Flask)
                    - Request validation
                    - Auth (optional)
                    - Rate limiting
                           |
                           v
                    [Application Layer]
                    - Business logic
                    - Prompt orchestration
                    - Agent loops
                           |
              +------------+------------+
              v            v            v
        [LLM Layer]  [Data Layer]  [Tools Layer]
        - OpenAI     - Vector DB   - File I/O
        - Embeddings - SQLite/PG   - Web search
        - Fine-tuned - S3/local   - Git/APIs
              |            |            |
              +------------+------------+
                           v
                    [Observability]
                    - Logging
                    - Metrics
                    - Cost tracking
```

### Recommended Repository Structure

```
project/
  README.md              # Setup, architecture, usage, metrics
  requirements.txt       # Pinned dependencies
  .env.example           # Environment variable template
  src/
    main.py              # Entry point
    config.py            # Configuration and env loading
    llm/                 # LLM interaction layer
    data/                # Data loading and processing
    api/                 # API endpoints
    agents/              # Agent logic (if applicable)
  tests/
    test_core.py         # Unit tests
    test_api.py          # Integration tests
  evaluation/
    eval_suite.py        # Evaluation scripts
    results/             # Saved evaluation outputs
  docs/
    architecture.md      # Detailed design notes
```

---

## Step 1: Set Up Your Project Repository

Start every capstone with a clean, documented repository:

```bash
mkdir rag-knowledge-assistant && cd rag-knowledge-assistant
git init
python -m venv .venv && source .venv/bin/activate
pip install openai chromadb fastapi uvicorn pydantic python-dotenv pytest httpx
pip freeze > requirements.txt
```

Create a `.env.example` file so others can configure the project:

```bash
# .env.example
OPENAI_API_KEY=sk-your-key-here
CHROMA_PERSIST_DIR=./chroma_db
LOG_LEVEL=INFO
```

Add a minimal `src/config.py`:

```python
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is required. Copy .env.example to .env and fill in your key.")
```

---

## Step 2: Design Before You Code

Spend your first session on design, not implementation. A 30-minute design doc saves hours of refactoring.

### Design Document Template

```markdown
# Project: [Name]

## Problem Statement
What problem does this solve? Who is the user?

## Success Criteria
- [ ] Criterion 1 (measurable)
- [ ] Criterion 2 (measurable)

## Architecture
[ASCII diagram]

## Data Flow
1. User does X
2. System does Y
3. Output is Z

## Tech Stack
- LLM: gpt-4.1-mini
- Vector DB: ChromaDB
- API: FastAPI
- Deployment: Railway

## Risks
- Risk 1: Hallucination → Mitigation: citations + confidence scoring
- Risk 2: High latency → Mitigation: caching + smaller model for retrieval

## Timeline
- Week 1: Prototype core pipeline
- Week 2: API + error handling + tests
- Week 3: Evaluation + deployment + docs
```

---

## Step 3: Build in Three Phases

### Phase 1: Minimal Working Prototype (Days 1-3)

Get the core AI pipeline working in a single script before adding API, UI, or polish:

```python
# prototype.py — prove the core idea works
from openai import OpenAI

client = OpenAI()

def ask(question: str, context: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "Answer based only on the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.1,
    )
    return response.choices[0].message.content

# Test with a hardcoded example
answer = ask("What is RAG?", context="RAG combines retrieval with generation...")
print(answer)
```

### Phase 2: Production Hardening (Days 4-10)

Add the layers that separate a demo from a product:

```python
# src/llm/client.py — production LLM wrapper with retries
import time
from openai import OpenAI, RateLimitError, APIError

client = OpenAI()

def chat_with_retry(messages: list, model: str = "gpt-4.1-mini", max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
            )
            return response.choices[0].message.content
        except RateLimitError:
            wait = 2 ** attempt
            time.sleep(wait)
        except APIError as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(1)
    raise RuntimeError("Max retries exceeded")
```

### Phase 3: Evaluation and Polish (Days 11-15)

Run quantitative evaluation and document results:

```python
# evaluation/eval_suite.py
import time
import json

TEST_CASES = [
    {"question": "What is hybrid search?", "expected_keywords": ["bm25", "vector", "fusion"]},
    {"question": "How do embeddings work?", "expected_keywords": ["vector", "similarity"]},
]

def run_evaluation(pipeline_fn) -> dict:
    results = {"passed": 0, "failed": 0, "latencies": [], "details": []}
    for case in TEST_CASES:
        start = time.time()
        answer = pipeline_fn(case["question"])
        latency = time.time() - start
        results["latencies"].append(latency)
        passed = any(kw.lower() in answer.lower() for kw in case["expected_keywords"])
        results["passed" if passed else "failed"] += 1
        results["details"].append({"question": case["question"], "passed": passed, "latency": latency})
    results["accuracy"] = results["passed"] / len(TEST_CASES)
    results["avg_latency"] = sum(results["latencies"]) / len(results["latencies"])
    return results
```

---

## Evaluation Rubric

Your capstone will be evaluated against this rubric. Use it as a self-assessment checklist during development:

| Category | Weight | Excellent (90%+) | Acceptable (70%) | Needs Work (<70%) |
|----------|--------|------------------|-------------------|-------------------|
| **Functionality** | 30% | Solves the stated problem reliably | Works on happy path | Breaks on common inputs |
| **Architecture** | 20% | Clean separation, extensible | Organized but coupled | Single monolithic script |
| **AI Engineering** | 25% | Proper RAG/agents/safety/eval | Basic LLM calls work | No AI-specific patterns |
| **Documentation** | 15% | README + diagram + setup in <5 min | README exists | No setup instructions |
| **Production Readiness** | 10% | Error handling, logging, config | Some error handling | Crashes on errors |

---

## Testing Your Build

Before calling a capstone complete, run through this testing protocol:

### Functional Testing

```bash
# Run your test suite
pytest tests/ -v

# Manual smoke test checklist
# 1. Start the server: uvicorn src.main:app --reload
# 2. Hit the health endpoint: curl http://localhost:8000/health
# 3. Run the primary user flow end-to-end
# 4. Test with empty input, very long input, and malformed input
# 5. Test with API key missing (should fail gracefully)
```

### AI-Specific Testing

| Test Type | What to Check | How |
|-----------|--------------|-----|
| **Retrieval quality** | Right chunks returned | 10-20 labeled Q&A pairs |
| **Answer accuracy** | Correct, cited answers | Human review + keyword match |
| **Latency** | <3s for typical queries | `time.time()` around pipeline |
| **Cost** | <$0.05 per query | Log token usage per request |
| **Safety** | Refuses harmful inputs | Red team test suite (Project 8) |
| **Edge cases** | Empty docs, long docs, non-English | Adversarial test inputs |

---

## Deployment Notes

Every capstone should be deployable. You do not need to deploy all projects, but at least one should have a live demo link in your portfolio.

### Quick Deployment Options

| Platform | Best For | Free Tier | Deploy Time |
|----------|----------|-----------|-------------|
| **Railway** | FastAPI + ChromaDB | $5 credit/month | 10 min |
| **Fly.io** | Dockerized apps | Limited free | 15 min |
| **Render** | Simple web services | Free tier available | 10 min |
| **Hugging Face Spaces** | Gradio/Streamlit demos | Free | 5 min |

### Deployment Checklist

- [ ] Environment variables configured on platform (not hardcoded)
- [ ] `requirements.txt` pinned with versions
- [ ] Health check endpoint at `/health`
- [ ] CORS configured if frontend is separate
- [ ] Persistent storage for vector DB (volume mount or cloud DB)
- [ ] README updated with live demo URL

### Example Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Portfolio Presentation

Your capstone exists to get you hired or promoted. Structure your README for a recruiter scanning it in 30 seconds:

```markdown
# RAG Knowledge Assistant

> Ask questions about your documents and get cited answers in seconds.

[![Demo](demo.gif)](https://your-demo-url.com)
**Live demo:** https://your-demo-url.com

## Results
- 92% answer accuracy on 50-question eval set
- 1.2s average latency (p95: 2.8s)
- $0.003 average cost per query

## Tech Stack
FastAPI · ChromaDB · OpenAI Embeddings · GPT-4.1-mini

## What I Learned
Hybrid search improved recall by 18% over vector-only retrieval.
The hardest part was chunk boundary handling for tables in PDFs.
```

---

## Extensions and Challenges

Once you have completed two capstones, level up with these challenges:

- **Combine projects**: Build a RAG assistant (Project 1) with a safety eval suite (Project 8) running in CI
- **Add observability**: Instrument all projects with Langfuse or LangSmith tracing
- **Multi-tenant**: Support multiple users with isolated document collections
- **Cost optimization**: Replace GPT-4.1 with a fine-tuned smaller model for 80% of queries
- **Open-source release**: Publish one project on GitHub with a polished README and demo GIF

---

## Key Takeaways

- Complete at least 2 capstone projects for a strong portfolio
- Every project needs a working app, evaluation metrics, and documentation
- Plan your time: 1 week design, 1 week build, 1 week polish
- Think about portfolio presentation from the start
- Use the evaluation rubric as a self-assessment tool throughout development

---

## Next Lesson

**Project 1: RAG Knowledge Assistant** — Build a full-featured document Q&A system with hybrid search, citations, and a web UI.
