---
title: 'Project 10: Deploy Your AI Application'
description: >-
  Take any capstone project and deploy it to production with Docker, monitoring,
  cost controls, and a public API
duration: 240 min
difficulty: advanced
has_code: true
---
# Project 10: Deploy Your AI Application

## Project Overview

Take any of the previous capstone projects and deploy it as a production service. This final project applies LLMOps, deployment, monitoring, and cost optimization skills.

**What you will build:**
- FastAPI application wrapping your AI project
- Docker container for deployment
- Rate limiting and cost controls
- Health checks and monitoring
- API key authentication

**Estimated time:** 4-6 hours

---

## Step 1: FastAPI Application

```python
# api.py
from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
import time
import os

app = FastAPI(title="AI Application API", version="1.0.0")

# Simple API key auth
API_KEYS = set(os.environ.get("API_KEYS", "demo-key").split(","))

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key

class QueryRequest(BaseModel):
    question: str
    max_results: int = 5

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_ms: int

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, api_key: str = Depends(verify_api_key)):
    start = time.time()

    # Replace with your actual AI logic
    from query import answer_question
    result = answer_question(request.question, k=request.max_results)

    latency = int((time.time() - start) * 1000)

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        latency_ms=latency,
    )
```

---

## Step 2: Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```text
# requirements.txt
fastapi==0.111.0
uvicorn==0.30.0
openai==1.30.0
chromadb==0.5.0
pydantic==2.7.0
```

---

## Step 3: Rate Limiting

```python
# rate_limiter.py
import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests = defaultdict(list)

    def is_allowed(self, api_key: str) -> bool:
        now = time.time()
        # Clean old requests
        self.requests[api_key] = [
            t for t in self.requests[api_key]
            if now - t < self.window
        ]
        if len(self.requests[api_key]) >= self.max_requests:
            return False
        self.requests[api_key].append(now)
        return True
```

---

## Step 4: Monitoring

```python
# monitoring.py
import time
import json
import logging

logger = logging.getLogger("ai_monitor")

class RequestMonitor:
    def __init__(self):
        self.total_requests = 0
        self.total_errors = 0
        self.total_latency_ms = 0
        self.total_tokens = 0

    def log_request(self, endpoint: str, latency_ms: int,
                    tokens_used: int = 0, error: str = None):
        self.total_requests += 1
        self.total_latency_ms += latency_ms
        self.total_tokens += tokens_used
        if error:
            self.total_errors += 1

        logger.info(json.dumps({
            "endpoint": endpoint,
            "latency_ms": latency_ms,
            "tokens": tokens_used,
            "error": error,
            "timestamp": time.time(),
        }))

    def get_stats(self) -> dict:
        avg_latency = (self.total_latency_ms / self.total_requests
                       if self.total_requests > 0 else 0)
        return {
            "total_requests": self.total_requests,
            "error_rate": self.total_errors / max(self.total_requests, 1),
            "avg_latency_ms": round(avg_latency),
            "total_tokens": self.total_tokens,
        }
```

---

## Step 5: Deploy

```bash
# Build and run locally
docker build -t my-ai-app .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... -e API_KEYS=my-key my-ai-app

# Test the API
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: my-key" \
  -d '{"question": "What is RAG?"}'

# Deploy to cloud (example: Railway, Render, or AWS)
# Most platforms support Dockerfile-based deployments
```

---

## Deployment Checklist

- [ ] API key authentication enabled
- [ ] Rate limiting configured
- [ ] Health check endpoint working
- [ ] Error handling returns safe messages (no stack traces to users)
- [ ] Environment variables for all secrets (no hardcoded API keys)
- [ ] Request logging enabled
- [ ] Cost monitoring in place
- [ ] Graceful shutdown handling
- [ ] README with setup and API documentation

---

## Extensions and Challenges

- **Add caching**: Cache frequent queries with Redis
- **Load testing**: Use `locust` to test API under load
- **CI/CD pipeline**: Auto-deploy on git push with GitHub Actions
- **Custom domain**: Set up a custom domain with SSL
- **Usage dashboard**: Build a simple dashboard showing API usage stats

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/) -- API framework
- [YouTube: Deploying AI Apps](https://www.youtube.com/watch?v=iMZY7x2MSYY) -- End-to-end deployment walkthrough
- [Docker Documentation](https://docs.docker.com/get-started/) -- Container basics
- [Railway](https://railway.app/) -- Simple cloud deployment platform

---

## Module and Course Complete!

Congratulations! You have completed all 17 modules of the AI Learning Platform.

**What you have built across these capstone projects:**
- RAG-powered knowledge assistant
- AI code review tool
- Autonomous research agent
- Multi-agent support system
- Content generation pipeline
- Semantic search engine
- Data extraction pipeline
- Chatbot with long-term memory
- AI safety evaluation suite
- Production deployment with monitoring

**You are now ready to:**
- Build production AI applications from scratch
- Work as an AI engineer
- Design and deploy autonomous AI agents
- Build scalable, safe LLM systems

Keep building and learning!
