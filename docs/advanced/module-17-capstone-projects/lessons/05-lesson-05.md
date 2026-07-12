---
title: 'Project 4: AI-Powered Content Platform'
description: >-
  Build a full-stack content platform with AI-assisted writing, SEO
  optimization, tone adjustment, and a content management dashboard
duration: 120 min
difficulty: advanced
has_code: true
module: module-17
youtube: 'https://www.youtube.com/watch?v=K1SFgs6sFQo'
objectives:
  - Build an AI content generation pipeline with tone and style controls
  - Implement SEO optimization suggestions using structured output
  - Create a content management API with CRUD operations
  - Design a dashboard UI for content creation and editing
---
# Project 4: AI-Powered Content Platform

## Project Overview

Build a content platform that uses AI to help users create, edit, and optimize written content:
- Generate blog posts, marketing copy, and documentation from outlines
- Adjust tone and style (professional, casual, technical)
- Optimize for SEO with keyword suggestions and meta descriptions
- Manage content with a simple dashboard

**Time estimate**: 12-18 hours
**Skills used**: Prompt Engineering, Structured Output, Full-Stack Development, RAG

---

## Architecture

```
[Dashboard UI]
    |
    v
[FastAPI Backend]
    |-- POST /generate  (outline -> draft)
    |-- POST /rewrite   (text + instructions -> improved text)
    |-- POST /seo       (text -> SEO analysis + suggestions)
    |-- CRUD /articles  (manage saved content)
    v
[AI Pipeline]
    |-- Content Generator (GPT-4.1-mini)
    |-- Tone Adjuster (system prompt variants)
    |-- SEO Analyzer (structured output)
    |-- Style Reference RAG (optional: learn from examples)
    v
[Database] (SQLite for simplicity)
```

---

## Step 1: Content Generation Pipeline

```python
from openai import OpenAI
import json

client = OpenAI()

TONE_PROMPTS = {
    "professional": "Write in a professional, authoritative tone. Use industry terminology. Be concise and data-driven.",
    "casual": "Write in a friendly, conversational tone. Use simple language. Include analogies and relatable examples.",
    "technical": "Write for a technical audience. Include code examples, specifications, and detailed explanations.",
    "marketing": "Write persuasive copy. Focus on benefits over features. Use power words and clear calls to action.",
}

def generate_content(
    topic: str,
    outline: list[str],
    tone: str = "professional",
    target_length: int = 800,
) -> dict:
    """Generate a content piece from a topic and outline."""
    tone_instruction = TONE_PROMPTS.get(tone, TONE_PROMPTS["professional"])
    
    outline_text = "
".join(f"- {item}" for item in outline)
    
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": f"""You are an expert content writer.
{tone_instruction}

Write well-structured content with:
- An engaging introduction that hooks the reader
- Clear section headings (use ## for sections)
- Substantive paragraphs with specific details
- A conclusion with key takeaways
- Target approximately {target_length} words"""
            },
            {
                "role": "user",
                "content": f"Topic: {topic}

Outline:
{outline_text}

Write the full article."
            }
        ],
        temperature=0.7
    )
    
    content = response.choices[0].message.content
    word_count = len(content.split())
    
    return {
        "content": content,
        "word_count": word_count,
        "tone": tone,
        "topic": topic,
    }
```

---

## Step 2: SEO Analyzer

```python
def analyze_seo(content: str, target_keyword: str) -> dict:
    """Analyze content for SEO and provide improvement suggestions."""
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": """You are an SEO expert. Analyze the given content and provide actionable recommendations.
Respond as JSON with these fields:
- title_suggestions: array of 3 SEO-optimized title options
- meta_description: a 155-character meta description
- keyword_density: approximate percentage of target keyword
- missing_keywords: related keywords that should be added
- readability_score: "easy" / "moderate" / "difficult"
- improvements: array of specific improvement suggestions
- overall_score: integer 1-100"""
            },
            {
                "role": "user",
                "content": f"Target keyword: {target_keyword}

Content:
{content}"
            }
        ],
        response_format={"type": "json_object"},
        temperature=0.2
    )
    
    return json.loads(response.choices[0].message.content)
```

---

## Step 3: Content Rewriter

```python
def rewrite_content(
    content: str,
    instructions: str,
    preserve_structure: bool = True
) -> str:
    """Rewrite content based on specific instructions."""
    structure_note = (
        "Preserve the existing heading structure and section organization."
        if preserve_structure else
        "Feel free to restructure if it improves the content."
    )
    
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": f"""You are an expert editor. Rewrite the given content 
following the user's instructions. {structure_note}
Return only the rewritten content, no commentary."""
            },
            {
                "role": "user",
                "content": f"Instructions: {instructions}

Content to rewrite:
{content}"
            }
        ],
        temperature=0.5
    )
    
    return response.choices[0].message.content
```

---

## Step 4: FastAPI Backend

```python
from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3
import uuid
from datetime import datetime

app = FastAPI(title="AI Content Platform")

# Simple SQLite storage
def get_db():
    conn = sqlite3.connect("content.db")
    conn.execute("""CREATE TABLE IF NOT EXISTS articles (
        id TEXT PRIMARY KEY,
        title TEXT,
        content TEXT,
        tone TEXT,
        seo_score INTEGER,
        created_at TEXT,
        updated_at TEXT
    )""")
    return conn

class GenerateRequest(BaseModel):
    topic: str
    outline: list[str]
    tone: str = "professional"
    target_length: int = 800

class RewriteRequest(BaseModel):
    content: str
    instructions: str

class SEORequest(BaseModel):
    content: str
    target_keyword: str

@app.post("/generate")
async def generate(req: GenerateRequest):
    result = generate_content(req.topic, req.outline, req.tone, req.target_length)
    return result

@app.post("/rewrite")
async def rewrite(req: RewriteRequest):
    result = rewrite_content(req.content, req.instructions)
    return {"content": result}

@app.post("/seo")
async def seo_analysis(req: SEORequest):
    result = analyze_seo(req.content, req.target_keyword)
    return result

@app.post("/articles")
async def save_article(title: str, content: str, tone: str = "professional"):
    db = get_db()
    article_id = str(uuid.uuid4())[:8]
    now = datetime.utcnow().isoformat()
    db.execute(
        "INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?, ?)",
        (article_id, title, content, tone, 0, now, now)
    )
    db.commit()
    return {"id": article_id, "message": "Article saved"}

@app.get("/articles")
async def list_articles():
    db = get_db()
    rows = db.execute("SELECT id, title, tone, seo_score, created_at FROM articles ORDER BY created_at DESC").fetchall()
    return [{"id": r[0], "title": r[1], "tone": r[2], "seo_score": r[3], "created_at": r[4]} for r in rows]
```

---

## Evaluation

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Content quality | 4+ / 5 rating | Human evaluation of 10 generated articles |
| SEO score | 70+ / 100 | Run SEO analyzer on generated content |
| Tone accuracy | 90%+ | Judge rates tone match for 20 samples |
| Rewrite quality | Improved readability | Compare before/after readability scores |
| API latency | < 10s for generation | Time each endpoint |

---

## Extension Ideas

- Add RAG over a style guide to ensure brand consistency
- Implement A/B title testing with click prediction
- Add image suggestion capabilities
- Build a Markdown preview in the dashboard
- Add content scheduling and publishing integration

---

## Resources

- **OpenAI Structured Outputs**: JSON mode for reliable parsing
- **Yoast SEO**: Reference for SEO scoring criteria
- **Hemingway Editor**: Reference for readability analysis
- **Next.js**: For building the dashboard frontend

---

## Next Project

**Project 5: Conversational AI with Memory** — Build a personal AI assistant with persistent memory that learns user preferences over time.
