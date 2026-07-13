---
title: Prompt Chaining and Pipelines
description: >-
  Learn to decompose complex tasks into multi-step prompt chains where each step
  builds on the previous output
duration: 40 min
difficulty: intermediate
has_code: true
module: module-14
---
# Prompt Chaining and Pipelines

## Prerequisites

Before this lesson you should be comfortable with:

- **Prompt anatomy and system prompts** — Lessons 1 and 3
- **Structured output** — getting parseable JSON between chain steps (Lesson 4)
- **Prompt templates** — rendering dynamic prompts (Lesson 5)
- **Basic async Python** — helpful for parallel chains but not required

You do not need LangChain or any orchestration framework. This lesson builds chains from first principles.

---

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Explain why single-shot prompts fail on complex tasks | 8 min | Intermediate |
| Build sequential, parallel, and conditional prompt chains | 12 min | Intermediate |
| Pass structured output between chain steps | 10 min | Intermediate |
| Implement error handling, retries, and failure recovery in pipelines | 10 min | Intermediate |

---

## Intuition First: The Assembly Line

Asking one LLM call to "read this article, extract entities, classify sentiment, generate a summary, and create action items" is like asking one person to build an entire car in a single step. They might succeed on a toy model, but quality drops fast as complexity grows.

An assembly line works better: Station 1 extracts entities. Station 2 classifies sentiment using those entities as context. Station 3 summarizes. Station 4 generates action items from the summary.

Each station does one focused job. Each output becomes the next station's input. If Station 2 fails, you debug Station 2 — not the entire pipeline.

That is prompt chaining. It is the most reliable architecture for complex LLM workflows in production.

---

## What Is Prompt Chaining?

Instead of asking an LLM to do everything in one shot, break the task into focused steps:

```
Single prompt (unreliable for complex tasks):
  "Read this article, extract entities, classify sentiment,
   generate a summary, and create action items"

Prompt chain (reliable):
  Step 1: Extract entities → entities
  Step 2: Classify sentiment (using entities) → sentiment
  Step 3: Summarize (using entities + sentiment) → summary
  Step 4: Generate action items (using summary) → actions
```

Each step has a narrow prompt, a clear output format, and a single evaluation criterion.

---

## Basic Chain Implementation

```python
from openai import OpenAI
import json

client = OpenAI()

def llm_call(prompt: str, system: str = "", json_mode: bool = False) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs = {"model": "gpt-4o-mini", "messages": messages, "temperature": 0.2}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content

def analyze_document(document: str) -> dict:
    """Multi-step document analysis pipeline."""

    # Step 1: Extract key entities
    entities = llm_call(
        f"Extract all key entities (people, organizations, technologies, dates) "
        f"from this document. Return as a JSON list.\n\n{document}",
        system="You extract entities. Respond only in JSON.",
        json_mode=True,
    )

    # Step 2: Classify the document
    classification = llm_call(
        f"Classify this document into one category: technical, business, "
        f"legal, or general.\n\nEntities found: {entities}\n\nDocument:\n{document}",
        system="You classify documents. Respond with just the category name.",
    )

    # Step 3: Generate summary using context from previous steps
    summary = llm_call(
        f"Summarize this {classification.strip()} document in 3 sentences.\n"
        f"Key entities: {entities}\n\nDocument:\n{document}",
        system="You write concise summaries.",
    )

    # Step 4: Extract action items
    actions = llm_call(
        f"Based on this summary, list specific action items:\n\n{summary}",
        system="Extract actionable next steps. Return as a numbered list.",
    )

    return {
        "entities": json.loads(entities),
        "classification": classification.strip(),
        "summary": summary,
        "action_items": actions,
    }
```

Notice how each step's prompt is short and focused. Step 3 gets richer context (entities + classification) than it would from a single monolithic prompt.

---

## Chain Patterns

### Sequential Chain

Each step depends on the previous output:

```
Input → Step A → Step B → Step C → Output
```

Best for: tasks where later steps genuinely need earlier results (summarize after classify, not before).

### Parallel Chain

Independent subtasks run concurrently; results merge at the end:

```python
import asyncio
from openai import AsyncOpenAI

async_client = AsyncOpenAI()

async def async_llm_call(prompt: str, system: str = "") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = await async_client.chat.completions.create(
        model="gpt-4o-mini", messages=messages, temperature=0.2,
    )
    return response.choices[0].message.content

async def parallel_analysis(document: str) -> dict:
    summary_task = async_llm_call(f"Summarize:\n{document}")
    sentiment_task = async_llm_call(f"Classify sentiment:\n{document}")
    entities_task = async_llm_call(f"Extract entities:\n{document}")

    summary, sentiment, entities = await asyncio.gather(
        summary_task, sentiment_task, entities_task,
    )
    return {"summary": summary, "sentiment": sentiment, "entities": entities}
```

Parallel chains cut latency when steps are independent. Three 2-second calls in parallel take ~2 seconds, not ~6.

### Conditional Chain

Route to different prompts based on intermediate results:

```python
def conditional_analysis(text: str) -> str:
    category = llm_call(
        f"Is this text about code, data, or general topic?\n\n{text}",
        system="Respond with one word: code, data, or general.",
    )

    routes = {
        "code": f"Review this code for bugs and improvements:\n\n{text}",
        "data": f"Analyze this data for patterns and anomalies:\n\n{text}",
    }
    prompt = routes.get(category.strip().lower(), f"Summarize the key points:\n\n{text}")
    return llm_call(prompt)
```

Conditional chains let you use specialized prompts without running all specialists on every input.

---

## Error Handling in Chains

Production chains need retries, validation, and graceful degradation:

```python
import logging

logger = logging.getLogger(__name__)

def robust_chain_step(
    prompt: str,
    system: str = "",
    retries: int = 2,
    min_length: int = 5,
) -> str:
    """Execute a chain step with retry logic."""
    for attempt in range(retries + 1):
        try:
            result = llm_call(prompt, system)
            if not result or len(result.strip()) < min_length:
                raise ValueError(f"Response too short: {repr(result)}")
            return result
        except Exception as e:
            logger.warning(f"Chain step attempt {attempt + 1} failed: {e}")
            if attempt == retries:
                return f"[STEP_FAILED: {e}]"
    return "[UNEXPECTED_FAILURE]"

def analyze_with_fallback(document: str) -> dict:
    """Pipeline that continues even if individual steps fail."""
    entities = robust_chain_step(
        f"Extract entities as JSON list:\n\n{document}",
        system="Respond only in JSON.",
    )

    if entities.startswith("[STEP_FAILED"):
        entities = "[]"

    classification = robust_chain_step(
        f"Classify as technical/business/legal/general:\n\n{document}",
    )
    if classification.startswith("[STEP_FAILED"):
        classification = "general"

    summary = robust_chain_step(
        f"Summarize in 3 sentences ({classification}):\n\n{document}",
    )

    return {
        "entities": entities,
        "classification": classification,
        "summary": summary,
        "partial": any(
            v.startswith("[STEP_FAILED")
            for v in [entities, classification, summary]
        ),
    }
```

When a step fails, downstream steps still run with degraded context rather than the entire pipeline crashing.

---

## Production Connection

Prompt chains are how most production LLM features are built:

- **Version each step independently** — improve the entity extractor without touching the summarizer. Log step name + version per call.
- **A/B test individual steps** — swap Step 2's classifier prompt while holding Steps 1, 3, 4 constant. Isolate which step drives accuracy changes.
- **Eval loops per step** — measure entity extraction F1, classification accuracy, and summary quality separately. A bad summary might be a bad summary prompt, not bad entities.
- **Failure recovery** — define fallback values per step (empty entity list, "general" classification). Mark outputs as `partial=True` so downstream systems know to handle degraded results.
- **Latency budgets** — sequential chains add latency linearly. Use parallel chains for independent steps. Set timeouts per step, not just per pipeline.
- **Cost tracking** — log token usage per step. Optimizing the most expensive step first gives the best ROI.

---

## Edge Cases & Common Misconceptions

**Misconception 1: More steps always means better quality.**
Unnecessary steps add latency, cost, and failure points. Only chain when steps genuinely benefit from prior context.

**Misconception 2: Chain output quality is limited by the weakest step.**
Yes — which is why per-step eval matters. Fix the bottleneck step, not the whole pipeline.

**Misconception 3: You need a framework for chains.**
Plain Python functions with clear inputs and outputs work well up to moderate complexity. Reach for LangChain or similar when you need observability, caching, or complex branching.

**Misconception 4: Parallel chains always save money.**
They save latency, not tokens. Three parallel calls still consume three calls' worth of tokens.

---

## Key Takeaways

- Decompose complex tasks into focused single-purpose prompt steps rather than one monolithic prompt.
- Sequential chains pass context forward; parallel chains run independent steps concurrently; conditional chains route by classification.
- Use structured output (JSON) between steps so downstream prompts receive clean, parseable input.
- Implement per-step retries, validation, and fallback values so one failure doesn't crash the pipeline.
- Version and eval each step independently — the bottleneck step determines overall pipeline quality.
- A/B test individual steps to isolate which prompt changes improve end-to-end results.
- Track latency and token cost per step; optimize the most expensive step first.
- Mark partial results when steps fail so downstream systems can handle degraded output gracefully.

---

## Next Lesson

**[Lesson 7: Prompt Optimization and Iteration](07-lesson-07.md)** — Learn systematic methods to improve prompt performance through evaluation, A/B testing, and iterative refinement.
