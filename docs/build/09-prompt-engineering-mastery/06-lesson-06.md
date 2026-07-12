---
title: Prompt Chaining and Pipelines
description: >-
  Learn to decompose complex tasks into multi-step prompt chains where each step
  builds on the previous output
duration: 40 min
difficulty: intermediate
has_code: false
---
# Prompt Chaining and Pipelines

## Learning Objectives

By the end of this lesson, you will be able to:
- Decompose complex tasks into sequential prompt steps
- Build prompt chains where each step refines the previous output
- Implement parallel prompt execution for independent subtasks
- Handle errors and fallbacks in multi-step pipelines

---

## What is Prompt Chaining?

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

---

## Basic Chain Implementation

```python
from openai import OpenAI

client = OpenAI()

def llm_call(prompt: str, system: str = "") -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
    )
    return response.choices[0].message.content

def analyze_document(document: str) -> dict:
    """Multi-step document analysis pipeline."""

    # Step 1: Extract key entities
    entities = llm_call(
        f"Extract all key entities (people, organizations, technologies, dates) "
        f"from this document. Return as a JSON list.

{document}",
        system="You extract entities. Respond only in JSON."
    )

    # Step 2: Classify the document
    classification = llm_call(
        f"Classify this document into one category: technical, business, "
        f"legal, or general.

Entities found: {entities}

Document:
{document}",
        system="You classify documents. Respond with just the category."
    )

    # Step 3: Generate summary using context from previous steps
    summary = llm_call(
        f"Summarize this {classification} document in 3 sentences. "
        f"Key entities: {entities}

Document:
{document}",
        system="You write concise summaries."
    )

    # Step 4: Extract action items
    actions = llm_call(
        f"Based on this summary, list specific action items:

{summary}",
        system="You extract actionable next steps. Return as a numbered list."
    )

    return {
        "entities": entities,
        "classification": classification,
        "summary": summary,
        "action_items": actions,
    }
```

---

## Chain Patterns

### Sequential Chain
Each step depends on the previous output.

```
Input → Step A → Step B → Step C → Output
```

### Parallel Chain
Independent subtasks run concurrently, results merge at the end.

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
        model="gpt-4o", messages=messages
    )
    return response.choices[0].message.content

async def parallel_analysis(document: str) -> dict:
    # Run independent analyses concurrently
    summary_task = async_llm_call(f"Summarize:
{document}")
    sentiment_task = async_llm_call(f"Classify sentiment:
{document}")
    entities_task = async_llm_call(f"Extract entities:
{document}")

    summary, sentiment, entities = await asyncio.gather(
        summary_task, sentiment_task, entities_task
    )

    return {"summary": summary, "sentiment": sentiment, "entities": entities}
```

### Conditional Chain
Route to different prompts based on intermediate results.

```python
def conditional_analysis(text: str) -> str:
    # Step 1: Classify
    category = llm_call(f"Is this text about code, data, or general topic?

{text}")

    # Step 2: Route to specialized prompt
    if "code" in category.lower():
        return llm_call(f"Review this code for bugs and improvements:

{text}")
    elif "data" in category.lower():
        return llm_call(f"Analyze this data for patterns and anomalies:

{text}")
    else:
        return llm_call(f"Summarize the key points:

{text}")
```

---

## Error Handling in Chains

```python
def robust_chain_step(prompt: str, system: str = "", retries: int = 2) -> str:
    """Execute a chain step with retry logic."""
    for attempt in range(retries + 1):
        try:
            result = llm_call(prompt, system)
            if not result or len(result.strip()) < 5:
                raise ValueError("Response too short or empty")
            return result
        except Exception as e:
            if attempt == retries:
                return f"[Step failed after {retries + 1} attempts: {e}]"
    return "[Unexpected failure]"
```

---

## Key Takeaways

- Complex tasks are more reliable when decomposed into focused prompt steps
- Sequential chains pass output from one step to the next as context
- Parallel chains speed up independent subtasks by running them concurrently
- Conditional chains route to specialized prompts based on intermediate classification
- Always add error handling and retries to production prompt chains

## Resources

- [YouTube: Prompt Chaining Patterns](https://www.youtube.com/watch?v=9JJ2OGJYKiU) -- Building reliable multi-step LLM workflows
- [LangChain: Chains](https://python.langchain.com/docs/how_to/#few-shot-prompting) -- Framework for building prompt chains
- [Anthropic: Prompt Chaining](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/chain-prompts) -- Decomposition strategies for Claude

---

Next: Prompt Optimization and Iteration
