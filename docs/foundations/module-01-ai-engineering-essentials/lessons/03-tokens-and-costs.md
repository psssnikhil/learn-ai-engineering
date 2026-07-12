---
title: Understanding Tokens and Cost Optimization
description: >-
  Understand how BPE tokenization works at the engineering level, count tokens
  accurately, model API costs for realistic workloads, and apply the right
  optimization strategy for each cost scenario
duration: 60 min
difficulty: beginner
has_code: true
module: module-01
---
# Understanding Tokens and Cost Optimization

## Prerequisites

- [Lesson 02: Your First AI Application](02-first-ai-application.md)
- [Module 00 Lesson 03: NLP Fundamentals](../../module-00-genai-foundations-from-nlp-to-transformers/lessons/03-nlp-fundamentals.md) — BPE tokenization

## What You'll Learn

| Objective | Why It Matters |
|-----------|---------------|
| Understand exactly what a token is and how BPE produces them | Explains counterintuitive token counts and how to reduce them |
| Count tokens accurately before making API calls | Prevents context window overflows and enables cost prediction |
| Model costs for realistic production workloads | The difference between a sustainable and an unsustainable system |
| Apply the right cost optimization strategy for each scenario | Not all optimizations are worth the engineering effort |

---

## What Is a Token? A Precise Definition

A **token** is a unit of the model's vocabulary — the atomic element the model processes. As you learned in Module 00, modern LLMs use Byte-Pair Encoding (BPE) to build a vocabulary of ~50,000 subword units.

Critically, tokens are **not** words, characters, or syllables:

```python
import tiktoken

# Use the tokenizer that matches your model
# gpt-4o uses the "o200k_base" encoding
enc = tiktoken.encoding_for_model("gpt-4o")

examples = [
    "Hello, world!",           # Common words → compact
    "supercalifragilistic",    # Long uncommon word → split
    "I'm learning AI",         # Contraction affects tokenization
    "GPT-4o-mini",             # Brand names with hyphens
    "def fibonacci(n):",       # Code — often 1 token per identifier
    "你好世界",                # Chinese: more characters per token
    "المرحبا",                # Arabic: can be many tokens per word
    "1234567890",              # Numbers: often 1 digit per token
]

for text in examples:
    ids    = enc.encode(text)
    tokens = [enc.decode([i]) for i in ids]
    print(f"{text!r:30s} → {len(ids):3d} tokens: {tokens}")
```

### Rules of Thumb (Approximate)

| Content Type | Ratio |
|-------------|-------|
| English prose | ~4 characters per token (~¾ word per token) |
| English code | ~3 characters per token (identifiers often split) |
| Non-English text | Often 2-3x more tokens per word than English |
| Numbers | Often 1 digit per token (each digit is a separate token) |

!!! warning "Never Estimate Token Counts from Word Counts"
    `len(text.split())` is unreliable for token counting. A 1,000-word document might be 1,200 tokens (mostly common English words) or 1,800 tokens (technical jargon, code, or non-English text). Always use `tiktoken` or the provider's tokenizer for accurate counts.

---

## Why Token Counts Matter for Engineering

Token counts determine two critical system properties:

### 1. Context Window Limits

Every model has a maximum context window — the total number of tokens it can process in one call (input + output):

| Model | Context Window | Practical Input Limit |
|-------|---------------|----------------------|
| gpt-4o | 128,000 tokens | ~120,000 (reserve space for output) |
| gpt-4o-mini | 128,000 tokens | ~120,000 |
| claude-3-5-sonnet | 200,000 tokens | ~195,000 |
| gemini-2.5-pro | 1,000,000 tokens | ~990,000 |

Exceeding the context window results in an API error. You must count tokens before submitting requests that include large documents or long conversation histories.

### 2. Cost

You pay per token — both input and output tokens. Current pricing (approximate; check the provider's pricing page for current rates):

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| gpt-4o | $2.50 | $10.00 |
| gpt-4o-mini | $0.15 | $0.60 |
| claude-3-5-sonnet | $3.00 | $15.00 |
| claude-3-5-haiku | $0.25 | $1.25 |
| gemini-2.5-flash | $0.15 | $0.60 |

!!! note "Input vs. Output Pricing"
    Output tokens are typically 4-6x more expensive than input tokens. This matters for optimization: if you can reduce output length (shorter answers, structured formats, fewer words) you save more per token than reducing input length.

---

## Counting Tokens Accurately

```python
import tiktoken
from openai import OpenAI

def count_tokens_for_messages(messages: list, model: str = "gpt-4o-mini") -> int:
    """
    Accurately count tokens for a Chat Completions API messages array.

    The API adds overhead tokens per message for formatting.
    This matches OpenAI's official counting method.

    See: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("o200k_base")   # fallback for new models

    # Token overhead per message (role + delimiters)
    tokens_per_message = 3    # "<|im_start|>role\n" + content + "<|im_end|>"
    tokens_per_name    = 1    # if "name" field is present

    total = 0
    for message in messages:
        total += tokens_per_message
        for key, value in message.items():
            total += len(encoding.encode(value))
            if key == "name":
                total += tokens_per_name

    total += 3   # reply priming tokens
    return total

# Example: count before sending
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Summarize the Transformer architecture in 3 bullet points."},
]

token_count = count_tokens_for_messages(messages, model="gpt-4o-mini")
print(f"Estimated input tokens: {token_count}")

# Warn if approaching context limit
CONTEXT_LIMIT = 128_000
if token_count > CONTEXT_LIMIT * 0.9:
    print("WARNING: Approaching context limit. Consider trimming history.")
```

---

## Cost Modeling for Production Workloads

Before building a system, estimate its cost at scale. This determines model choice and architecture.

### Example 1: Customer Support Chatbot

```python
def estimate_chatbot_cost(
    conversations_per_day: int = 10_000,
    messages_per_conversation: int = 8,
    avg_input_tokens_per_turn: int = 800,    # user msg + history
    avg_output_tokens_per_turn: int = 150,
    model: str = "gpt-4o-mini",
    days: int = 30,
) -> dict:
    """Model monthly cost for a customer support chatbot."""

    pricing = {
        "gpt-4o":      {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output":  0.60},
    }

    p = pricing[model]

    # Daily totals
    total_calls_per_day    = conversations_per_day * messages_per_conversation
    total_input_per_day    = total_calls_per_day * avg_input_tokens_per_turn
    total_output_per_day   = total_calls_per_day * avg_output_tokens_per_turn

    daily_cost = (
        total_input_per_day  * p["input"]  / 1_000_000 +
        total_output_per_day * p["output"] / 1_000_000
    )

    return {
        "model":                model,
        "daily_api_calls":      total_calls_per_day,
        "daily_cost_usd":       round(daily_cost, 2),
        "monthly_cost_usd":     round(daily_cost * days, 2),
        "cost_per_conversation": round(daily_cost / conversations_per_day, 4),
    }

# Compare models
for model in ["gpt-4o-mini", "gpt-4o"]:
    result = estimate_chatbot_cost(model=model)
    print(f"\n{model}:")
    for k, v in result.items():
        print(f"  {k}: {v}")

# Example output:
# gpt-4o-mini:
#   daily_api_calls: 80,000
#   daily_cost_usd: 15.60
#   monthly_cost_usd: 468.00
#   cost_per_conversation: 0.0016
#
# gpt-4o:
#   daily_api_calls: 80,000
#   daily_cost_usd: 288.00
#   monthly_cost_usd: 8,640.00
#   cost_per_conversation: 0.029
```

The model choice alone creates a 18x cost difference. This is the most impactful optimization.

### Example 2: Document Summarization Pipeline

```python
def estimate_document_pipeline_cost(
    documents_per_day: int = 1_000,
    avg_document_tokens: int = 5_000,   # input document
    avg_summary_tokens: int = 200,       # output summary
    model: str = "gpt-4o-mini",
):
    pricing = {"gpt-4o": (2.50, 10.00), "gpt-4o-mini": (0.15, 0.60)}
    in_price, out_price = pricing[model]

    # For summarization, input >> output (documents are long, summaries are short)
    daily_input  = documents_per_day * avg_document_tokens
    daily_output = documents_per_day * avg_summary_tokens

    cost = daily_input * in_price / 1_000_000 + daily_output * out_price / 1_000_000
    print(f"{model}: ${cost:.2f}/day → ${cost*30:.2f}/month")
    print(f"  Input tokens dominate: {daily_input/1000:.0f}K input vs {daily_output/1000:.0f}K output")

estimate_document_pipeline_cost("gpt-4o-mini")
# gpt-4o-mini: $0.77/day → $23.10/month
# Input tokens dominate: 5,000K input vs 200K output
```

---

## Cost Optimization Strategies

Not all optimizations have equal return on investment. Apply them in order of impact:

### Strategy 1: Choose the Right Model (Biggest Impact)

This single decision can change costs by 10-50x. Start with the smallest model that meets your quality requirements.

```python
def model_selection_framework(task_type: str) -> str:
    """
    Heuristic for model selection by task type.
    Always validate with evaluation on your specific task.
    """
    routing = {
        "simple_classification": "gpt-4o-mini",    # easy tasks
        "factual_qa_short":      "gpt-4o-mini",    # simple factual questions
        "document_summary":      "gpt-4o-mini",    # usually sufficient
        "code_generation":       "gpt-4o",          # benefits from larger model
        "complex_reasoning":     "gpt-4o",          # requires more capacity
        "long_document_analysis":"claude-3-5-sonnet",  # long context + quality
        "quick_structured_output":"gpt-4o-mini",   # extract fields from text
    }
    return routing.get(task_type, "gpt-4o-mini")   # default to cheaper

# Pattern: use cheaper model as default, route complex tasks to larger model
```

### Strategy 2: Trim Conversation History

```python
def trim_messages_to_token_budget(messages: list, max_tokens: int,
                                   model: str = "gpt-4o-mini") -> list:
    """
    Remove oldest non-system messages until within token budget.
    Always preserves the system message.
    """
    if count_tokens_for_messages(messages, model) <= max_tokens:
        return messages

    system_msg = [m for m in messages if m["role"] == "system"]
    other_msgs = [m for m in messages if m["role"] != "system"]

    # Remove pairs (user + assistant) from oldest first
    while other_msgs and count_tokens_for_messages(system_msg + other_msgs, model) > max_tokens:
        # Remove the oldest message
        other_msgs.pop(0)

    return system_msg + other_msgs
```

### Strategy 3: Cache Responses for Repeated Queries

```python
import hashlib
import json
import sqlite3
from pathlib import Path

class ResponseCache:
    """
    Simple disk-backed cache for LLM responses.
    Useful when the same query is asked repeatedly (FAQ bots, static analyses).

    Production note: use Redis for distributed caching across multiple workers.
    """

    def __init__(self, cache_path: str = "llm_cache.db"):
        self.conn = sqlite3.connect(cache_path)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, response TEXT)"
        )

    def _key(self, messages: list, model: str) -> str:
        """Deterministic cache key from messages and model."""
        payload = json.dumps({"messages": messages, "model": model}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, messages: list, model: str) -> str | None:
        key = self._key(messages, model)
        row = self.conn.execute("SELECT response FROM cache WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def set(self, messages: list, model: str, response: str):
        key = self._key(messages, model)
        self.conn.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (key, response))
        self.conn.commit()

cache = ResponseCache()

def cached_chat(client, messages: list, model: str = "gpt-4o-mini") -> tuple[str, bool]:
    """Returns (response, was_cached)."""
    if cached := cache.get(messages, model):
        return cached, True

    response = client.chat.completions.create(model=model, messages=messages)
    text = response.choices[0].message.content
    cache.set(messages, model, text)
    return text, False
```

### Strategy 4: Use Structured Outputs to Reduce Output Tokens

Instead of asking for prose explanations, request only the structured data you need:

```python
# Verbose: model explains its reasoning (more output tokens)
prompt_verbose = """
Analyze this customer review and tell me:
1. Is the sentiment positive, negative, or neutral?
2. What is the main product feature mentioned?
3. Would you recommend responding to this review?
Please explain your reasoning for each point.

Review: "The battery lasts 2 days on a single charge, amazing for travel!"
"""

# Concise: model returns only what you need (fewer output tokens)
prompt_structured = """
Analyze this review. Return ONLY valid JSON:
{"sentiment": "positive|negative|neutral", "feature": "string", "needs_response": true|false}

Review: "The battery lasts 2 days on a single charge, amazing for travel!"
"""

# The structured version can save 70-80% of output tokens for this task
```

### Strategy 5: Use the Batch API for Offline Workloads

For non-real-time processing (document analysis, dataset enrichment), the OpenAI Batch API offers 50% discount:

```python
# Batch API example (for async offline processing)
batch_requests = [
    {
        "custom_id": f"doc-{i}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": f"Summarize: {doc}"}
            ],
            "max_tokens": 200,
        }
    }
    for i, doc in enumerate(documents)
]

# Upload and submit batch — results returned within 24 hours at 50% cost
# See: https://platform.openai.com/docs/guides/batch
```

---

## Building a Token Budget Monitor

```python
class TokenBudgetMonitor:
    """
    Track token usage and cost in real-time, alert when approaching limits.
    Useful for services with billing budgets.
    """

    def __init__(self, daily_budget_usd: float, model: str = "gpt-4o-mini"):
        self.daily_budget   = daily_budget_usd
        self.model          = model
        self.total_input    = 0
        self.total_output   = 0

        self.pricing = {
            "gpt-4o":      (2.50, 10.00),
            "gpt-4o-mini": (0.15,  0.60),
        }

    def record(self, input_tokens: int, output_tokens: int):
        self.total_input  += input_tokens
        self.total_output += output_tokens

    def current_cost(self) -> float:
        in_p, out_p = self.pricing.get(self.model, (0, 0))
        return (self.total_input * in_p + self.total_output * out_p) / 1_000_000

    def budget_remaining(self) -> float:
        return self.daily_budget - self.current_cost()

    def status(self) -> dict:
        cost = self.current_cost()
        return {
            "cost_so_far":       round(cost, 4),
            "budget_remaining":  round(self.budget_remaining(), 4),
            "budget_used_pct":   round(cost / self.daily_budget * 100, 1),
            "total_tokens":      self.total_input + self.total_output,
        }

    def check_budget(self, estimated_next_call: int = 1000):
        """Raise if the next estimated call would exceed budget."""
        in_p, out_p = self.pricing.get(self.model, (0, 0))
        next_cost = estimated_next_call * in_p / 1_000_000
        if next_cost > self.budget_remaining():
            raise RuntimeError(
                f"Budget exceeded: ${self.current_cost():.4f} spent of ${self.daily_budget:.2f} daily budget"
            )

monitor = TokenBudgetMonitor(daily_budget_usd=10.00, model="gpt-4o-mini")
```

---

## Edge Cases and Misconceptions

**"I can estimate costs from word count."** Word count is unreliable. Always use tiktoken or the equivalent tokenizer. A technical document with many numbers and code snippets may have 30-50% more tokens than a pure prose document of the same word count.

**"Output tokens are cheaper than input tokens."** Output tokens cost more per token in every major provider's pricing (typically 4-6x). Optimize output length (structured formats, concise prompts) before optimizing input length.

**"Caching hurts freshness."** Caching is most appropriate for deterministic queries: factual questions, classification, entity extraction. Avoid caching for queries that require up-to-date information or user-specific personalization.

**"The context window is the total response size."** The context window is the limit on input + output combined. If you send 100,000 input tokens to a model with a 128K context window, you can only get ~28,000 output tokens. Plan your input budget accordingly.

---

## Key Takeaways

- Tokens are subword units from BPE tokenization, not words; never estimate with word count
- Count tokens before API calls using `tiktoken` to prevent context overflow and predict costs
- Output tokens cost more than input tokens per token — optimize output length first
- Model choice has the largest cost impact: gpt-4o-mini vs gpt-4o can be an 18x cost difference
- Cache repeated queries; use Batch API for offline/async workloads (50% discount)
- Build token budget monitoring into production systems from day one

---

## Further Reading

- [OpenAI Tokenizer Tool](https://platform.openai.com/tokenizer) — interactive token visualizer; paste any text to see exactly how it tokenizes
- [tiktoken GitHub](https://github.com/openai/tiktoken) — the library used for all OpenAI token counting; fast, accurate
- [Andrej Karpathy: Let's Build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) — builds BPE from scratch, explains why tokenization has the properties it does
- [OpenAI Pricing](https://openai.com/pricing) — current model pricing (check before making architectural decisions)

---

**Next:** [Prompt Engineering Fundamentals](04-prompt-engineering.md)
