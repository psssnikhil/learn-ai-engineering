---
title: Understanding Tokens and Cost Optimization
description: >-
  Master token economics, pricing, and optimization strategies for AI
  applications
duration: 35 min
difficulty: beginner
has_code: true
module: module-01
youtube: 'https://www.youtube.com/watch?v=5xUk6CRPL_k'
objectives:
  - Calculate token counts for messages
  - Estimate API costs accurately
  - Implement token optimization
---
# Understanding Tokens & Cost Optimization

![Tokens](https://images.unsplash.com/photo-1579621970795-87facc2f976d?w=800)

## What Are Tokens?

Tokens are the fundamental units that LLMs process. They're NOT words!

### Examples

```
"Hello world" = 2 tokens
"artificial intelligence" = 2 tokens  
"I'm learning AI" = 4 tokens
"GPT-4o-mini" = 4 tokens (G, PT, -, 4, o, -, mini)
```

### Rules of Thumb

- 1 token ≈ 4 characters in English
- 1 token ≈ ¾ of a word
- 100 tokens ≈ 75 words
- 1,000 tokens ≈ 750 words

## Why Tokens Matter

### 1. Pricing 💰
You pay per token!

**GPT-4o Pricing:**
- Input: $2.50 per 1M tokens
- Output: $10.00 per 1M tokens

**GPT-4o-mini Pricing:**
- Input: $0.15 per 1M tokens
- Output: $0.60 per 1M tokens

### 2. Context Windows
Models have token limits:

| Model | Context Window |
|-------|---------------|
| GPT-4o | 128,000 tokens |
| GPT-4o-mini | 128,000 tokens |
| Claude 3.5 Sonnet | 200,000 tokens |
| Claude 3.5 Haiku | 200,000 tokens |
| Gemini 1.5 Pro | 2,000,000 tokens |

### 3. Speed
More tokens = slower response

## Cost Optimization Strategies

### Strategy 1: Choose the Right Model

Don't always use the biggest model!

```python
# Bad: Using GPT-4o for simple tasks
response = openai.chat.completions.create(
    model="gpt-4o",  # Expensive!
    messages=[{"role": "user", "content": "Say hello"}]
)

# Good: Using GPT-4o-mini for simple tasks
response = openai.chat.completions.create(
    model="gpt-4o-mini",  # 16x cheaper!
    messages=[{"role": "user", "content": "Say hello"}]
)
```

### Strategy 2: Limit Output Tokens

```python
# Control max response length
response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Explain AI"}],
    max_tokens=100  # Cap the output!
)
```

### Strategy 3: Compress System Prompts

```python
# Bad: Verbose system prompt (150 tokens)
system = '''
You are a highly capable AI assistant designed to help users
with their questions. You should always be polite, professional,
and provide detailed explanations. Never be rude or unhelpful.
Make sure to always...
'''

# Good: Concise system prompt (20 tokens)
system = 'You are a helpful, concise AI assistant.'
```

### Strategy 4: Use Caching

Cache common responses:

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_ai_response(prompt: str):
    # Only calls API once per unique prompt
    return openai.chat.completions.create(...)
```

## Real Cost Examples

### Example 1: Customer Support Chatbot

**Specs:**
- 10,000 conversations/day
- Average: 10 messages per conversation
- Average: 50 tokens per message
- Model: GPT-4o-mini

**Calculation:**
```
Input tokens/day = 10,000 × 10 × 50 = 5,000,000
Output tokens/day = 10,000 × 10 × 50 = 5,000,000

Input cost = 5M × $0.15 / 1M = $0.75/day
Output cost = 5M × $0.60 / 1M = $3.00/day

Total = $3.75/day = $112.50/month = $1,350/year
```

### Example 2: Document Summarization

**Specs:**
- 1,000 documents/day
- Average: 5,000 tokens per document
- Summary: 200 tokens
- Model: GPT-4o-mini

**Calculation:**
```
Input tokens/day = 1,000 × 5,000 = 5,000,000
Output tokens/day = 1,000 × 200 = 200,000

Input cost = 5M × $0.15 / 1M = $0.75/day
Output cost = 0.2M × $0.60 / 1M = $0.12/day

Total = $0.87/day = $26.10/month
```

## Code Exercise

Build a token counter and cost calculator!

---

## 📹 Recommended Videos

- [What are Tokens? LLM Tokenization Explained](https://www.youtube.com/watch?v=5xUk6CRPL_k) — Visual explanation of tokenization
- [Andrej Karpathy: Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) — Deep dive into BPE tokenization
- [How to Reduce OpenAI API Costs](https://www.youtube.com/watch?v=Z7_JCGikPEs) — Practical cost optimization tips

---

## 📚 Additional Resources

### Articles & Blogs:
- [OpenAI Tokenizer Tool](https://platform.openai.com/tokenizer) — Interactive token counter
- [OpenAI Pricing](https://openai.com/pricing) — Current model pricing
- [Anthropic Pricing](https://www.anthropic.com/pricing) — Claude model pricing
- [Understanding Tokens](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them) — OpenAI guide on tokens

### Tools:
- [tiktoken](https://github.com/openai/tiktoken) — OpenAI's fast token counting library
- [LLM Price Check](https://llmpricecheck.com/) — Compare pricing across providers
