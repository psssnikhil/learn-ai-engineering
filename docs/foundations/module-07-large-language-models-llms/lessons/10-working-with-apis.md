---
title: Working with LLM APIs
description: Master how to use LLM APIs effectively for building applications
duration: 40 min
difficulty: intermediate
has_code: false
module: module-07
youtube: 'https://www.youtube.com/watch?v=T9aRN5JkmL8'
---
# Working with LLM APIs

## Popular APIs

1. **OpenAI**: GPT-4, GPT-3.5
2. **Anthropic**: Claude 3 (Opus, Sonnet, Haiku)
3. **Google**: Gemini Pro
4. **Cohere**: Command, Generate
5. **Mistral**: Mistral Large, Medium
6. **Open Source**: LLaMA via Together, Replicate

## OpenAI API Example

```python
from openai import OpenAI

client = OpenAI(api_key="your-api-key")

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing simply"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

## Key Parameters

**temperature** (0-2):
- 0: Deterministic, focused
- 1: Balanced
- 2: Very creative, random

**max_tokens**: Maximum response length

**top_p** (0-1): Nucleus sampling
- 0.1: Very focused
- 1.0: Consider all tokens

**frequency_penalty** (-2 to 2): Reduce repetition

**presence_penalty** (-2 to 2): Encourage new topics

## Streaming Responses

```python
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Write a story"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

## Function Calling

```python
functions = [
    {
        "name": "get_weather",
        "description": "Get current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    functions=functions,
    function_call="auto"
)

# Model returns: function call to execute
```

## Cost Management

```python
import tiktoken

def count_tokens(text, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

prompt = "Long prompt here..."
tokens = count_tokens(prompt)
estimated_cost = (tokens / 1000) * 0.03  # GPT-4 input pricing

print(f"Tokens: {tokens}, Est. cost: ${estimated_cost:.4f}")
```

## Best Practices

1. **System prompts**: Set behavior/persona
2. **Few-shot examples**: Show desired format
3. **Error handling**: Rate limits, retries
4. **Caching**: Cache common responses
5. **Async calls**: Parallel requests
6. **Token limits**: Monitor usage

## Error Handling

```python
from openai import RateLimitError, APIError
import time

def call_with_retry(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            return response
        except RateLimitError:
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"Rate limit hit, waiting {wait_time}s...")
            time.sleep(wait_time)
        except APIError as e:
            print(f"API error: {e}")
            return None
```

---

## 🎉 Module Complete!

**You've mastered LLMs**:
- ✅ What LLMs are and how they work
- ✅ Evolution from Transformers
- ✅ Pre-training strategies
- ✅ Tokenization
- ✅ Embeddings
- ✅ Fine-tuning techniques
- ✅ Instruction tuning
- ✅ RLHF alignment
- ✅ Major architectures
- ✅ API integration

**Next Module**: AI Engineering Essentials
- Building production AI apps
- Prompt engineering mastery
- Cost optimization
- Testing & deployment

**Keep building!** 🚀

---

## 📹 Recommended Videos

- [OpenAI API Full Tutorial](https://www.youtube.com/watch?v=T9aRN5JkmL8) — Complete guide to the OpenAI API
- [Claude API Getting Started](https://www.youtube.com/watch?v=hkhDdcM5V94) — Anthropic Claude API tutorial
- [LLM API Comparison](https://www.youtube.com/watch?v=qYSWDk4-NHI) — OpenAI vs Anthropic vs Google

---

## 📚 Additional Resources

- [OpenAI API Reference](https://platform.openai.com/docs/api-reference) — Official OpenAI docs
- [Anthropic API Docs](https://docs.anthropic.com/) — Claude API reference
- [Google AI Studio](https://ai.google.dev/) — Gemini API documentation
- [LiteLLM](https://github.com/BerriAI/litellm) — Unified interface for 100+ LLM providers
