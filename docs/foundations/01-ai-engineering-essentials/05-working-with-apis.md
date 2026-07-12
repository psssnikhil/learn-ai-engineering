---
title: 'Working with LLM APIs - OpenAI, Anthropic, and More'
description: >-
  Master modern LLM APIs. Learn streaming, function calling, structured outputs,
  error handling, and cost optimization for production systems.
duration: 60 min
difficulty: intermediate
has_code: false
youtube: 'https://www.youtube.com/watch?v=eC6vEkmqWMA'
objectives:
  - Make API calls to multiple LLM providers
  - Implement streaming responses
  - Use function calling for tool integration
  - Handle rate limits and errors gracefully
  - Calculate and optimize API costs
---
# Working with LLM APIs - Production Guide 🔌

## 📋 What You'll Learn

By the end of this lesson, you'll understand:
- ✅ How to use OpenAI, Anthropic, and Google APIs
- ✅ Streaming responses for better UX
- ✅ Function calling for tool integration
- ✅ Error handling and retry strategies
- ✅ Cost optimization techniques

---

## 🎯 Chapter 1: The Major LLM Providers

### Provider Comparison (2025)

| Provider | Best Models | Strengths | Pricing |
|----------|------------|-----------|---------|
| **OpenAI** | GPT-4, GPT-4-turbo, GPT-3.5 | General purpose, function calling | $$$ |
| **Anthropic** | Claude 3 Opus, Sonnet, Haiku | Long context (200K), safety | $$ |
| **Google** | Gemini Pro, Ultra | Multimodal, fast | $$ |
| **Meta** | LLaMA 3 (70B, 405B) | Open-source, self-host | Free |
| **Mistral** | Mistral Large, Medium | European, fast | $ |

### Installation

```bash
# Install SDKs
pip install openai anthropic google-generativeai

# Set API keys (never hardcode!)
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

---

## 🎯 Chapter 2: OpenAI API

### Basic Chat Completion

```python
from openai import OpenAI

client = OpenAI()  # Reads OPENAI_API_KEY from environment

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing in simple terms"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

### Streaming Responses

```python
def stream_response(prompt):
    """
    Stream tokens as they're generated (better UX!)
    """
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        stream=True  # Enable streaming!
    )
    
    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            full_response += token
            print(token, end="", flush=True)  # Print immediately
    
    return full_response

# Usage
response = stream_response("Write a poem about AI")
# Tokens appear one by one, like typing! ✨
```

### Function Calling (Tools)

```python
import json

# Define tools the LLM can use
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name, e.g., San Francisco"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "Temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    }
]

# Actual function implementations
def get_weather(location, unit="celsius"):
    """Mock weather API"""
    # In production, call real weather API
    return {
        "location": location,
        "temperature": 72 if unit == "fahrenheit" else 22,
        "condition": "Sunny"
    }

# Chat with function calling
def chat_with_tools(user_message):
    messages = [{"role": "user", "content": user_message}]
    
    # Initial request
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools,
        tool_choice="auto"  # Let model decide when to call tools
    )
    
    response_message = response.choices[0].message
    
    # Check if model wants to call a function
    if response_message.tool_calls:
        # Model decided to call a function!
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # Call the actual function
            if function_name == "get_weather":
                function_response = get_weather(**function_args)
            
            # Add function response to conversation
            messages.append(response_message)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": json.dumps(function_response)
            })
        
        # Get final response with function results
        final_response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        
        return final_response.choices[0].message.content
    
    return response_message.content

# Test it!
result = chat_with_tools("What's the weather like in San Francisco?")
print(result)
# "The weather in San Francisco is currently sunny with a temperature of 72°F."
```

### Structured Outputs

```python
from pydantic import BaseModel

class MovieRecommendation(BaseModel):
    """Structured output schema"""
    title: str
    year: int
    genre: list[str]
    rating: float
    why_recommended: str

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Recommend a sci-fi movie"}],
    response_format={"type": "json_object"},  # Force JSON output
)

# Parse response
import json
movie = MovieRecommendation(**json.loads(response.choices[0].message.content))

print(f"{movie.title} ({movie.year})")
print(f"Genre: {', '.join(movie.genre)}")
print(f"Rating: {movie.rating}/10")
print(f"Why: {movie.why_recommended}")
```

---

## 🎯 Chapter 3: Anthropic Claude API

### Basic Usage

```python
from anthropic import Anthropic

client = Anthropic()  # Reads ANTHROPIC_API_KEY from environment

response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "Explain neural networks"}
    ]
)

print(response.content[0].text)
```

### System Prompts (Claude specialty)

```python
# Claude excels with detailed system prompts
response = client.messages.create(
    model="claude-3-opus-20240229",
    max_tokens=2000,
    system="""You are an expert AI tutor. Your teaching style:
    1. Start with intuition, then dive into details
    2. Use analogies and examples
    3. Check understanding with questions
    4. Build from simple to complex
    """,
    messages=[
        {"role": "user", "content": "Teach me about transformers"}
    ]
)
```

### Streaming with Claude

```python
def stream_claude(prompt):
    with client.messages.stream(
        model="claude-3-opus-20240229",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)

stream_claude("Write a haiku about coding")
```

---

## 🎯 Chapter 4: Error Handling & Retries

### Common Errors

```python
from openai import OpenAI, APIError, RateLimitError, APIConnectionError
import time

def robust_api_call(prompt, max_retries=3):
    """
    Handle common API errors gracefully
    """
    client = OpenAI()
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        
        except RateLimitError:
            # Too many requests, wait and retry
            wait_time = (2 ** attempt)  # Exponential backoff
            print(f"⏳ Rate limited. Waiting {wait_time}s...")
            time.sleep(wait_time)
        
        except APIConnectionError:
            # Network issue, retry
            print(f"🌐 Connection error. Retrying ({attempt + 1}/{max_retries})...")
            time.sleep(1)
        
        except APIError as e:
            # General API error
            print(f"❌ API Error: {e}")
            if e.status_code >= 500:  # Server error, retry
                time.sleep(2)
            else:  # Client error (bad request), don't retry
                raise
    
    raise Exception("Max retries exceeded")

# Usage
result = robust_api_call("Hello, world!")
```

### Rate Limiting

```python
from ratelimit import limits, sleep_and_retry
import time

# OpenAI limits (example: 3 requests per minute for tier 1)
CALLS_PER_MINUTE = 3

@sleep_and_retry
@limits(calls=CALLS_PER_MINUTE, period=60)
def rate_limited_api_call(prompt):
    """
    Automatically handles rate limiting
    """
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Make multiple calls - will automatically wait if needed
for i in range(10):
    result = rate_limited_api_call(f"Question {i}")
    print(f"✅ Response {i} received")
```

---

## 🎯 Chapter 5: Cost Optimization

### Calculate Costs

```python
import tiktoken

def estimate_cost(prompt, response, model="gpt-4"):
    """
    Calculate API call cost
    """
    # Pricing (as of 2024, check current prices!)
    pricing = {
        "gpt-4": {"input": 0.03, "output": 0.06},  # per 1K tokens
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "claude-3-opus": {"input": 0.015, "output": 0.075}
    }
    
    # Count tokens
    enc = tiktoken.encoding_for_model(model)
    input_tokens = len(enc.encode(prompt))
    output_tokens = len(enc.encode(response))
    
    # Calculate cost
    input_cost = (input_tokens / 1000) * pricing[model]["input"]
    output_cost = (output_tokens / 1000) * pricing[model]["output"]
    
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_cost": input_cost + output_cost
    }

# Example
prompt = "Explain machine learning"
response = "Machine learning is..."
cost = estimate_cost(prompt, response, "gpt-4")

print(f"Input: {cost['input_tokens']} tokens")
print(f"Output: {cost['output_tokens']} tokens")
print(f"Total cost: ${cost['total_cost']:.4f}")
```

### Optimization Strategies

```python
class CostOptimizer:
    """
    Strategies to reduce API costs
    """
    
    @staticmethod
    def use_cheaper_model_first(prompt):
        """
        Try GPT-3.5 first, upgrade to GPT-4 if needed
        """
        # Try cheap model
        response_35 = call_api(prompt, model="gpt-3.5-turbo")
        
        # Check if response is good (use a classifier or heuristic)
        if is_good_response(response_35):
            return response_35  # Save money! 💰
        
        # Fall back to expensive model
        return call_api(prompt, model="gpt-4")
    
    @staticmethod
    def cache_responses(prompt):
        """
        Cache responses to avoid duplicate API calls
        """
        import hashlib
        import json
        
        # Create cache key
        cache_key = hashlib.md5(prompt.encode()).hexdigest()
        
        # Check cache
        try:
            with open(f"cache/{cache_key}.json", "r") as f:
                return json.load(f)["response"]
        except FileNotFoundError:
            pass
        
        # Make API call
        response = call_api(prompt)
        
        # Save to cache
        with open(f"cache/{cache_key}.json", "w") as f:
            json.dump({"prompt": prompt, "response": response}, f)
        
        return response
    
    @staticmethod
    def batch_requests(prompts):
        """
        Batch multiple requests (when possible)
        
        Note: Not all APIs support batching
        """
        # Some models support batch API (cheaper!)
        # OpenAI Batch API: 50% discount but delayed responses
        pass
    
    @staticmethod
    def optimize_prompt_length(prompt):
        """
        Remove unnecessary tokens
        """
        # Remove extra whitespace
        prompt = " ".join(prompt.split())
        
        # Remove redundant instructions
        prompt = prompt.replace("Please", "").replace("thank you", "")
        
        # Use abbreviations where clear
        # "for example" → "e.g."
        
        return prompt
```

---

## 💻 Your Turn: Build a Multi-Provider Client

Create a unified interface for multiple LLM providers:

```python
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    """Base class for LLM providers"""
    
    @abstractmethod
    def chat(self, messages, **kwargs):
        """Send chat request"""
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key=None):
        # TODO: Initialize OpenAI client
        pass
    
    def chat(self, messages, **kwargs):
        # TODO: Implement OpenAI chat
        pass

class AnthropicProvider(LLMProvider):
    def __init__(self, api_key=None):
        # TODO: Initialize Anthropic client
        pass
    
    def chat(self, messages, **kwargs):
        # TODO: Implement Anthropic chat
        pass

class UnifiedLLM:
    """Unified interface for all providers"""
    
    def __init__(self, primary="openai", fallback="anthropic"):
        self.providers = {
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider()
        }
        self.primary = primary
        self.fallback = fallback
    
    def chat(self, messages, **kwargs):
        """
        TODO: Implement with fallback
        
        1. Try primary provider
        2. If fails, try fallback
        3. Return response or error
        """
        pass

# Test it
llm = UnifiedLLM(primary="openai", fallback="anthropic")
response = llm.chat([{"role": "user", "content": "Hello!"}])
print(response)
```

---

## 🎓 Summary

### What We Learned

1. ✅ **Multiple providers** available (OpenAI, Anthropic, Google, etc.)
2. ✅ **Streaming** improves user experience
3. ✅ **Function calling** enables tool use
4. ✅ **Error handling** is critical for production
5. ✅ **Cost optimization** saves money at scale

### Best Practices

```python
production_checklist = {
    "api_keys": "Store in environment variables (never commit!)",
    "error_handling": "Always implement retries with exponential backoff",
    "rate_limiting": "Respect provider limits",
    "cost_tracking": "Monitor spend per request",
    "caching": "Cache responses when possible",
    "fallbacks": "Have backup providers",
    "streaming": "Use for better UX",
    "logging": "Log all requests for debugging"
}
```

### Cost Comparison (1M tokens)

```python
# Input + Output (as of 2024)
costs_per_1m_tokens = {
    "GPT-4": "$30-60",
    "GPT-3.5-turbo": "$1.50-2",
    "Claude 3 Opus": "$15-75",
    "Claude 3 Haiku": "$0.25-1.25",
    "Gemini Pro": "$0.50-1.50"
}
```

---

## 📚 Additional Resources

### Docs:
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Anthropic Claude Docs](https://docs.anthropic.com/)
- [Google AI Studio](https://ai.google.dev/)

### Tools:
- [LiteLLM](https://github.com/BerriAI/litellm) - Unified interface for 100+ LLMs
- [Tiktoken](https://github.com/openai/tiktoken) - Token counting
- [Helicone](https://www.helicone.ai/) - LLM observability & cost tracking

---

## 🎯 Next Steps

Now that you can work with LLM APIs, you're ready for:
- **Prompt engineering** (next lesson!)
- **Building RAG systems** (Module 4)
- **Creating AI agents** (Module 5)

---

**🎉 You're now a production LLM API expert!** You can build reliable, cost-effective AI applications with any major LLM provider.
