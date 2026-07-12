---
title: Your First AI Application
description: >-
  Set up your environment, understand the Chat Completions API contract, and
  build a stateful multi-turn chatbot with streaming — understanding each design
  decision along the way
duration: 60 min
difficulty: beginner
has_code: true
module: module-01
---
# Your First AI Application

## Prerequisites

- [Lesson 01: What is AI Engineering](01-what-is-ai-engineering.md)
- Python with `pip` installed; basic async/await understanding is helpful but not required
- An OpenAI API key (see setup below)

## What You'll Learn

| Objective | Why It Matters |
|-----------|---------------|
| Understand the Chat Completions API contract | The fundamental interface; every provider follows this or a close variant |
| Build a multi-turn chatbot with conversation history | Production chatbots must maintain state; this is the core pattern |
| Implement streaming responses | Users experience much better UX with streaming; required for production |
| Handle API errors properly | Unhanded errors cause silent failures in production |
| Understand temperature and how it affects output | This is the most-asked-about parameter by new users |

---

## Setup

### Get an OpenAI API Key

1. Create an account at [platform.openai.com](https://platform.openai.com)
2. Navigate to **API Keys** → **Create new secret key**
3. Copy the key immediately — you cannot view it again

### Install Dependencies

```bash
pip install openai python-dotenv
```

### Create a `.env` File

```bash
# .env (add this file to .gitignore — never commit API keys!)
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
```

!!! warning "API Key Security"
    Your API key is a secret credential. Never commit it to version control, never log it, and never pass it as a command-line argument (it appears in process lists). Always load it from environment variables. Set up billing limits in the OpenAI dashboard to prevent unexpected charges.

---

## Understanding the API Contract

The Chat Completions API follows a simple but important pattern: you send a **messages array**, and the model produces the next message in that conversation.

```python
from openai import OpenAI

client = OpenAI()   # reads OPENAI_API_KEY from environment

# The minimal API call
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "What is 2 + 2?"}
    ]
)

print(response.choices[0].message.content)   # "4" or "2 + 2 = 4"
print(response.usage.prompt_tokens)          # number of input tokens
print(response.usage.completion_tokens)      # number of output tokens
```

### The Three Message Roles

Every message has a `role` field. Understanding these roles is essential:

| Role | Purpose | Notes |
|------|---------|-------|
| `system` | Sets the model's behavior, personality, and constraints | Applied at the start; persists through the conversation |
| `user` | Human input (the person typing) | Your user's messages |
| `assistant` | The model's previous responses | Added to history to maintain context |

```python
# A proper conversation structure
messages = [
    {
        "role": "system",
        "content": (
            "You are a helpful technical documentation assistant. "
            "Provide concise, accurate answers with code examples when relevant. "
            "If you are uncertain about something, say so explicitly."
        )
    },
    {
        "role": "user",
        "content": "How does Python's GIL work?"
    },
    # After the model responds, we add:
    # {"role": "assistant", "content": "<model's response>"},
    # Then the next user message:
    # {"role": "user", "content": "Does it affect async code?"}
]
```

!!! note "Why Conversation History Is in the Request"
    The API is **stateless** — each call is independent. The model has no memory of previous calls unless you send the previous messages in the current request. This is why multi-turn chatbots must maintain their own conversation history and send it with each new message. It also means you pay for all historical messages every turn.

---

## Building a Multi-Turn Chatbot

```python
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

class Chatbot:
    """
    A stateful chatbot that maintains conversation history.
    The history list grows with each turn and is sent with every request.
    """

    def __init__(self, system_prompt: str = "You are a helpful assistant.",
                 model: str = "gpt-4o-mini"):
        self.model    = model
        self.history  = [{"role": "system", "content": system_prompt}]
        self.total_tokens = 0

    def chat(self, user_message: str) -> str:
        """
        Send a user message and return the assistant's response.
        Automatically maintains conversation history.
        """
        # Add user message to history
        self.history.append({"role": "user", "content": user_message})

        # Call the API with full conversation history
        response = client.chat.completions.create(
            model=self.model,
            messages=self.history,
            temperature=0.7,
            max_tokens=1024,
        )

        # Extract and store the assistant's response
        assistant_message = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": assistant_message})

        # Track token usage for cost monitoring
        self.total_tokens += response.usage.total_tokens

        return assistant_message

    def reset(self):
        """Clear conversation history (keep system prompt)."""
        self.history = [self.history[0]]  # keep only system prompt
        self.total_tokens = 0

    def get_token_count(self) -> int:
        """Return total tokens used in this conversation."""
        return self.total_tokens


# Example usage
bot = Chatbot(
    system_prompt="You are a Python expert. Always show runnable code examples.",
    model="gpt-4o-mini"
)

# Turn 1
response_1 = bot.chat("What is a list comprehension?")
print("Bot:", response_1)
print(f"Tokens so far: {bot.get_token_count()}")

# Turn 2 — the model remembers the previous exchange
response_2 = bot.chat("Can you show me a nested one?")
print("Bot:", response_2)
print(f"Tokens so far: {bot.get_token_count()}")
```

### Why Token Count Grows with Conversation Length

```
Turn 1 API call: [system, user_1]                    → 50 tokens
Turn 2 API call: [system, user_1, assistant_1, user_2] → 200 tokens
Turn 3 API call: [system, user_1, assistant_1, user_2, assistant_2, user_3] → 500 tokens
```

Each turn pays for ALL previous turns again. A 100-turn conversation can accumulate thousands of tokens of history. Production systems handle this with:
- **Window trimming**: keep only the last N turns
- **Summarization**: compress old history into a shorter summary
- **Selective retention**: keep only the most relevant past exchanges

---

## Streaming Responses

Without streaming, the user sees nothing until the entire response is generated. For a response that takes 5 seconds, this is a poor experience. Streaming sends tokens as they are generated:

```python
def stream_chat(client: OpenAI, messages: list, model: str = "gpt-4o-mini") -> str:
    """
    Stream a response token-by-token.
    Returns the complete response as a string after streaming finishes.
    """
    full_response = ""

    # stream=True returns a generator of token chunks
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True
    )

    for chunk in stream:
        # Each chunk may contain one or more tokens
        if chunk.choices and chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            full_response += token
            print(token, end="", flush=True)  # print immediately, no newline

    print()  # newline after streaming completes
    return full_response

# Use streaming in a chatbot
messages = [
    {"role": "system", "content": "You are a storyteller. Write vivid, engaging prose."},
    {"role": "user", "content": "Write a one paragraph story about a robot discovering music."}
]

response = stream_chat(client, messages)
print(f"\nTotal characters: {len(response)}")
```

!!! note "Streaming and Token Counting"
    When using streaming, you do not get `response.usage` in the stream chunks by default. To get token counts with streaming, set `stream_options={"include_usage": True}` in the API call. The usage will appear in the final chunk.

---

## Understanding Temperature

Temperature controls how the model samples from its output probability distribution:

```python
def demonstrate_temperature(prompt: str, temperatures: list):
    """
    Show how temperature affects output diversity.
    Run multiple times at each temperature to see the effect.
    """
    for temp in temperatures:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=50
        )
        output = response.choices[0].message.content.strip()
        print(f"Temperature {temp}: {output}")

# Example: creative vs. deterministic output
demonstrate_temperature(
    prompt="Complete this: 'The old lighthouse keeper'",
    temperatures=[0.0, 0.7, 1.5]
)
# Temperature 0.0: consistent, deterministic, "saw the storm approaching"
# Temperature 0.7: varied but coherent
# Temperature 1.5: creative, sometimes unexpected or less coherent
```

**What temperature actually does:** Temperature T scales the logits (raw model scores) before softmax: `p_i = softmax(logits / T)`. At T=0, the highest-scoring token always wins (deterministic). At T=1, the original probability distribution is used. At T>1, lower-probability tokens get more weight.

```
Practical guidelines:
  temperature=0.0:   Facts, math, classification, code where correctness matters
  temperature=0.3:   Technical writing, summarization
  temperature=0.7:   General purpose — balanced default
  temperature=1.0:   Creative writing with mild constraint
  temperature=1.5+:  Brainstorming, highly creative tasks (watch for incoherence)
```

---

## Handling API Errors Properly

API calls fail. Rate limits are real. Network issues happen. Production code must handle this:

```python
import time
from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError

def robust_chat(client: OpenAI, messages: list,
                model: str = "gpt-4o-mini",
                max_retries: int = 3) -> str:
    """
    API call with exponential backoff retry for transient failures.

    Handles:
    - RateLimitError (429): too many requests — wait and retry
    - APIConnectionError: network issue — retry
    - APIStatusError 5xx: server error — retry
    - APIStatusError 4xx: client error — raise (don't retry, you'll keep failing)
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
            )
            return response.choices[0].message.content

        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise  # exhausted retries
            wait_time = (2 ** attempt) * 1.0   # exponential backoff: 1s, 2s, 4s
            print(f"Rate limited. Waiting {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})...")
            time.sleep(wait_time)

        except APIConnectionError:
            if attempt == max_retries - 1:
                raise
            print(f"Connection error. Retrying ({attempt + 1}/{max_retries})...")
            time.sleep(1)

        except APIStatusError as e:
            if e.status_code >= 500:   # server error — might be transient
                if attempt == max_retries - 1:
                    raise
                time.sleep(2)
            else:                       # 4xx — client error, no point retrying
                raise

    raise RuntimeError("max_retries exceeded")  # should not reach here


# Usage
try:
    response = robust_chat(client, [{"role": "user", "content": "Hello!"}])
    print(response)
except RateLimitError:
    print("Error: API rate limit. Consider upgrading your tier or reducing request rate.")
except Exception as e:
    print(f"Unhandled error: {e}")
```

---

## A Complete Chatbot with All Features

```python
import os
import time
from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

class ProductionChatbot:
    """
    A production-ready chatbot with:
    - Streaming responses
    - Conversation history with window trimming
    - Error handling with retry
    - Token and cost tracking
    """

    # Approximate cost per 1M tokens (update from https://openai.com/pricing)
    PRICING = {
        "gpt-4o":      {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output":  0.60},
    }

    def __init__(self, system_prompt: str, model: str = "gpt-4o-mini",
                 max_history_turns: int = 20):
        self.model            = model
        self.max_history_turns = max_history_turns
        self.history          = [{"role": "system", "content": system_prompt}]
        self.total_input_tokens  = 0
        self.total_output_tokens = 0

    def _trim_history(self):
        """Keep only the system prompt + last N turns."""
        system_msg = self.history[0]
        recent     = self.history[1:][-self.max_history_turns * 2:]  # 2 msgs per turn
        self.history = [system_msg] + recent

    def chat(self, user_message: str, stream: bool = True) -> str:
        self.history.append({"role": "user", "content": user_message})
        self._trim_history()

        for attempt in range(3):
            try:
                if stream:
                    return self._stream_response()
                else:
                    return self._sync_response()
            except RateLimitError:
                time.sleep(2 ** attempt)
            except APIConnectionError:
                time.sleep(1)

        raise RuntimeError("API unavailable after 3 attempts")

    def _stream_response(self) -> str:
        full_response = ""
        stream = client.chat.completions.create(
            model=self.model,
            messages=self.history,
            stream=True,
            stream_options={"include_usage": True},
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response += token
                print(token, end="", flush=True)
            if chunk.usage:
                self.total_input_tokens  += chunk.usage.prompt_tokens
                self.total_output_tokens += chunk.usage.completion_tokens
        print()

        self.history.append({"role": "assistant", "content": full_response})
        return full_response

    def _sync_response(self) -> str:
        response = client.chat.completions.create(
            model=self.model,
            messages=self.history,
        )
        text = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": text})
        self.total_input_tokens  += response.usage.prompt_tokens
        self.total_output_tokens += response.usage.completion_tokens
        return text

    def estimated_cost(self) -> float:
        pricing = self.PRICING.get(self.model, {"input": 0.0, "output": 0.0})
        return (self.total_input_tokens  * pricing["input"]  / 1_000_000 +
                self.total_output_tokens * pricing["output"] / 1_000_000)

    def stats(self) -> dict:
        return {
            "input_tokens":  self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "estimated_cost_usd": round(self.estimated_cost(), 6),
        }


# Interactive demo
bot = ProductionChatbot(
    system_prompt="You are a concise technical mentor. Answer clearly and include examples.",
    model="gpt-4o-mini"
)

questions = [
    "What is a vector database and why is it used in AI applications?",
    "How does it differ from a traditional SQL database?",
]

for q in questions:
    print(f"\nUser: {q}")
    print("Bot:", end=" ")
    bot.chat(q)

print(f"\nSession stats: {bot.stats()}")
```

---

## Edge Cases and Misconceptions

**"The `system` message controls the model completely."** The system message influences behavior significantly but is not enforced at a hardware level. Sufficiently creative prompting can sometimes override system instructions. For safety-critical applications, rely on content filters and output validation, not system prompts alone.

**"Conversation history must include every message."** You can and should trim history. Sending hundreds of old messages wastes money and consumes context window space. The model's attention has finite capacity; very long histories mean recent messages get less "attention budget."

**"temperature=0 is always deterministic."** Almost always, but not guaranteed. For truly deterministic output, set temperature=0 AND seed=some_integer. Even then, floating-point non-determinism across different hardware can cause minor variations.

**"max_tokens limits the response to that many tokens."** Correct, but it also means the model will truncate mid-sentence if it reaches the limit. For short structured outputs, set max_tokens tightly. For long prose, set it generously or omit it.

---

## Key Takeaways

- The Chat Completions API is stateless: you must send full conversation history with every request
- The three message roles (`system`, `user`, `assistant`) have distinct purposes; using them correctly is the foundation of good prompt design
- Conversation history grows every turn — production systems need trimming or summarization strategies
- Streaming dramatically improves perceived latency and is almost always worth implementing
- Temperature controls output diversity by scaling the logit distribution before sampling
- All API calls need retry logic with exponential backoff — transient failures are normal

---

## Further Reading

- [OpenAI Chat Completions API Reference](https://platform.openai.com/docs/api-reference/chat) — the authoritative API documentation
- [OpenAI Cookbook: Techniques to Improve Reliability](https://cookbook.openai.com/techniques_to_improve_reliability) — practical patterns for production
- [Andrej Karpathy: State of GPT](https://www.youtube.com/watch?v=bZQun8Y4L2A) — explains what LLMs can and cannot do (helps set realistic expectations)

---

**Next:** [Understanding Tokens and Costs](03-tokens-and-costs.md)
