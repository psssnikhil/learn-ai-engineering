---
title: Agent Types & Patterns
description: >-
  Learn the major categories of AI agents including reactive, deliberative,
  tool-using, and autonomous agents, with practical examples
duration: 35 min
difficulty: advanced
has_code: false
objectives:
  - 'Classify agents by their architecture (reactive, deliberative, hybrid)'
  - Explain the tool-use agent pattern and implement a basic version
  - Describe the retrieval-augmented agent pattern
  - Compare conversational agents vs task-oriented agents
  - Design an agent for a given use case using the appropriate pattern
---
# Agent Types & Patterns

## What You'll Learn

By the end of this lesson, you'll understand:
- The spectrum of AI agent architectures
- Reactive vs. deliberative vs. hybrid agents
- Common agent patterns: tool-use, RAG, conversational, autonomous
- How to choose the right agent type for your use case
- Design trade-offs between agent patterns

**Time to Complete**: 35 minutes
**Difficulty**: Advanced

---

## The Agent Spectrum

Agents range from simple (reactive) to complex (fully autonomous). Understanding this spectrum helps you pick the right architecture.

```
Simple ◄──────────────────────────────────────► Complex

Reactive    Tool-Use    RAG Agent    Deliberative    Autonomous
(no state)  (single     (knowledge-  (plans ahead)   (self-directed
             loop)      augmented)                    goals)
```

---

## 1. Reactive Agents

Respond directly to input with no memory or planning. Essentially an LLM call with a good system prompt.

```python
class ReactiveAgent:
    def __init__(self, llm_client, system_prompt: str):
        self.client = llm_client
        self.system_prompt = system_prompt

    def respond(self, user_input: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_input}
            ]
        )
        return response.choices[0].message.content

# Example: A classification agent
classifier = ReactiveAgent(
    llm_client=client,
    system_prompt="Classify the following text as POSITIVE, NEGATIVE, or NEUTRAL. Return only the label."
)
```

**When to use**: Classification, formatting, simple Q&A, stateless transformations.

**Limitations**: No memory, no tool use, no multi-step reasoning.

---

## 2. Tool-Use Agents

Extend an LLM with the ability to call functions. The agent decides when and which tools to use.

```python
class ToolUseAgent:
    def __init__(self, llm_client, tools: list[dict]):
        self.client = llm_client
        self.tools = tools

    def run(self, user_input: str) -> str:
        messages = [{"role": "user", "content": user_input}]

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=self.tools,  # OpenAI function calling format
            tool_choice="auto"
        )

        message = response.choices[0].message

        # If the model wants to call a tool
        if message.tool_calls:
            for tool_call in message.tool_calls:
                result = self._execute_tool(
                    tool_call.function.name,
                    tool_call.function.arguments
                )
                messages.append(message)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })

            # Get the final response incorporating tool results
            final = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=self.tools
            )
            return final.choices[0].message.content

        return message.content

    def _execute_tool(self, name: str, args: str) -> str:
        import json
        parsed_args = json.loads(args)
        # Dispatch to the appropriate function
        return tool_registry[name](**parsed_args)
```

**When to use**: Tasks requiring external data (search, APIs, databases), calculations, system interactions.

**Limitations**: Single-turn tool use; complex tasks may need multiple rounds.

---

## 3. Retrieval-Augmented (RAG) Agents

Agents that search a knowledge base before answering. Combines retrieval with generation for grounded responses.

```python
class RAGAgent:
    def __init__(self, llm_client, vector_store, top_k: int = 5):
        self.client = llm_client
        self.vector_store = vector_store
        self.top_k = top_k

    def answer(self, question: str) -> dict:
        # Step 1: Retrieve relevant documents
        docs = self.vector_store.similarity_search(question, k=self.top_k)
        context = "

".join(doc.page_content for doc in docs)

        # Step 2: Generate answer with context
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": (
                    "Answer the user's question based on the provided context. "
                    "If the context doesn't contain the answer, say so. "
                    "Always cite which document(s) you used."
                )},
                {"role": "user", "content": (
                    f"Context:
{context}

Question: {question}"
                )}
            ]
        )

        return {
            "answer": response.choices[0].message.content,
            "sources": [doc.metadata for doc in docs]
        }
```

**When to use**: Customer support, documentation Q&A, internal knowledge bases, any scenario where answers must be grounded in specific data.

**Limitations**: Quality depends on retrieval; may miss relevant documents or retrieve irrelevant ones.

---

## 4. Conversational Agents

Maintain a conversation over multiple turns with memory and personality.

```python
class ConversationalAgent:
    def __init__(self, llm_client, persona: str):
        self.client = llm_client
        self.persona = persona
        self.history = []
        self.max_history = 50

    def chat(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})

        messages = [
            {"role": "system", "content": self.persona},
            *self.history[-self.max_history:]
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )

        assistant_message = response.choices[0].message.content
        self.history.append({"role": "assistant", "content": assistant_message})
        return assistant_message

# Example: A tutoring agent
tutor = ConversationalAgent(
    llm_client=client,
    persona=(
        "You are a patient AI tutor specializing in Python programming. "
        "Adapt your explanations to the student's level. "
        "Ask follow-up questions to check understanding. "
        "Use code examples to illustrate concepts."
    )
)
```

**When to use**: Chatbots, tutoring, therapy bots, interactive assistants, any long-running dialogue.

---

## 5. Deliberative (Planning) Agents

Create and execute plans for complex, multi-step tasks.

```python
class DeliberativeAgent:
    def __init__(self, llm_client, tools: dict):
        self.client = llm_client
        self.tools = tools

    def solve(self, task: str) -> dict:
        # Phase 1: Plan
        plan = self._create_plan(task)

        # Phase 2: Execute plan step by step
        results = []
        for step in plan:
            result = self._execute_step(step, results)
            results.append({"step": step, "result": result})

        # Phase 3: Synthesize final answer
        answer = self._synthesize(task, results)
        return {"plan": plan, "steps": results, "answer": answer}

    def _create_plan(self, task: str) -> list[str]:
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"Create a numbered list of steps to accomplish: {task}
Return ONLY the numbered list."
            }]
        )
        steps = response.choices[0].message.content.strip().split("
")
        return [s.lstrip("0123456789. ") for s in steps if s.strip()]

    def _execute_step(self, step: str, prior_results: list) -> str:
        context = "
".join(f"- {r['step']}: {r['result']}" for r in prior_results)
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"Previous results:
{context}

Now execute this step: {step}"
            }]
        )
        return response.choices[0].message.content

    def _synthesize(self, task: str, results: list) -> str:
        context = "
".join(f"Step: {r['step']}
Result: {r['result']}" for r in results)
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": f"Task: {task}

Results from each step:
{context}

Synthesize a complete answer."
            }]
        )
        return response.choices[0].message.content
```

**When to use**: Research tasks, report generation, multi-step data analysis, project planning.

---

## Choosing the Right Agent Type

| Use Case | Recommended Pattern | Why |
|----------|-------------------|-----|
| Text classification | Reactive | Stateless, single-step |
| Customer support | RAG + Conversational | Needs knowledge base + multi-turn |
| Code assistant | Tool-Use | Needs to run code, search docs |
| Research assistant | Deliberative | Multi-step, requires planning |
| Data analysis | Tool-Use + Deliberative | Needs tools + structured approach |
| Personal assistant | Conversational + Tool-Use | Persistent + capable |

---

## Resources

- **Anthropic's Agent Patterns Guide** -- Practical taxonomy of agent designs
- **Andrew Ng's Agentic Design Patterns** -- Four key patterns for AI agents
- **Lilian Weng's "LLM Powered Autonomous Agents"** -- Comprehensive survey of agent architectures

---

## Key Takeaways

1. **Reactive agents** are simplest -- use them when stateless processing is enough
2. **Tool-use agents** extend LLMs with real-world capabilities
3. **RAG agents** ground responses in specific knowledge for accuracy
4. **Deliberative agents** plan before acting for complex multi-step tasks
5. **Combine patterns** for powerful agents (e.g., RAG + tool-use + conversation)
