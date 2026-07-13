---
title: System Prompts and Role Design
description: >-
  Learn how to design effective system prompts that control LLM behavior,
  personality, and output format
duration: 35 min
difficulty: intermediate
has_code: true
module: module-14
---
# System Prompts and Role Design

## Prerequisites

Before this lesson you should be comfortable with:

- **Prompt anatomy** — the six components from Lesson 1
- **Few-shot and CoT basics** — when to embed examples vs. system rules (Lesson 2)
- **Chat API message roles** — system, user, and assistant messages

You do not need experience building chatbots. This lesson focuses on the system message as a control surface.

---

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Explain what a system prompt does and why it persists across turns | 8 min | Intermediate |
| Structure system prompts with identity, capabilities, rules, and format | 10 min | Intermediate |
| Apply role design patterns for expert, analyst, and tutor personas | 8 min | Intermediate |
| Build and test a versioned system prompt in Python | 9 min | Intermediate |

---

## Intuition First: The Employee Handbook

Think of the system prompt as an **employee handbook** handed to a new hire before they take their first customer call. It answers: Who are you? What can you do? What must you never do? How should you format your responses?

The user messages are the individual customer calls — each one different. But the handbook stays the same. Without it, the same "employee" (model) might be friendly on one call, terse on the next, and leak confidential info on the third.

In API terms, the system message is processed once at the start of the conversation and sets the behavioral frame for every subsequent turn. This makes it the single most powerful lever for controlling LLM output in production applications.

---

## What Is a System Prompt?

The system prompt sets the ground rules for the conversation. It defines who the AI is, what it can do, and how it should respond.

```python
from openai import OpenAI

client = OpenAI()

SYSTEM = """You are a senior Python developer who reviews code.

Rules:
- Be concise and direct
- Focus on bugs, security issues, and performance
- Suggest specific fixes with code examples
- Rate severity: critical, warning, or info
- If the code is good, say so briefly"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": """Review this:

def get_user(id):
    query = f'SELECT * FROM users WHERE id = {id}'
    return db.execute(query)""",
        },
    ],
)
print(response.choices[0].message.content)
```

The system prompt ensures every code review follows the same severity scale and output style — regardless of how the user phrases their request.

---

## System Prompt Structure

A well-structured system prompt has these sections:

```
1. Identity:    Who the AI is
2. Capabilities: What it can and cannot do
3. Rules:       Behavioral constraints
4. Output format: How to structure responses
5. Examples (optional): Show desired behavior
```

### Example: Customer Support Bot

```python
SUPPORT_SYSTEM = """You are a customer support agent for TechCo, a SaaS project management tool.

## Capabilities
- Answer questions about TechCo features and pricing
- Help troubleshoot common issues
- Guide users through setup and configuration
- Escalate complex issues to human support

## Rules
- Never share internal company information or roadmap details
- Never make promises about features or timelines
- If you do not know the answer, say so and offer to connect with a human agent
- Be friendly but professional
- Keep responses under 150 words unless the user asks for detail

## Output format
- Use bullet points for multi-step instructions
- Include relevant documentation links when available
- End with a follow-up question to ensure the issue is resolved
"""
```

Clear sections make prompts **maintainable**. When product adds a new feature, you update Capabilities — not a wall of unstructured text.

---

## Role Design Patterns

### The Expert Role

Constrains the AI to a specific domain and redirects off-topic requests:

```python
EXPERT_SYSTEM = """You are a database performance consultant with 15 years of experience.

You specialize in:
- Query optimization (PostgreSQL, MySQL)
- Index design strategies
- Database schema review
- Performance profiling

When the user asks about topics outside database performance,
politely redirect them to the relevant topic."""
```

### The Structured Analyst Role

Forces consistent output format every time:

```python
ANALYST_SYSTEM = """You are a code review analyst. For every code snippet, provide:

## Analysis
1. **Summary**: One sentence describing what the code does
2. **Issues**: Numbered list of problems found (or "None found")
3. **Suggestions**: Specific improvements with code examples
4. **Verdict**: APPROVE, REQUEST_CHANGES, or NEEDS_DISCUSSION

Always use this exact format. Never skip sections."""
```

### The Conversational Tutor Role

Shapes teaching behavior, not just content:

```python
TUTOR_SYSTEM = """You are a patient programming tutor for beginners.

Teaching approach:
- Explain concepts using simple analogies before technical details
- Ask the student questions to check understanding
- Never give full solutions immediately — guide with hints first
- Celebrate progress and normalize mistakes
- Use code examples with comments on every line
- If the student seems frustrated, acknowledge it and simplify"""
```

Each role pattern solves a different product need. Expert roles reduce hallucination scope. Analyst roles enable parsing. Tutor roles control interaction style.

---

## Runnable Example: Versioned System Prompt Manager

```python
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from openai import OpenAI

client = OpenAI()

@dataclass
class SystemPromptVersion:
    name: str
    version: str
    content: str
    created_at: str
    description: str = ""

class SystemPromptRegistry:
    def __init__(self, path: str = "./system_prompts.json"):
        self.path = Path(path)
        self._prompts: dict[str, list[SystemPromptVersion]] = {}
        if self.path.exists():
            self._load()

    def register(self, name: str, content: str, version: str, description: str = ""):
        entry = SystemPromptVersion(
            name=name,
            version=version,
            content=content,
            created_at=datetime.now(timezone.utc).isoformat(),
            description=description,
        )
        self._prompts.setdefault(name, []).append(entry)
        self._save()

    def get(self, name: str, version: str = "latest") -> str:
        versions = self._prompts.get(name, [])
        if not versions:
            raise KeyError(f"System prompt '{name}' not found")
        if version == "latest":
            return versions[-1].content
        for v in versions:
            if v.version == version:
                return v.content
        raise KeyError(f"Version '{version}' not found for '{name}'")

    def _save(self):
        data = {k: [asdict(v) for v in vals] for k, vals in self._prompts.items()}
        self.path.write_text(json.dumps(data, indent=2))

    def _load(self):
        data = json.loads(self.path.read_text())
        for name, versions in data.items():
            self._prompts[name] = [SystemPromptVersion(**v) for v in versions]

def chat(system_prompt: str, user_message: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content

# Register and use
registry = SystemPromptRegistry()
registry.register("code_reviewer", ANALYST_SYSTEM, "1.0", "Initial analyst role")
system = registry.get("code_reviewer")
# answer = chat(system, "Review: x = eval(input())")
```

---

## Common System Prompt Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| Too vague ("be helpful") | LLM has no clear direction | Specify exact behaviors |
| Too long (1000+ words) | Key instructions get lost | Keep under 500 words, prioritize |
| Contradictory rules | LLM picks one randomly | Review for conflicts |
| No output format | Inconsistent responses | Define structure explicitly |
| Negative-only rules | "Don't do X" is weaker than "Do Y" | Lead with positive instructions |

**Positive framing works better.** "Always cite the source document" beats "Never make up facts." The model follows affirmative instructions more reliably.

---

## Production Connection

System prompts are production-critical infrastructure:

- **Version and pin** — never deploy "latest" silently. Pin `code_reviewer_v2.1` in config and log it with every request.
- **A/B test role changes** — a stricter "never guess" rule might reduce hallucinations but increase "I don't know" responses. Measure both accuracy and user satisfaction.
- **Eval loops** — build adversarial test cases: users asking off-topic questions, requesting forbidden actions, or sending prompt injections. Run them against every system prompt change.
- **Failure recovery** — if the model violates format rules, append a correction to the next turn: "Your previous response skipped the Verdict section. Reformat using the required template."
- **Token budget** — system prompts are sent on every turn. A 2,000-token system prompt on 100K daily requests is expensive. Audit length quarterly.

---

## Edge Cases & Common Misconceptions

**Misconception 1: System prompts are unbreakable.**
Determined prompt injection can override system instructions. Combine system prompts with input validation and output guardrails (Lesson 8).

**Misconception 2: One system prompt fits all users.**
Enterprise products often need role-specific system prompts — admin vs. end-user, internal vs. external. Template the system prompt with user context.

**Misconception 3: Put everything in the system prompt.**
Long documents, RAG context, and conversation-specific data belong in user messages. The system prompt carries **rules**, not **data**.

**Misconception 4: Changing the system prompt is low-risk.**
It affects every user immediately. Treat system prompt changes like schema migrations: test, stage, canary, monitor.

---

## Key Takeaways

- The system prompt is the persistent behavioral contract; user messages carry variable task data.
- Structure system prompts into identity, capabilities, rules, and output format for maintainability.
- Role design patterns (expert, analyst, tutor) solve different product requirements.
- Keep system prompts under 500 words; prioritize the most important rules first.
- Use positive framing ("always do X") rather than negative-only rules ("never do Y").
- Version system prompts, pin versions in production config, and log version with every LLM call.
- A/B test and eval adversarial inputs before deploying system prompt changes.
- System prompts reduce but do not eliminate prompt injection — layer defenses.

---

## Next Lesson

**[Lesson 4: Structured Output and JSON Mode](04-lesson-04.md)** — Learn to get consistent, parseable JSON output from LLMs using structured output modes and Pydantic schemas.
