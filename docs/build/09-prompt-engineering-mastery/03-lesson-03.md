---
title: System Prompts and Role Design
description: >-
  Learn how to design effective system prompts that control LLM behavior,
  personality, and output format
duration: 35 min
difficulty: intermediate
has_code: false
---
# System Prompts and Role Design

## Learning Objectives

By the end of this lesson, you will be able to:
- Write system prompts that reliably control LLM behavior
- Design role-based prompts for different application needs
- Structure system prompts for consistency and maintainability
- Avoid common pitfalls in system prompt design

---

## What is a System Prompt?

The system prompt sets the ground rules for the conversation. It defines who the AI is, what it can do, and how it should respond.

```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": """You are a senior Python developer who reviews code.
            
Rules:
- Be concise and direct
- Focus on bugs, security issues, and performance
- Suggest specific fixes with code examples
- Rate severity: critical, warning, or info
- If the code is good, say so briefly"""
        },
        {
            "role": "user",
            "content": "Review this:

def get_user(id):
    query = f'SELECT * FROM users WHERE id = {id}'
    return db.execute(query)"
        }
    ]
)
```

---

## System Prompt Structure

A well-structured system prompt has these sections:

```
1. Identity: Who the AI is
2. Capabilities: What it can and cannot do
3. Rules: Behavioral constraints
4. Output format: How to structure responses
5. Examples (optional): Show desired behavior
```

### Example: Customer Support Bot

```python
SYSTEM_PROMPT = """You are a customer support agent for TechCo, a SaaS project management tool.

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

---

## Role Design Patterns

### The Expert Role

```python
# Constrain the AI to a specific domain expertise
expert_prompt = """You are a database performance consultant with 15 years of experience.

You specialize in:
- Query optimization (PostgreSQL, MySQL)
- Index design strategies
- Database schema review
- Performance profiling

When the user asks about topics outside database performance, 
politely redirect them to the relevant topic."""
```

### The Structured Analyst Role

```python
# Force consistent output format
analyst_prompt = """You are a code review analyst. For every code snippet, provide:

## Analysis
1. **Summary**: One sentence describing what the code does
2. **Issues**: Numbered list of problems found (or "None found")
3. **Suggestions**: Specific improvements with code examples
4. **Verdict**: APPROVE, REQUEST_CHANGES, or NEEDS_DISCUSSION

Always use this exact format. Never skip sections."""
```

### The Conversational Tutor Role

```python
tutor_prompt = """You are a patient programming tutor for beginners.

Teaching approach:
- Explain concepts using simple analogies before technical details
- Ask the student questions to check understanding
- Never give full solutions immediately -- guide with hints first
- Celebrate progress and normalize mistakes
- Use code examples with comments on every line
- If the student seems frustrated, acknowledge it and simplify"""
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

---

## Key Takeaways

- System prompts are the most powerful tool for controlling LLM behavior
- Structure prompts with clear sections: identity, capabilities, rules, output format
- Be specific and positive -- "always do X" works better than "never do Y"
- Keep system prompts under 500 words to avoid instruction dilution
- Test with edge cases to verify the prompt handles unexpected inputs

## Resources

- [OpenAI: Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) -- Official best practices
- [YouTube: System Prompts That Work](https://www.youtube.com/watch?v=ahnGLM-RC1Y) -- Practical examples and patterns
- [Anthropic: Prompt Design Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering) -- Claude-specific prompting techniques

---

Next: Structured Output and JSON Mode
