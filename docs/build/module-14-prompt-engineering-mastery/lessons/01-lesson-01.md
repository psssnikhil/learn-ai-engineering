---
title: Foundations of Prompt Engineering
description: >-
  Learn the core principles of effective prompt engineering, including clarity,
  specificity, and structured instructions
duration: 40 min
difficulty: intermediate
has_code: false
module: module-14
---
# Foundations of Prompt Engineering

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand what prompt engineering is and why it matters | 40 min | Intermediate |
| Learn the anatomy of an effective prompt | | |
| Master the core principles: clarity, specificity, structure | | |
| Practice writing prompts for different use cases | | |

---

## What is Prompt Engineering?

**Prompt engineering** is the practice of designing inputs to language models that reliably produce high-quality, accurate outputs. It sits at the intersection of communication, programming, and understanding how LLMs process language.

### Why It Matters

The same model can produce wildly different outputs depending on how you ask:

```python
# Vague prompt - unpredictable output
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "Tell me about Python"}]
)
# Could be about the snake, the programming language, or Monty Python!

# Specific prompt - targeted output
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": 
        "Explain 3 key advantages of Python for data science, "
        "with a brief code example for each. Target audience: "
        "developers switching from Java."}]
)
# Focused, actionable, audience-appropriate response
```

---

## The Anatomy of a Great Prompt

Every effective prompt contains some or all of these components:

```
┌─────────────────────────────────────────────┐
│  1. ROLE / PERSONA                          │
│     Who should the model act as?            │
│                                             │
│  2. CONTEXT                                 │
│     Background information needed           │
│                                             │
│  3. TASK / INSTRUCTION                      │
│     What exactly should the model do?       │
│                                             │
│  4. FORMAT / STRUCTURE                      │
│     How should the output look?             │
│                                             │
│  5. CONSTRAINTS                             │
│     Boundaries, limits, things to avoid     │
│                                             │
│  6. EXAMPLES (optional)                     │
│     Show what good output looks like        │
└─────────────────────────────────────────────┘
```

### Example: All Components Together

```python
prompt = """
You are a senior Python developer who specializes in code review.  # ROLE

I have a Flask API endpoint that handles user registration.         # CONTEXT
The code works but I'm concerned about security and best practices.

Review this code and provide:                                       # TASK
1. Security vulnerabilities found
2. Performance improvements
3. A refactored version with fixes applied

Format your response as:                                            # FORMAT
- Use markdown headers for each section
- Include severity ratings (Critical/High/Medium/Low)
- Add inline code comments in the refactored version

Constraints:                                                        # CONSTRAINTS
- Keep the same API contract (don't change endpoint paths)
- Use only standard library + Flask (no new dependencies)
- Assume Python 3.11+

Here is the code:
{code}
"""
```

---

## Core Principle 1: Be Specific

The biggest mistake in prompt engineering is being too vague.

### Bad vs Good

| Bad Prompt | Good Prompt |
|-----------|------------|
| "Summarize this" | "Summarize this article in 3 bullet points, each under 20 words, focusing on the business impact" |
| "Write code" | "Write a Python function that validates email addresses using regex, returns True/False, and handles edge cases like '+' aliases" |
| "Explain AI" | "Explain how transformer attention mechanisms work to a computer science undergrad who understands matrix multiplication" |

---

## Core Principle 2: Provide Structure

LLMs follow structure remarkably well. Use it to your advantage.

```python
# Using XML-style tags for clear separation
prompt = """
<context>
You are helping a user debug a React application.
The app uses Next.js 15 with the App Router.
</context>

<error_message>
TypeError: Cannot read properties of undefined (reading 'map')
at ProductList (./components/ProductList.tsx:12:24)
</error_message>

<task>
1. Identify the root cause of this error
2. Explain why it happens in simple terms
3. Provide the fix with before/after code
</task>

<format>
Use markdown. Start with a one-line diagnosis, then details.
</format>
"""
```

---

## Core Principle 3: Use the System Message

Most APIs support a system message that sets persistent behavior:

```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {
            "role": "system",
            "content": (
                "You are a database expert. Always suggest "
                "the most performant query. When writing SQL, "
                "include EXPLAIN ANALYZE output estimates. "
                "If the user's approach has a better alternative, "
                "mention it proactively."
            )
        },
        {
            "role": "user",
            "content": "How do I find duplicate emails in a users table with 10M rows?"
        }
    ]
)
```

### System Message Best Practices

| Do | Don't |
|----|-------|
| Define the role and expertise | Leave system message empty |
| Set output format preferences | Put conversation history in system |
| Specify constraints and boundaries | Make it extremely long (wastes tokens) |
| Include domain-specific instructions | Contradict the system message in user messages |

---

## Core Principle 4: Iterate and Refine

Prompt engineering is iterative. Start simple, then refine based on output quality.

```
Iteration 1: "Translate this to Spanish"
  Problem: Too formal for a casual chat app

Iteration 2: "Translate this to casual Latin American Spanish"
  Problem: Sometimes uses slang that's region-specific

Iteration 3: "Translate this to casual Spanish suitable for a
  general Latin American audience. Avoid country-specific slang.
  Keep the tone friendly and conversational."
  Result: Consistent, appropriate translations
```

---

## Practice Exercise

Write prompts for each of these scenarios. Aim to include role, context, task, format, and constraints:

1. **Code Review**: You want the model to review a pull request for a Node.js microservice
2. **Data Analysis**: You have a CSV of sales data and want insights
3. **Content Writing**: You need a technical blog post about WebSockets
4. **Debugging**: You have a Python script that runs slowly and need optimization advice

```python
# Template to get started
def create_prompt(role, context, task, format_spec, constraints):
    return f"""
Role: {role}
Context: {context}
Task: {task}
Output Format: {format_spec}
Constraints: {constraints}
"""

# Try it:
review_prompt = create_prompt(
    role="Senior Node.js developer and security expert",
    context="Reviewing a PR for an authentication microservice using Express.js",
    task="Identify bugs, security issues, and suggest improvements",
    format_spec="Markdown table with columns: Issue, Severity, Line, Fix",
    constraints="Focus on OWASP Top 10 vulnerabilities. Max 10 findings."
)
```

---

## Key Takeaways

- Specificity beats brevity: the more precise your prompt, the better the output
- Structure your prompts with clear sections (role, context, task, format, constraints)
- Use the system message for persistent behavior settings
- Iterate: treat prompts like code - test, evaluate, refine
- Different tasks need different prompt strategies (covered in upcoming lessons)

---

## Next Lesson

**Lesson 2: Few-Shot and Chain-of-Thought Prompting** - Learn advanced techniques that dramatically improve model reasoning and output quality.
