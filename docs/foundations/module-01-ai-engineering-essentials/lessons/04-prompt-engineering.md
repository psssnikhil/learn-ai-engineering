---
title: Prompt Engineering Fundamentals
description: Master the art of crafting effective prompts to get better AI outputs
duration: 40 min
difficulty: beginner
has_code: true
module: module-01
youtube: 'https://www.youtube.com/watch?v=_ZvnD96BQyI'
objectives:
  - Write clear and specific prompts
  - Use few-shot learning effectively
  - Apply prompt engineering patterns
---
# Prompt Engineering Fundamentals

![Prompt Engineering](https://images.unsplash.com/photo-1676277791608-ac5c134b5ed5?w=800)

## Why Prompt Engineering Matters

Your prompt is your control interface to AI. A good prompt can make the difference between:
- Generic → Exceptional output
- Confused → Clear responses
- Slow → Fast results
- Expensive → Cost-effective

## The Anatomy of a Great Prompt

### Basic Structure

```
[ROLE] + [TASK] + [CONTEXT] + [FORMAT] + [CONSTRAINTS]
```

### Example Breakdown

**Bad Prompt:**
```
Tell me about AI.
```

**Good Prompt:**
```
[ROLE] You are an experienced AI engineer.

[TASK] Explain how transformers work

[CONTEXT] to someone with programming experience but no ML background.

[FORMAT] Use analogies and simple code examples.

[CONSTRAINTS] Keep it under 200 words.
```

## Prompt Engineering Patterns

### Pattern 1: Role-Playing

Give the AI a specific role:

```
You are a {ROLE}...
- Python expert
- Marketing strategist
- Technical writer
- Data scientist
```

**Example:**
```
You are a senior Python developer with 10 years of experience.
Review this code and suggest improvements focusing on:
- Performance
- Readability
- Best practices
```

### Pattern 2: Few-Shot Learning

Provide examples of what you want:

```
Convert customer feedback to sentiment scores (1-5):

Feedback: "Amazing product! Love it!"
Sentiment: 5

Feedback: "It's okay, nothing special."
Sentiment: 3

Feedback: "Terrible experience, would not recommend."
Sentiment: 1

Now analyze this:
Feedback: "Great features but buggy interface."
Sentiment: ?
```

### Pattern 3: Chain of Thought

Ask the AI to think step-by-step:

```
Problem: Calculate 15% tip on $47.50

Let's solve this step by step:
1. First, identify the amount: $47.50
2. Convert 15% to decimal: 0.15
3. Multiply: $47.50 × 0.15 = ?
4. Final answer with rounding
```

### Pattern 4: Output Formatting

Specify exactly how you want the output:

```
Analyze this text and return a JSON object:
{
  "sentiment": "positive|negative|neutral",
  "confidence": 0.0-1.0,
  "key_phrases": ["phrase1", "phrase2"],
  "summary": "one sentence"
}
```

### Pattern 5: Constraints

Set clear boundaries:

```
Write a product description with these constraints:
- Exactly 50 words
- Include keywords: "eco-friendly", "durable", "modern"
- Target audience: millennials
- Tone: casual but professional
- No emojis
```

## Advanced Techniques

### Temperature Control

```python
# Deterministic (temperature=0)
response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "List the planets"}],
    temperature=0  # Same output every time
)

# Creative (temperature=0.9)
response = openai.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Write a poem about AI"}],
    temperature=0.9  # More varied and creative
)
```

**When to use:**
- **Low (0-0.3)**: Facts, code, analysis, classification
- **Medium (0.5-0.7)**: General tasks, balanced creativity
- **High (0.8-1.0)**: Creative writing, brainstorming

### System vs User Messages

```python
messages=[
    {
        "role": "system",
        "content": "You are a helpful assistant that speaks like Shakespeare."
    },  # Sets behavior/personality
    {
        "role": "user",
        "content": "Tell me about computers."
    }  # The actual query
]
```

**System message:** Sets persistent behavior  
**User message:** The actual requests  

## Common Mistakes

### ❌ Mistake 1: Vague Prompts

**Bad:**
```
Write about dogs.
```

**Good:**
```
Write a 150-word article about the top 3 health benefits of owning a dog,
targeting busy professionals in their 30s. Include at least one scientific study reference.
```

### ❌ Mistake 2: No Examples

**Bad:**
```
Classify these as urgent or not urgent.
```

**Good:**
```
Classify emails as "urgent" or "not urgent":

"Server is down!" → urgent
"Next week's meeting agenda" → not urgent
"Password reset expiring in 1 hour" → urgent

Now classify: "Quarterly report draft ready for review"
```

### ❌ Mistake 3: Ignoring Context Length

**Bad:**
```
Analyze this 10,000 word document... [huge paste]
```

**Good:**
```
I'll share a document in chunks. After each chunk, summarize key points.
Then provide an overall summary at the end.

Chunk 1: [first 1000 words]
```

## Prompt Templates Library

### Code Review Template
```
Review this {LANGUAGE} code for:
1. Bugs
2. Performance issues
3. Security concerns
4. Best practices

Code:
{CODE}

Provide:
- Issues found (with line numbers)
- Severity (high/medium/low)
- Suggested fixes
```

### Content Creation Template
```
Create a {CONTENT_TYPE} about {TOPIC}

Audience: {TARGET_AUDIENCE}
Tone: {TONE}
Length: {WORD_COUNT} words
Key points to cover:
- {POINT_1}
- {POINT_2}
- {POINT_3}

Include: {REQUIREMENTS}
Avoid: {RESTRICTIONS}
```

### Data Analysis Template
```
Analyze this data:
{DATA}

Provide:
1. Key statistics (mean, median, outliers)
2. Trends or patterns
3. Anomalies
4. Recommendations

Format as: {JSON/Markdown/Text}
```

## Code Exercise

Build a prompt engineering toolkit with reusable templates!

---

## 📹 Recommended Videos

- [Prompt Engineering Full Course](https://www.youtube.com/watch?v=_ZvnD96BQyI) — freeCodeCamp comprehensive guide
- [ChatGPT Prompt Engineering for Developers](https://www.youtube.com/watch?v=H4YK_7MAckk) — DeepLearning.AI with Andrew Ng and Isa Fulford
- [Advanced Prompt Engineering](https://www.youtube.com/watch?v=T9aRN5JkmL8) — Chain-of-thought, few-shot, and more

---

## 📚 Additional Resources

### Articles & Blogs:
- [Anthropic's Prompt Engineering Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering) — Official Claude prompting best practices
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering) — Official OpenAI prompting guide
- [Prompt Engineering Guide](https://www.promptingguide.ai/) — Community-maintained comprehensive resource

### Papers:
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) — Original CoT paper by Wei et al.
- [Tree of Thoughts](https://arxiv.org/abs/2305.10601) — Advanced reasoning framework

---

## Key Takeaways

✅ Structure prompts with role, task, context, format, constraints  
✅ Use few-shot learning for complex tasks  
✅ Ask for step-by-step reasoning for better results  
✅ Control creativity with temperature  
✅ Provide examples of desired output  
✅ Be specific about format and length  
✅ Iterate and refine your prompts

## What's Next?

In the next lessons, we'll explore:
- Advanced prompt patterns (ReAct, Tree of Thoughts)
- Prompt injection and security
- Optimizing prompts for cost and speed
- Building dynamic prompt systems
