---
title: Structured Output and JSON Mode
description: >-
  Learn to get consistent, parseable JSON output from LLMs using structured
  output modes, Pydantic models, and output schemas
duration: 40 min
difficulty: intermediate
has_code: false
module: module-14
---
# Structured Output and JSON Mode

## Learning Objectives

By the end of this lesson, you will be able to:
- Use OpenAI's JSON mode and structured outputs
- Define output schemas with Pydantic models
- Handle parsing errors and validation gracefully
- Build reliable data extraction pipelines with LLMs

---

## Why Structured Output?

Free-text LLM responses are hard to use programmatically. Structured output guarantees your code can parse the result:

```python
# Without structured output (fragile)
response_text = "The sentiment is positive with a confidence of 0.92"
# How do you reliably extract "positive" and 0.92? Regex? String splitting?

# With structured output (reliable)
response_json = {"sentiment": "positive", "confidence": 0.92}
# Direct dictionary access, always works
```

---

## OpenAI JSON Mode

```python
from openai import OpenAI
import json

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "system",
            "content": "You extract product information. Always respond in JSON with fields: name, price, category, in_stock (boolean)."
        },
        {
            "role": "user",
            "content": "The new MacBook Pro 16-inch with M4 chip is available for $2,499 in the laptops section."
        }
    ]
)

data = json.loads(response.choices[0].message.content)
print(data)
# {"name": "MacBook Pro 16-inch M4", "price": 2499, "category": "laptops", "in_stock": true}
```

---

## Structured Outputs with Pydantic

OpenAI's structured outputs feature guarantees the response matches your Pydantic schema:

```python
from pydantic import BaseModel
from openai import OpenAI

client = OpenAI()

class ProductInfo(BaseModel):
    name: str
    price: float
    category: str
    in_stock: bool
    features: list[str]

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Extract product information from the text."},
        {"role": "user", "content": "The Sony WH-1000XM5 headphones ($348) are wireless noise-cancelling headphones with 30-hour battery life and multipoint connection. Currently available."}
    ],
    response_format=ProductInfo,
)

product = response.choices[0].message.parsed
print(product.name)       # "Sony WH-1000XM5"
print(product.price)      # 348.0
print(product.features)   # ["wireless", "noise-cancelling", "30-hour battery", "multipoint connection"]
```

---

## Complex Schemas

```python
from pydantic import BaseModel, Field
from enum import Enum

class Severity(str, Enum):
    critical = "critical"
    warning = "warning"
    info = "info"

class CodeIssue(BaseModel):
    line: int = Field(description="Line number where the issue occurs")
    severity: Severity
    description: str = Field(description="Brief description of the issue")
    fix: str = Field(description="Suggested code fix")

class CodeReview(BaseModel):
    summary: str = Field(description="One-sentence summary of the code")
    issues: list[CodeIssue]
    overall_quality: int = Field(ge=1, le=10, description="Quality score 1-10")

response = client.beta.chat.completions.parse(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "Review the provided code and identify issues."},
        {"role": "user", "content": code_to_review}
    ],
    response_format=CodeReview,
)

review = response.choices[0].message.parsed
for issue in review.issues:
    print(f"Line {issue.line} [{issue.severity.value}]: {issue.description}")
```

---

## Handling Failures

```python
def safe_structured_call(messages, schema, retries=2):
    """Call LLM with structured output, retry on failure."""
    for attempt in range(retries + 1):
        try:
            response = client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=messages,
                response_format=schema,
            )
            result = response.choices[0].message.parsed
            if result is not None:
                return result
        except Exception as e:
            if attempt == retries:
                raise
            print(f"Attempt {attempt + 1} failed: {e}")
    return None
```

---

## Key Takeaways

- JSON mode forces valid JSON output; structured outputs guarantee schema compliance
- Pydantic models define exact types, required fields, and validation rules
- Use `Field(description=...)` to guide the LLM on what each field should contain
- Structured outputs eliminate fragile parsing logic from your application code
- Always add retry logic for production applications

## Resources

- [OpenAI: Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs) -- Official documentation
- [YouTube: Structured Outputs Deep Dive](https://www.youtube.com/watch?v=B-N1JqomZZc) -- Practical examples with Pydantic
- [Pydantic Documentation](https://docs.pydantic.dev/) -- Schema definition reference
- [Instructor Library](https://github.com/jxnl/instructor) -- Popular library for structured LLM outputs

---

Next: Prompt Templates and Variables
