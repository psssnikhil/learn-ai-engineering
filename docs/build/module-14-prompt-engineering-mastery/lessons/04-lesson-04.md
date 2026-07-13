---
title: Structured Output and JSON Mode
description: >-
  Learn to get consistent, parseable JSON output from LLMs using structured
  output modes, Pydantic models, and output schemas
duration: 40 min
difficulty: intermediate
has_code: true
module: module-14
---
# Structured Output and JSON Mode

## Prerequisites

Before this lesson you should be comfortable with:

- **Prompt anatomy and system prompts** — Lessons 1 and 3
- **Python dictionaries and JSON** — parsing and serialization
- **Basic Pydantic models** — defining typed data classes (helpful but not required)

You do not need experience with API schema design. This lesson focuses on getting machine-readable output from LLMs.

---

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Explain why free-text LLM output breaks downstream pipelines | 8 min | Intermediate |
| Use OpenAI JSON mode and structured outputs with Pydantic | 12 min | Intermediate |
| Define complex schemas with enums, nested models, and Field descriptions | 10 min | Intermediate |
| Build a production extraction pipeline with retries and validation | 10 min | Intermediate |

---

## Intuition First: The Form vs. the Essay

Imagine you ask 100 people to describe their mood. Some write paragraphs, some say "good," some say "7/10 happy." You cannot build a dashboard from that.

Now give them a form: `{"sentiment": "positive|negative|neutral", "confidence": 0.0-1.0}`. Every response is parseable, validatable, and storable.

Free-text LLM responses are essays. Structured output is the form. Production systems — routing tickets, updating databases, triggering workflows — need forms. Regex on "The sentiment is positive with a confidence of 0.92" is fragile. A JSON object is not.

---

## Why Structured Output?

```python
# Without structured output (fragile)
response_text = "The sentiment is positive with a confidence of 0.92"
# How do you reliably extract "positive" and 0.92? Regex? String splitting?

# With structured output (reliable)
response_json = {"sentiment": "positive", "confidence": 0.92}
# Direct dictionary access, always works
```

Every point where you parse free text with regex is a future production incident. Structured output eliminates that entire class of bugs.

---

## OpenAI JSON Mode

JSON mode guarantees the model outputs valid JSON. You must still describe the schema in the prompt:

```python
from openai import OpenAI
import json

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "system",
            "content": (
                "You extract product information. Always respond in JSON "
                "with fields: name, price, category, in_stock (boolean)."
            ),
        },
        {
            "role": "user",
            "content": (
                "The new MacBook Pro 16-inch with M4 chip is available "
                "for $2,499 in the laptops section."
            ),
        },
    ],
)

data = json.loads(response.choices[0].message.content)
print(data)
# {"name": "MacBook Pro 16-inch M4", "price": 2499, "category": "laptops", "in_stock": true}
```

JSON mode ensures syntactic validity. It does **not** guarantee the right keys or types — the model might return `"price": "2499"` (string) instead of a number.

---

## Structured Outputs with Pydantic

OpenAI's structured outputs feature guarantees the response matches your Pydantic schema:

```python
from pydantic import BaseModel, Field
from openai import OpenAI

client = OpenAI()

class ProductInfo(BaseModel):
    name: str
    price: float
    category: str
    in_stock: bool
    features: list[str]

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Extract product information from the text."},
        {
            "role": "user",
            "content": (
                "The Sony WH-1000XM5 headphones ($348) are wireless "
                "noise-cancelling headphones with 30-hour battery life "
                "and multipoint connection. Currently available."
            ),
        },
    ],
    response_format=ProductInfo,
)

product = response.choices[0].message.parsed
print(product.name)       # "Sony WH-1000XM5"
print(product.price)      # 348.0
print(product.features)   # ["wireless", "noise-cancelling", ...]
```

The API constrains token generation to valid schema instances. Your application code receives typed Python objects — no parsing logic required.

---

## Complex Schemas

Use `Field(description=...)` to guide the model on what each field should contain:

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

code_to_review = """
def login(user, pwd):
    q = f"SELECT * FROM users WHERE name='{user}' AND pass='{pwd}'"
    return db.execute(q)
"""

response = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Review the provided code and identify issues."},
        {"role": "user", "content": code_to_review},
    ],
    response_format=CodeReview,
)

review = response.choices[0].message.parsed
for issue in review.issues:
    print(f"Line {issue.line} [{issue.severity.value}]: {issue.description}")
```

Enums constrain the model to valid categorical values. Field constraints (`ge=1, le=10`) enforce numeric bounds at generation time.

---

## Production Pipeline with Retries

```python
import logging
from typing import TypeVar
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

def safe_structured_call(
    messages: list[dict],
    schema: type[T],
    retries: int = 2,
    model: str = "gpt-4o-mini",
) -> T | None:
    """Call LLM with structured output, retry on failure."""
    for attempt in range(retries + 1):
        try:
            response = client.beta.chat.completions.parse(
                model=model,
                messages=messages,
                response_format=schema,
            )
            result = response.choices[0].message.parsed
            if result is not None:
                return result
            logger.warning(f"Attempt {attempt + 1}: parsed result was None")
        except ValidationError as e:
            logger.warning(f"Attempt {attempt + 1}: validation failed: {e}")
        except Exception as e:
            logger.error(f"Attempt {attempt + 1}: API error: {e}")
            if attempt == retries:
                raise
    return None

class SentimentResult(BaseModel):
    sentiment: str = Field(description="One of: positive, negative, neutral")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="Brief explanation")

result = safe_structured_call(
    messages=[
        {"role": "system", "content": "Classify sentiment of the text."},
        {"role": "user", "content": "Absolutely love this product!"},
    ],
    schema=SentimentResult,
)
if result:
    print(f"{result.sentiment} ({result.confidence:.0%})")
```

---

## Production Connection

Structured output is the bridge between LLM reasoning and application logic:

- **Version schemas alongside prompts** — when you add a field to `ProductInfo`, bump the schema version and migrate downstream consumers.
- **A/B test schema designs** — a flat schema vs. nested schema can affect extraction accuracy. Measure field-level precision, not just overall success rate.
- **Eval loops** — maintain a golden set with expected JSON outputs. Run field-by-field comparison (exact match for enums, fuzzy match for strings).
- **Failure recovery ladder**:
  1. Retry with structured outputs (same schema)
  2. Fall back to JSON mode + manual Pydantic validation
  3. Fall back to free-text + regex extraction
  4. Return a safe default and log for human review
- **Monitor parse failures** — track `parsed is None` rate as a first-class metric.

---

## Edge Cases & Common Misconceptions

**Misconception 1: JSON mode equals schema compliance.**
JSON mode only guarantees valid JSON syntax. Use structured outputs when you need guaranteed field names and types.

**Misconception 2: Complex schemas always work better.**
Deeply nested schemas with 20+ fields increase failure rates. Start minimal, add fields when eval proves they're needed.

**Misconception 3: Field descriptions are optional.**
Descriptions are prompt instructions for the model. `"price: float"` without description may extract `"$2,499"` as a string. Description: "Numeric price in USD, no currency symbol" fixes this.

**Misconception 4: Structured output eliminates hallucination.**
The model will confidently fill every field — even with fabricated data. Add confidence scores and source citations where accuracy matters.

---

## Worked Example: End-to-End Extraction Pipeline

Let's trace a support ticket through a complete structured extraction pipeline:

**Input**: "Customer John Smith (john@acme.com) reports that invoice #4521 for $1,200 was charged twice on March 15."

**Step 1 — Define schema**

```python
class SupportTicket(BaseModel):
    customer_name: str
    customer_email: str
    issue_type: str = Field(description="billing, technical, or general")
    invoice_id: str | None = None
    amount: float | None = None
    priority: str = Field(description="low, medium, or high")
```

**Step 2 — Extract**

```python
result = safe_structured_call(
    messages=[
        {"role": "system", "content": "Extract structured ticket data."},
        {"role": "user", "content": ticket_text},
    ],
    schema=SupportTicket,
)
# SupportTicket(customer_name='John Smith', customer_email='john@acme.com',
#               issue_type='billing', invoice_id='4521', amount=1200.0, priority='high')
```

**Step 3 — Route downstream**

```python
if result and result.issue_type == "billing" and result.amount and result.amount > 500:
    route_to = "billing_escalation"
elif result and result.issue_type == "technical":
    route_to = "tech_support"
else:
    route_to = "general_queue"
```

The extracted object feeds directly into your routing logic — no regex, no string parsing, no `if "billing" in response.text`.

---

## Key Takeaways

- Free-text LLM output requires fragile parsing; structured output gives you typed, validated objects.
- JSON mode guarantees valid JSON syntax; structured outputs guarantee schema compliance.
- Use Pydantic models with Field descriptions to guide the model on each field's meaning.
- Enums and numeric constraints (ge, le) enforce valid values at generation time.
- Build retry ladders: structured outputs → JSON mode → free-text → safe default.
- Version schemas, eval field-by-field, and monitor parse failure rates in production.
- Start with minimal schemas and add complexity only when eval data supports it.
- Structured output reduces format bugs but does not prevent hallucinated field values.

---

## Next Lesson

**[Lesson 5: Prompt Templates and Variables](05-lesson-05.md)** — Build reusable, testable prompt templates with dynamic variables, conditional logic, and version control.
