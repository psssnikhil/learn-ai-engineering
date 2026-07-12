---
title: 'Project 7: LLM-Powered Data Extraction Pipeline'
description: >-
  Build a pipeline that extracts structured data from unstructured documents
  like invoices, resumes, and contracts using LLMs
duration: 180 min
difficulty: advanced
has_code: true
module: module-17
---
# Project 7: LLM-Powered Data Extraction Pipeline

## Project Overview

Build a data extraction system that takes unstructured documents (invoices, resumes, articles) and extracts structured data using LLMs with schema validation. This applies structured outputs, prompt engineering, and data pipeline skills.

**What you will build:**
- Schema-driven extraction using Pydantic models
- Multi-document type support (invoice, resume, article)
- Validation and confidence scoring
- Batch processing pipeline

**Estimated time:** 3-5 hours

---

## Implementation

```python
# extractor.py
from pydantic import BaseModel, Field
from openai import OpenAI
from pathlib import Path

client = OpenAI()

# Define extraction schemas
class InvoiceData(BaseModel):
    vendor_name: str
    invoice_number: str
    date: str
    line_items: list[dict] = Field(description="List of {description, quantity, unit_price, total}")
    subtotal: float
    tax: float
    total: float
    currency: str = "USD"

class ResumeData(BaseModel):
    name: str
    email: str = ""
    phone: str = ""
    summary: str = Field(description="Professional summary in 1-2 sentences")
    skills: list[str]
    experience: list[dict] = Field(description="List of {company, title, duration, highlights}")
    education: list[dict] = Field(description="List of {school, degree, year}")

class ArticleData(BaseModel):
    title: str
    author: str = ""
    published_date: str = ""
    summary: str = Field(description="3-sentence summary")
    key_topics: list[str]
    sentiment: str = Field(description="positive, negative, or neutral")

# Document type to schema mapping
SCHEMAS = {
    "invoice": InvoiceData,
    "resume": ResumeData,
    "article": ArticleData,
}

def extract_data(text: str, doc_type: str) -> dict:
    """Extract structured data from unstructured text."""
    schema = SCHEMAS.get(doc_type)
    if not schema:
        raise ValueError(f"Unknown document type: {doc_type}")

    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": f"Extract structured data from this {doc_type}. "
                           f"If a field cannot be determined, use empty string or 0."
            },
            {"role": "user", "content": text}
        ],
        response_format=schema,
    )

    result = response.choices[0].message.parsed
    return result.model_dump()

def batch_extract(files: list[tuple[str, str]]) -> list[dict]:
    """Extract data from multiple files. Each item: (filepath, doc_type)."""
    results = []
    for filepath, doc_type in files:
        text = Path(filepath).read_text()
        try:
            data = extract_data(text, doc_type)
            results.append({"file": filepath, "status": "success", "data": data})
        except Exception as e:
            results.append({"file": filepath, "status": "error", "error": str(e)})
    return results
```

---

## Extensions and Challenges

- **Add OCR**: Use `pytesseract` to extract text from scanned PDFs
- **Confidence scoring**: Ask the LLM to rate confidence for each extracted field
- **Custom schemas**: Let users define extraction schemas via JSON config
- **Database storage**: Save extracted data to SQLite or PostgreSQL
- **Validation rules**: Add business rules (e.g., invoice total must equal sum of line items)

## Resources

- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs) -- Schema-enforced extraction
- [YouTube: Document AI](https://www.youtube.com/watch?v=xEPX9Hkm5CQ) -- Building document processing systems
- [Instructor Library](https://github.com/jxnl/instructor) -- Structured extraction from LLMs

---

Next: Project 8 -- Chatbot with Long-Term Memory
