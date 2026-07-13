---
title: 'Project 6: LLM-Powered Data Extraction Pipeline'
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

Build a production data extraction system that takes unstructured documents — invoices, resumes, contracts, articles — and returns validated structured data. This is one of the highest-value enterprise AI use cases: turning messy PDFs and emails into database-ready records.

Unlike a chatbot that answers questions, this pipeline is **deterministic in output shape** — every document type maps to a Pydantic schema, and every field is validated before storage.

**Time estimate**: 12-16 hours
**Skills used**: Structured Outputs, Pydantic, Prompt Engineering, Batch Processing, OCR

---

## Prerequisites

| Module | Topics Used |
|--------|------------|
| **Module 2: LLM Fundamentals** | Chat completions, system prompts |
| **Module 7: Prompt Engineering** | Schema-guided extraction prompts |
| **Module 9: Structured Output** | Pydantic models, JSON schema enforcement |
| **Module 13: LLMOps** | Batch jobs, error handling, logging |

**Environment setup:**

```bash
pip install openai pydantic python-dotenv fastapi uvicorn pypdf pytest
# Optional for scanned documents:
pip install pytesseract pdf2image pillow
```

---

## What You'll Build

### Acceptance Criteria Checklist

- [ ] Pydantic schemas for at least 3 document types (invoice, resume, article)
- [ ] LLM extraction using OpenAI structured outputs (`response_format`)
- [ ] Post-extraction validation (business rules, field completeness)
- [ ] Confidence scoring per extracted field
- [ ] Batch processing pipeline for a folder of documents
- [ ] FastAPI endpoints: `POST /extract`, `POST /extract/batch`, `GET /schemas`
- [ ] Results stored as JSON with extraction metadata (model, timestamp, confidence)
- [ ] Evaluation script measuring field-level accuracy on 20+ labeled documents
- [ ] Error handling for unreadable files, unknown document types, and API failures

---

## Architecture

```
[Document Input]
    |  PDF / TXT / MD files
    v
[Ingestion Layer]
    |-- Text extraction (PyPDF / plain text)
    |-- OCR fallback (pytesseract, optional)
    |-- Document type detection (filename or classifier)
    v
[Extraction Layer]
    |-- Select Pydantic schema by doc type
    |-- Build extraction prompt with schema hints
    |-- LLM call with structured output enforcement
    |-- Parse response into Pydantic model
    v
[Validation Layer]
    |-- Schema validation (Pydantic)
    |-- Business rules (totals match, dates valid)
    |-- Confidence scoring per field
    |-- Flag low-confidence fields for human review
    v
[Output Layer]
    |-- JSON results per document
    |-- Batch summary report
    |-- SQLite or JSONL persistence
    v
[FastAPI Server]
    |-- POST /extract       (single document)
    |-- POST /extract/batch (folder upload)
    |-- GET  /schemas       (available schemas)
    |-- GET  /results/{id}  (retrieved extraction)
```

---

## Step 1: Define Extraction Schemas

Start with strict Pydantic models. The schema is your contract with downstream systems.

```python
# src/schemas.py
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from datetime import date

class LineItem(BaseModel):
    description: str
    quantity: float = 1.0
    unit_price: float
    total: float

    @field_validator("total")
    @classmethod
    def total_matches(cls, v, info):
        data = info.data
        if "quantity" in data and "unit_price" in data:
            expected = round(data["quantity"] * data["unit_price"], 2)
            if abs(v - expected) > 0.01:
                raise ValueError(f"Line item total {v} != {expected}")
        return v

class InvoiceData(BaseModel):
    vendor_name: str
    invoice_number: str
    date: str
    line_items: list[LineItem]
    subtotal: float
    tax: float
    total: float
    currency: str = "USD"

class ExperienceEntry(BaseModel):
    company: str
    title: str
    duration: str
    highlights: list[str] = Field(default_factory=list)

class ResumeData(BaseModel):
    name: str
    email: str = ""
    phone: str = ""
    summary: str = Field(description="Professional summary in 1-2 sentences")
    skills: list[str]
    experience: list[ExperienceEntry] = Field(default_factory=list)
    education: list[dict] = Field(default_factory=list)

class ArticleData(BaseModel):
    title: str
    author: str = ""
    published_date: str = ""
    summary: str = Field(description="3-sentence summary")
    key_topics: list[str]
    sentiment: str = Field(description="positive, negative, or neutral")

SCHEMAS: dict[str, type[BaseModel]] = {
    "invoice": InvoiceData,
    "resume": ResumeData,
    "article": ArticleData,
}

def get_schema_info() -> dict:
    return {
        name: schema.model_json_schema()
        for name, schema in SCHEMAS.items()
    }
```

---

## Step 2: Build the Extraction Engine

Core extraction with structured outputs, validation, and confidence scoring.

```python
# src/extractor.py
import json
import time
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from pydantic import ValidationError
from src.schemas import SCHEMAS

client = OpenAI()

EXTRACTION_PROMPTS = {
    "invoice": (
        "Extract all invoice fields from the document. "
        "For line items, include description, quantity, unit_price, and total. "
        "If a field is missing, use empty string for text and 0 for numbers."
    ),
    "resume": (
        "Extract structured resume data. Parse experience into company, title, duration, and highlights. "
        "Extract all skills mentioned, including those in job descriptions."
    ),
    "article": (
        "Extract article metadata and content summary. "
        "Identify 3-5 key topics and classify sentiment as positive, negative, or neutral."
    ),
}

class ExtractionResult:
    def __init__(self, file: str, doc_type: str, status: str, data: dict = None,
                 errors: list = None, confidence: dict = None, latency_ms: float = 0):
        self.file = file
        self.doc_type = doc_type
        self.status = status
        self.data = data or {}
        self.errors = errors or []
        self.confidence = confidence or {}
        self.latency_ms = latency_ms
        self.timestamp = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "doc_type": self.doc_type,
            "status": self.status,
            "data": self.data,
            "errors": self.errors,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
        }

def extract_text(filepath: str) -> str:
    """Extract text from a file. Supports .txt, .md, and .pdf."""
    path = Path(filepath)
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")

def score_confidence(data: dict, doc_type: str) -> dict:
    """Score field completeness as a proxy for confidence."""
    scores = {}
    for field, value in data.items():
        if isinstance(value, str):
            scores[field] = 1.0 if value.strip() else 0.0
        elif isinstance(value, (int, float)):
            scores[field] = 1.0 if value != 0 else 0.3
        elif isinstance(value, list):
            scores[field] = 1.0 if len(value) > 0 else 0.0
        else:
            scores[field] = 0.5
    return scores

def extract_data(text: str, doc_type: str) -> ExtractionResult:
    """Extract structured data from unstructured text."""
    schema = SCHEMAS.get(doc_type)
    if not schema:
        return ExtractionResult("", doc_type, "error", errors=[f"Unknown doc type: {doc_type}"])

    start = time.time()
    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": EXTRACTION_PROMPTS[doc_type]},
                {"role": "user", "content": f"Extract data from this document:\n\n{text[:8000]}"},
            ],
            response_format=schema,
        )
        parsed = response.choices[0].message.parsed
        data = parsed.model_dump()
        confidence = score_confidence(data, doc_type)
        latency = (time.time() - start) * 1000
        return ExtractionResult("", doc_type, "success", data=data, confidence=confidence, latency_ms=latency)
    except ValidationError as e:
        return ExtractionResult("", doc_type, "validation_error", errors=[str(e)])
    except Exception as e:
        return ExtractionResult("", doc_type, "error", errors=[str(e)])

def extract_file(filepath: str, doc_type: str) -> ExtractionResult:
    """Extract data from a file on disk."""
    try:
        text = extract_text(filepath)
        if not text.strip():
            return ExtractionResult(filepath, doc_type, "error", errors=["No text extracted from file"])
        result = extract_data(text, doc_type)
        result.file = filepath
        return result
    except FileNotFoundError:
        return ExtractionResult(filepath, doc_type, "error", errors=[f"File not found: {filepath}"])
    except Exception as e:
        return ExtractionResult(filepath, doc_type, "error", errors=[str(e)])

def batch_extract(files: list[tuple[str, str]]) -> list[dict]:
    """Extract data from multiple files. Each item: (filepath, doc_type)."""
    results = []
    for filepath, doc_type in files:
        result = extract_file(filepath, doc_type)
        results.append(result.to_dict())
    return results

def validate_business_rules(data: dict, doc_type: str) -> list[str]:
    """Apply domain-specific validation beyond Pydantic."""
    errors = []
    if doc_type == "invoice":
        line_total = sum(item.get("total", 0) for item in data.get("line_items", []))
        subtotal = data.get("subtotal", 0)
        if abs(line_total - subtotal) > 0.05:
            errors.append(f"Line items sum ({line_total}) != subtotal ({subtotal})")
        expected_total = subtotal + data.get("tax", 0)
        if abs(expected_total - data.get("total", 0)) > 0.05:
            errors.append(f"Subtotal + tax ({expected_total}) != total ({data.get('total')})")
    return errors
```

---

## Step 3: FastAPI Server and Batch Pipeline

```python
# src/api.py
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from src.extractor import extract_data, extract_file, batch_extract, validate_business_rules
from src.schemas import get_schema_info

app = FastAPI(title="Data Extraction Pipeline")

class ExtractRequest(BaseModel):
    text: str
    doc_type: str

class BatchFileRequest(BaseModel):
    files: list[dict]  # [{"path": "...", "doc_type": "invoice"}]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/schemas")
def list_schemas():
    return get_schema_info()

@app.post("/extract")
def extract_single(req: ExtractRequest):
    if req.doc_type not in ("invoice", "resume", "article"):
        raise HTTPException(400, f"Unknown doc_type: {req.doc_type}")
    result = extract_data(req.text, req.doc_type)
    output = result.to_dict()
    if result.status == "success":
        rule_errors = validate_business_rules(result.data, req.doc_type)
        if rule_errors:
            output["validation_warnings"] = rule_errors
    return output

@app.post("/extract/file")
async def extract_upload(file: UploadFile = File(...), doc_type: str = "invoice"):
    content = await file.read()
    text = content.decode("utf-8")
    result = extract_data(text, doc_type)
    return result.to_dict()

@app.post("/extract/batch")
def extract_batch(req: BatchFileRequest):
    file_tuples = [(f["path"], f["doc_type"]) for f in req.files]
    results = batch_extract(file_tuples)
    summary = {
        "total": len(results),
        "success": sum(1 for r in results if r["status"] == "success"),
        "errors": sum(1 for r in results if r["status"] == "error"),
        "avg_latency_ms": sum(r.get("latency_ms", 0) for r in results) / max(len(results), 1),
    }
    return {"summary": summary, "results": results}
```

Run the server:

```bash
uvicorn src.api:app --reload --port 8000
```

Test with curl:

```bash
curl -X POST http://localhost:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"doc_type": "invoice", "text": "INVOICE\nVendor: Acme Corp\nInvoice #: INV-2024-001\nDate: 2024-03-15\nItem: Widget x10 @ $5.00 = $50.00\nSubtotal: $50.00\nTax: $4.00\nTotal: $54.00"}'
```

---

## Step 4: Evaluation Suite

Measure extraction accuracy against labeled ground truth.

```python
# evaluation/extraction_eval.py
import json
from pathlib import Path
from src.extractor import extract_file

# Ground truth: {"file": "...", "doc_type": "...", "expected": {...}}
def load_test_set(path: str = "evaluation/labeled_docs.json") -> list[dict]:
    return json.loads(Path(path).read_text())

def field_accuracy(expected: dict, actual: dict) -> dict:
    """Compare field-by-field accuracy."""
    results = {}
    for key, expected_val in expected.items():
        actual_val = actual.get(key)
        if isinstance(expected_val, str):
            results[key] = 1.0 if str(actual_val).strip().lower() == expected_val.strip().lower() else 0.0
        elif isinstance(expected_val, (int, float)):
            results[key] = 1.0 if actual_val == expected_val else 0.0
        elif isinstance(expected_val, list):
            results[key] = 1.0 if len(actual_val or []) >= len(expected_val) * 0.8 else 0.0
        else:
            results[key] = 0.5 if actual_val else 0.0
    return results

def run_evaluation(test_set: list[dict]) -> dict:
    all_scores = []
    latencies = []
    for case in test_set:
        result = extract_file(case["file"], case["doc_type"])
        if result.status != "success":
            all_scores.append(0.0)
            continue
        scores = field_accuracy(case["expected"], result.data)
        all_scores.append(sum(scores.values()) / len(scores))
        latencies.append(result.latency_ms)
    return {
        "document_accuracy": sum(1 for s in all_scores if s > 0.8) / len(all_scores),
        "field_accuracy": sum(all_scores) / len(all_scores),
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
        "total_documents": len(test_set),
    }

if __name__ == "__main__":
    test_set = load_test_set()
    print(json.dumps(run_evaluation(test_set), indent=2))
```

---

## Testing Your Build

### Unit Tests

```python
# tests/test_extractor.py
import pytest
from src.extractor import extract_data, validate_business_rules, score_confidence

SAMPLE_INVOICE = """
INVOICE
Vendor: Acme Corporation
Invoice Number: INV-2024-042
Date: March 15, 2024
Line Items:
  - Consulting Services: 10 hours @ $150.00 = $1,500.00
  - Software License: 1 @ $299.00 = $299.00
Subtotal: $1,799.00
Tax (8%): $143.92
Total: $1,942.92
"""

@pytest.mark.integration
def test_invoice_extraction():
    result = extract_data(SAMPLE_INVOICE, "invoice")
    assert result.status == "success"
    assert result.data["vendor_name"]
    assert result.data["total"] > 0
    assert len(result.data.get("line_items", [])) > 0

def test_unknown_doc_type():
    result = extract_data("some text", "unknown")
    assert result.status == "error"

def test_business_rule_validation():
    bad_invoice = {
        "line_items": [{"description": "A", "quantity": 1, "unit_price": 10, "total": 10}],
        "subtotal": 100,
        "tax": 0,
        "total": 100,
    }
    errors = validate_business_rules(bad_invoice, "invoice")
    assert len(errors) > 0

def test_confidence_scoring():
    data = {"vendor_name": "Acme", "total": 54.0, "notes": ""}
    scores = score_confidence(data, "invoice")
    assert scores["vendor_name"] == 1.0
    assert scores["notes"] == 0.0
```

### Manual Testing Checklist

- [ ] Invoice with line items extracts correctly
- [ ] Resume extracts name, skills, and experience
- [ ] Article extracts title, summary, and sentiment
- [ ] Business rule validation catches mismatched totals
- [ ] Batch endpoint processes multiple files
- [ ] PDF text extraction works (if implemented)
- [ ] API returns clear errors for empty/invalid input

---

## Deployment Notes

### Production Architecture

| Component | Dev | Production |
|-----------|-----|------------|
| Extraction | Synchronous API | Queue-based (Celery/RQ) |
| Storage | JSON files | PostgreSQL + S3 for originals |
| OCR | Optional local | AWS Textract or Google Document AI |
| Review | Manual | Human-in-the-loop UI for low-confidence fields |

### Scaling Batch Processing

```python
# For large batches, use async processing with a job queue
import uuid

jobs: dict[str, dict] = {}

def submit_batch_job(files: list[tuple[str, str]]) -> str:
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "results": []}
    # In production: push to Celery/RQ worker
    jobs[job_id]["results"] = batch_extract(files)
    jobs[job_id]["status"] = "complete"
    return job_id
```

### Cost Management

- Use `gpt-4o-mini` for simple documents (resumes, articles) — 10x cheaper
- Reserve `gpt-4o` for complex invoices with tables
- Cache extractions by document hash to avoid re-processing
- Truncate input to 8,000 characters unless full document is needed

---

## Extensions and Challenges

- **OCR pipeline**: Add `pytesseract` + `pdf2image` for scanned PDFs
- **Custom schemas**: Let users define extraction schemas via JSON config uploaded at runtime
- **Human review UI**: Build a Streamlit app showing extractions with editable fields
- **Database storage**: Persist results to SQLite/PostgreSQL with full audit trail
- **Multi-page invoices**: Split PDF pages and merge extraction results
- **Active learning**: Flag low-confidence extractions for human labeling, then fine-tune

## Resources

- [OpenAI Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs) — Schema-enforced extraction
- [YouTube: Document AI](https://www.youtube.com/watch?v=xEPX9Hkm5CQ) — Building document processing systems
- [Instructor Library](https://github.com/jxnl/instructor) — Structured extraction from LLMs

---

## Next Lesson

**Project 8: Chatbot with Long-Term Memory** — Build a chatbot that remembers user preferences, past conversations, and important facts across sessions.
