---
title: Hallucination and Factual Accuracy
description: >-
  Understand why LLMs fabricate information and learn practical techniques to
  detect, measure, and reduce hallucinations
duration: 40 min
difficulty: intermediate
has_code: false
module: module-16
youtube: 'https://www.youtube.com/watch?v=HQn1QKQYyJA'
objectives:
  - Explain why LLMs hallucinate
  - Implement hallucination detection techniques
  - Use RAG and grounding to reduce fabrication
  - Build citation and verification systems
---
# Hallucination and Factual Accuracy

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand the root causes of hallucination | 40 min | Intermediate |
| Detect hallucinations programmatically | | |
| Apply grounding techniques to reduce fabrication | | |
| Build verification pipelines | | |

---

## Why LLMs Hallucinate

LLMs are trained to predict the most likely next token, not to be truthful. This leads to confident-sounding but incorrect outputs.

### Types of Hallucination

| Type | Description | Example |
|------|-------------|---------|
| **Factual fabrication** | Inventing facts that sound plausible | "The Eiffel Tower was built in 1892" (actual: 1889) |
| **Entity confusion** | Mixing attributes of different entities | Attributing one author's work to another |
| **Unfaithful reasoning** | Correct premises leading to wrong conclusions | Math errors in multi-step problems |
| **Source fabrication** | Inventing citations and references | "According to a 2023 study in Nature..." (doesn't exist) |
| **Temporal confusion** | Mixing up dates or timelines | Describing events in the wrong chronological order |

### Root Causes

1. **Statistical pattern matching**: Models generate likely text, not verified text
2. **Training data contamination**: Errors in training data become model knowledge
3. **Knowledge cutoff**: Models don't know about events after training
4. **Ambiguity in prompts**: Vague questions invite guessing
5. **Confidence calibration**: Models don't know what they don't know

---

## Detecting Hallucinations

### Approach 1: Self-Consistency Checking

Generate multiple responses and check for agreement:

```python
from openai import OpenAI
import json

client = OpenAI()

def self_consistency_check(prompt, model="gpt-4.1-mini", num_samples=5):
    """Generate multiple responses and check for consistency."""
    responses = []
    
    for _ in range(num_samples):
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8  # Higher temp for diversity
        )
        responses.append(response.choices[0].message.content)
    
    # Use GPT-4.1 to check consistency
    consistency_check = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{
            "role": "user",
            "content": f"""Compare these {num_samples} responses to the same question.
            
Question: {prompt}

Responses:
{chr(10).join(f'{i+1}. {r}' for i, r in enumerate(responses))}

Identify:
1. Claims that appear in ALL responses (likely reliable)
2. Claims that appear in only SOME responses (potentially unreliable)
3. Claims that contradict across responses (likely hallucinated)

Return JSON: {{
    "consistent_claims": [...],
    "inconsistent_claims": [...],
    "contradictions": [...],
    "reliability_score": <0-1>
}}"""
        }],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    return json.loads(consistency_check.choices[0].message.content)

result = self_consistency_check("When was the Python programming language first released?")
print(f"Reliability: {result['reliability_score']}")
```

### Approach 2: Claim Extraction and Verification

```python
def extract_and_verify_claims(text, model="gpt-4.1"):
    """Extract factual claims from text and flag those that need verification."""
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": f"""Extract all factual claims from this text. For each claim, rate your confidence (high/medium/low).

Text: {text}

Return JSON array: [
    {{"claim": "...", "confidence": "high|medium|low", "verifiable": true|false}},
    ...
]"""
        }],
        temperature=0,
        response_format={"type": "json_object"}
    )
    
    claims = json.loads(response.choices[0].message.content)
    
    # Flag claims that need human verification
    flagged = [c for c in claims if isinstance(c, dict) and c.get("confidence") != "high"]
    
    return {"all_claims": claims, "flagged_for_review": flagged}
```

---

## Reducing Hallucinations

### Strategy 1: RAG (Retrieval-Augmented Generation)

Ground model responses in retrieved documents:

```python
def grounded_response(query, retrieved_docs, model="gpt-4.1-mini"):
    """Generate a response grounded in retrieved documents."""
    context = "

".join(
        f"[Source {i+1}]: {doc}" for i, doc in enumerate(retrieved_docs)
    )
    
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "system",
            "content": """Answer the user's question based ONLY on the provided sources.
If the sources don't contain enough information, say "I don't have enough information to answer that."
Always cite your sources using [Source N] notation.
Never make claims that aren't supported by the sources."""
        }, {
            "role": "user",
            "content": f"Sources:
{context}

Question: {query}"
        }],
        temperature=0.3  # Lower temperature for factual tasks
    )
    
    return response.choices[0].message.content
```

### Strategy 2: Ask the Model to Express Uncertainty

```python
system_prompt = """You are a helpful assistant. When answering questions:
- If you are confident in your answer, state it clearly
- If you are uncertain, say "I'm not fully certain, but..."
- If you don't know, say "I don't have reliable information about that"
- Never fabricate sources, citations, or specific statistics
- Distinguish between facts and your interpretation"""
```

### Strategy 3: Structured Output with Citations

```python
def response_with_citations(query, context_docs, model="gpt-4.1-mini"):
    """Generate a response with inline citations."""
    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "system",
            "content": "Respond in JSON format with 'answer' and 'citations' fields. Each citation should reference a specific source document."
        }, {
            "role": "user",
            "content": f"Context: {context_docs}

Question: {query}"
        }],
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    
    return json.loads(response.choices[0].message.content)
```

### Strategy 4: Chain-of-Thought Verification

```python
def verified_response(query, model="gpt-4.1"):
    """Two-pass approach: generate then verify."""
    # Pass 1: Generate response
    initial = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": query}],
        temperature=0.3
    ).choices[0].message.content
    
    # Pass 2: Verify claims
    verification = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": f"""Review this response for factual accuracy:

Question: {query}
Response: {initial}

For each factual claim, assess whether it is:
- CORRECT (you are highly confident)
- UNCERTAIN (you are not sure)  
- LIKELY INCORRECT (conflicts with your knowledge)

Return the corrected response with uncertain claims flagged."""
        }],
        temperature=0
    ).choices[0].message.content
    
    return {"initial": initial, "verified": verification}
```

---

## Hallucination Metrics

| Metric | How to Measure | Good Threshold |
|--------|----------------|----------------|
| Factual accuracy | Human evaluation of claims | >95% for high-stakes |
| Faithfulness | Does response stay within given context? | >90% |
| Citation accuracy | Are cited sources real and relevant? | 100% |
| Abstention rate | How often does model say "I don't know"? | 5-15% (too high = unhelpful) |

---

## Resources

- **Paper: "A Survey on Hallucination in LLMs"** — Comprehensive overview of the problem
- **FActScore**: Automated framework for measuring factual precision
- **RAGAS**: Evaluation framework for RAG-based systems
- **Video: Why AI Lies** — Understanding hallucination at a fundamental level

---

## Key Takeaways

- Hallucination is inherent to how LLMs work — they predict likely text, not truthful text
- Use self-consistency checking to identify unreliable claims
- RAG (grounding in documents) is the most effective hallucination reduction technique
- Encourage uncertainty expression: models that say "I don't know" are more trustworthy
- For high-stakes applications, always include human verification in the pipeline

---

## Next Lesson

**Lesson 4: Prompt Injection and LLM Security** — Learn how attackers exploit LLMs through prompt injection and how to defend your applications.
