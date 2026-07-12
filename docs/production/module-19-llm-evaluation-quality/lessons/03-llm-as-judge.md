---
title: LLM-as-Judge
description: >-
  Design evaluation rubrics, understand judge bias, calibrate against human
  labels, and know when LLM-as-judge is the wrong approach
duration: 40 min
difficulty: intermediate
has_code: false
module: module-19
---
# LLM-as-Judge

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Design effective evaluation rubrics | 40 min | Intermediate |
| Understand and mitigate judge bias | | |
| Calibrate LLM judges against human labels | | |
| Know when not to use LLM-as-judge | | |

---

## Why Use an LLM to Judge an LLM?

Many LLM outputs cannot be evaluated with string matching or regex. "Is this response helpful?" "Does this summary capture the key points?" "Is the tone appropriate for a grieving customer?" These are subjective judgments that historically required human reviewers.

**LLM-as-judge** uses a capable model to score another model's output against a rubric. It scales evaluation to thousands of cases at cents per judgment instead of dollars per human review.

```
Traditional:  Human reviews 100 cases → 8 hours → $400
LLM-as-judge: GPT-4.1 reviews 1,000 cases → 10 minutes → $5
```

Both Promptfoo and DeepEval support LLM-as-judge natively. Promptfoo uses `llm-rubric` assertions; DeepEval provides metrics like `G-Eval` that implement the pattern in Python.

---

## Designing Evaluation Rubrics

A rubric is the single most important factor in judge quality. Vague rubrics produce vague, unreliable scores.

### Bad Rubric vs Good Rubric

```
BAD:  "Rate the response quality from 1-5."

GOOD: "Rate the response on a scale of 1-5:
       1 = Completely wrong or harmful
       2 = Partially relevant but missing key information
       3 = Adequate — answers the question but lacks depth
       4 = Good — accurate, complete, well-structured
       5 = Excellent — accurate, complete, concise, and actionable

       The response must:
       - Directly address the user's question (not tangential topics)
       - Include specific steps or information (not vague platitudes)
       - Not contain fabricated facts or URLs

       Question: {query}
       Response: {response}
       Score:"
```

### Rubric Design Principles

| Principle | Example |
|-----------|---------|
| **Anchored scales** | Define what each score level means with examples |
| **Binary sub-checks** | "Does it mention the refund policy? Yes/No" before holistic score |
| **Negative criteria** | Explicitly list what constitutes a failure |
| **Domain context** | Include company policies, product constraints in the rubric |
| **Structured output** | Request JSON with score + reasoning, not just a number |

```python
JUDGE_RUBRIC = """
You are an expert evaluator for a customer support AI.

Evaluate the following response against these criteria:

1. ACCURACY (weight: 40%)
   - Is the information factually correct per the provided context?
   - Score 0 if any fabricated policy, price, or feature is stated.

2. COMPLETENESS (weight: 30%)
   - Does the response fully address all parts of the user's question?
   - Score 0 if a sub-question is completely ignored.

3. TONE (weight: 15%)
   - Is the tone professional, empathetic, and appropriate?
   - Score 0 if dismissive, condescending, or overly casual.

4. ACTIONABILITY (weight: 15%)
   - Does the user know what to do next?
   - Score 0 if no clear next step is provided.

Return JSON:
{
  "accuracy": 0-10,
  "completeness": 0-10,
  "tone": 0-10,
  "actionability": 0-10,
  "overall": 0-10,
  "reasoning": "Brief explanation of scores",
  "failures": ["List of specific issues, if any"]
}
"""

def llm_judge(query: str, response: str, context: str, judge_model: str = "gpt-4.1") -> dict:
    import json

    prompt = f"{JUDGE_RUBRIC}\n\nContext: {context}\nQuestion: {query}\nResponse: {response}"
    result = call_llm(judge_model, prompt, response_format="json")
    return json.loads(result)
```

---

## Judge Bias: What Goes Wrong

LLM judges are useful but systematically biased. Ignoring these biases produces eval scores that look good in dashboards but do not predict human satisfaction.

### Known Biases

| Bias | Description | Mitigation |
|------|-------------|------------|
| **Position bias** | Prefers the first or longer response in pairwise comparison | Swap order, average both runs |
| **Self-preference** | GPT-4 tends to rate GPT-4 outputs higher | Use a different model family as judge |
| **Verbosity bias** | Longer responses score higher regardless of quality | Add conciseness criteria to rubric |
| **Authority bias** | Confident, well-formatted answers score higher even when wrong | Add fact-checking sub-criteria |
| **Leniency bias** | Judges cluster scores around 7-8, avoiding extremes | Use binary pass/fail for critical criteria |
| **Anchoring bias** | Early scores influence later judgments in batch eval | Evaluate each case independently |

### Mitigation: Position Swapping

```python
def pairwise_compare(response_a: str, response_b: str, rubric: str, judge_model: str) -> dict:
    """Compare two responses with position swapping to reduce bias."""
    # Run 1: A first, B second
    result_1 = judge_pair(response_a, response_b, rubric, judge_model)

    # Run 2: B first, A second (swapped positions)
    result_2 = judge_pair(response_b, response_a, rubric, judge_model)

    # Consistent winner across both orderings
    a_wins = result_1["winner"] == "A" and result_2["winner"] == "B"
    b_wins = result_1["winner"] == "B" and result_2["winner"] == "A"

    return {
        "winner": "A" if a_wins else "B" if b_wins else "tie",
        "confident": a_wins or b_wins,
        "run_1": result_1,
        "run_2": result_2,
    }
```

### Choosing a Judge Model

| Strategy | When to Use |
|----------|-------------|
| **Same model judges itself** | Fast iteration, development only |
| **Stronger model judges weaker** | GPT-4.1 judges GPT-4.1-mini outputs |
| **Different family** | Claude judges GPT outputs (reduces self-preference) |
| **Ensemble of judges** | High-stakes decisions, average across 2-3 judges |

---

## Calibration Against Human Labels

An uncalibrated judge is worse than no judge — it gives you false confidence. Calibration means measuring how well LLM judge scores correlate with human judgments, then adjusting.

### The Calibration Process

```python
def calibrate_judge(judge_scores: list[float], human_scores: list[float]) -> dict:
    """Measure agreement between LLM judge and human reviewers."""
    from scipy.stats import pearsonr, spearmanr

    pearson_r, pearson_p = pearsonr(judge_scores, human_scores)
    spearman_r, spearman_p = spearmanr(judge_scores, human_scores)

    # Cohen's kappa for binary pass/fail agreement
    judge_binary = [1 if s >= 7 else 0 for s in judge_scores]
    human_binary = [1 if s >= 7 else 0 for s in human_scores]
    agreement = sum(j == h for j, h in zip(judge_binary, human_binary)) / len(judge_binary)

    return {
        "pearson_correlation": pearson_r,
        "spearman_correlation": spearman_r,
        "binary_agreement": agreement,
        "calibrated": pearson_r >= 0.7 and agreement >= 0.8,
    }
```

### Calibration Workflow

1. **Sample 100–200 cases** from your golden set
2. **Two human reviewers** score each case independently
3. **Run the LLM judge** on the same cases
4. **Compute correlation** — target Pearson r ≥ 0.7, binary agreement ≥ 80%
5. **If correlation is low** — revise the rubric, try a different judge model, add binary sub-checks
6. **Re-calibrate quarterly** — model updates shift judge behavior

```python
# DeepEval G-Eval with custom criteria
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

support_quality = GEval(
    name="Support Quality",
    criteria="""
    Evaluate whether the customer support response:
    1. Directly answers the customer's question
    2. Provides accurate information per company policy
    3. Maintains a professional and empathetic tone
    4. Includes a clear next step for the customer
    """,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    threshold=0.7,
)
```

If correlation stays below 0.6 after rubric iteration, the task may be too subjective — fall back to human review for that category.

---

## When NOT to Use LLM-as-Judge

LLM-as-judge is a powerful tool, not a universal solution. Recognize its limits.

### Do Not Use LLM-as-Judge For

| Scenario | Why | Alternative |
|----------|-----|-------------|
| **Factual verification** | Judges hallucinate too — they cannot verify facts they do not know | Retrieval + exact match, knowledge base lookup |
| **Code correctness** | LLMs cannot reliably execute or test code | Unit tests, sandboxed execution, linting |
| **Structured output validation** | JSON schema, regex, and parsers are deterministic and free | `jsonschema.validate()`, regex assertions |
| **Safety-critical decisions** | Medical, legal, financial advice needs human oversight | Human review + rule-based guardrails |
| **Latency/cost measurement** | Judges add latency and cost to every eval run | Instrumentation, direct measurement |
| **Subjective creative tasks** | "Is this poem beautiful?" has no stable rubric | Human panels, user preference data |

### The Right Layering

Use the cheapest, most reliable evaluator for each criterion:

```
Layer 1: Deterministic checks (free, instant)
  → JSON schema, regex, exact match, code execution

Layer 2: Retrieval-based checks (cheap, fast)
  → Faithfulness via RAGAS, citation verification

Layer 3: LLM-as-judge (moderate cost, scalable)
  → Subjective quality, tone, completeness

Layer 4: Human review (expensive, authoritative)
  → Calibration, edge cases, final arbitration
```

Promptfoo makes this layering natural — combine `contains`, `javascript`, `llm-rubric`, and `model-graded-closedqa` assertions in a single test suite.

---

---

## Key Takeaways

- LLM-as-judge scales subjective evaluation but requires careful rubric design
- **Anchored rubrics** with binary sub-checks and structured JSON output produce the most reliable scores
- Judges are biased (position, verbosity, self-preference) — mitigate with position swapping and cross-family judges
- **Calibrate against human labels** before trusting judge scores in quality gates (target r ≥ 0.7)
- Layer evaluators: deterministic checks first, LLM judge for subjective criteria, humans for calibration and edge cases
- Do not use LLM-as-judge for factual verification, code correctness, or safety-critical decisions

---

## Next Lesson

**Lesson 4: Agent Trajectory Evals** — Evaluate multi-step agent behavior at the step level and outcome level, including tool call correctness and reasoning chain validation.
