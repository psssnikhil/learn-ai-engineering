---
title: LLM-as-Judge
description: >-
  Design evaluation rubrics, understand judge bias, calibrate against human
  labels, and know when LLM-as-judge is the wrong approach
duration: 50 min
difficulty: intermediate
has_code: true
module: module-19
---
# LLM-as-Judge

## Prerequisites

- Completed Lessons 1 and 2 (Why LLM Evals Matter and Golden Datasets)
- Understanding of the RAGAS metrics introduced in Lesson 2
- Basic Python and familiarity with JSON output from LLM APIs

---

## What You'll Learn

| Objective | Outcome |
|-----------|---------|
| Explain why LLM-as-judge scales evaluation beyond what humans can review | Can calculate the cost-per-judgment and make the economic case |
| Design an effective evaluation rubric with anchored scales | Can write rubrics that produce reliable, consistent scores |
| Identify and mitigate the six known judge biases | Can reduce score noise from position, verbosity, and self-preference bias |
| Calibrate an LLM judge against human labels | Can measure judge reliability and decide when it's trustworthy |
| Choose the right evaluation layer for each criterion | Can build a layered eval strategy that avoids over-relying on LLM judgment |

---

## Intuition First: The Scalability Gap

Your golden set has 300 test cases. You've just updated the system prompt. You need to know whether the new version is better or worse before deploying.

Option A: Have two human reviewers read all 300 responses from both versions. At 5 minutes per response, that's 50 hours of reviewer time per deployment. Deployments take weeks.

Option B: Use an LLM to judge all 600 responses. At $0.005 per judgment (GPT-4.1 at current pricing), that's $3 for the whole evaluation. It takes 10 minutes. You can run this on every PR.

LLM-as-judge fills the scalability gap between deterministic checks (too narrow) and human review (too slow). It enables evaluation at the speed of code review.

```
Evaluation layer costs and speed:

Deterministic checks (regex, JSON schema, exact match):
  Cost: $0.000     Speed: <1s     Coverage: Format, structure only

LLM-as-judge (subjective quality):
  Cost: ~$0.005    Speed: 1-3s    Coverage: Relevance, tone, completeness

Human review (authoritative):
  Cost: ~$5-20     Speed: 5-15min Coverage: Everything, but slowly
```

The key constraint: LLM-as-judge is *scalable but imperfect*. It requires careful rubric design, bias mitigation, and regular calibration against human labels to remain trustworthy.

---

## Why Use an LLM to Judge an LLM?

Many important response properties cannot be evaluated with string matching:

- "Is this response helpful to a frustrated customer?" — Subjective
- "Does this summary capture the key points without distorting them?" — Requires reading comprehension
- "Is the tone appropriate for a medical context?" — Domain-specific judgment
- "Does this explanation make sense to a non-expert?" — Requires modeling the reader

A capable judge model can evaluate these dimensions at scale. Both Promptfoo (`llm-rubric` assertions) and DeepEval (`GEval` metric) support LLM-as-judge natively.

```
Human review cost and speed (baseline):
  100 cases × 2 reviewers × $25/hr × 5 min/case = $417, 8+ hours

LLM-as-judge for same 100 cases:
  100 cases × $0.005/judgment = $0.50, ~3 minutes
  Calibration overhead: 50 human-labeled cases/quarter = $2,500/year
  Net: 99% cost reduction with calibration maintained
```

---

## Designing Evaluation Rubrics

The rubric is the single most important factor in judge quality. A vague rubric produces vague, unreliable scores. A concrete, anchored rubric produces scores that correlate with human judgment.

### The Bad vs. Good Rubric Pattern

```
BAD rubric (vague, unreliable):
  "Rate the response quality from 1-5."

  Problems:
  - What does 3 mean? What does 5 mean?
  - No criteria for what counts as "quality"
  - Different judges will score inconsistently
  - No structured output for automated parsing

GOOD rubric (anchored, actionable):
  "You are an expert evaluator for a customer support AI.
   Rate the response on this scale:

   1 = Completely wrong or potentially harmful
   2 = Misses the main question or contains significant errors
   3 = Addresses the question but incomplete or missing key details
   4 = Accurate, complete, and professionally toned
   5 = Excellent: accurate, complete, concise, and with a clear next step

   Before scoring, check these binary criteria:
   □ Does it directly address the user's question (not a tangential topic)?
   □ Does it avoid fabricating facts, policies, or prices?
   □ Does it include a clear next step for the user?

   Immediately score 1 if: any fabricated policy or price is stated.
   Immediately score 1 if: the response is dismissive or condescending.

   Return JSON:
   {
     'criteria_checks': {'addresses_question': bool, 'no_fabrication': bool, 'has_next_step': bool},
     'score': 1-5,
     'reasoning': 'Two sentences explaining the score'
   }"
```

### Rubric Design Principles

| Principle | Implementation |
|-----------|---------------|
| **Anchored scale** | Define what each score level means with concrete examples |
| **Binary sub-checks first** | Run pass/fail checks before holistic score; instant failures prevent score inflation |
| **Negative criteria explicit** | State what constitutes score=1, not just what constitutes score=5 |
| **Domain context included** | Add relevant policies, product constraints, and audience description |
| **Structured JSON output** | Request `{score: int, reasoning: str, criteria: {key: bool}}` for reliable parsing |
| **Reasoning required** | Force the judge to articulate its reasoning; exposes incorrect logic |

### A Production-Quality Judge Implementation

```python
import json
import re
from typing import Optional

CUSTOMER_SUPPORT_RUBRIC = """
You are an expert quality evaluator for a customer support AI system.

YOUR TASK: Evaluate the AI response against the evaluation criteria below.

BINARY CHECKS (evaluate these first):
1. addresses_question: Does the response directly address what the user asked? (not a tangential topic)
2. no_fabrication: Is the response free of fabricated policies, prices, features, or URLs?
3. professional_tone: Is the tone professional, empathetic, and appropriate for a frustrated customer?
4. has_next_step: Does the user know what to do next after reading this response?

INSTANT FAIL (score 1) if ANY of these are true:
- The response contains a price, policy, or feature that is not in the provided context
- The response is dismissive, condescending, or sarcastic
- The response ignores the user's question entirely

SCORING SCALE:
1 = Instant fail condition triggered
2 = Most binary checks fail; response is largely unhelpful
3 = Most binary checks pass; response helps but lacks depth or next steps
4 = All binary checks pass; response is complete and professional
5 = All checks pass + exceptionally concise, empathetic, and actionable

CONTEXT (use this as ground truth for accuracy checks):
{context}

USER QUESTION:
{question}

AI RESPONSE:
{response}

Return ONLY valid JSON:
{
  "binary_checks": {
    "addresses_question": true/false,
    "no_fabrication": true/false,
    "professional_tone": true/false,
    "has_next_step": true/false
  },
  "instant_fail": true/false,
  "score": 1-5,
  "reasoning": "Two sentences: what the response did well and what it missed.",
  "primary_failure": "null or the most important issue"
}
"""

def llm_judge(
    question: str,
    response: str,
    context: str,
    judge_model: str = "gpt-4.1",
    llm_client = None,
) -> dict:
    """
    Run LLM-as-judge evaluation with the customer support rubric.
    Returns structured evaluation result with score and reasoning.
    """
    prompt = CUSTOMER_SUPPORT_RUBRIC.format(
        context=context,
        question=question,
        response=response,
    )

    raw = llm_client.chat.completions.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,      # Deterministic judge
        response_format={"type": "json_object"},
    )
    result_text = raw.choices[0].message.content

    try:
        result = json.loads(result_text)
        return {
            "score": result.get("score", 1),
            "passed": result.get("score", 1) >= 4,
            "binary_checks": result.get("binary_checks", {}),
            "instant_fail": result.get("instant_fail", False),
            "reasoning": result.get("reasoning", ""),
            "primary_failure": result.get("primary_failure"),
            "judge_model": judge_model,
        }
    except (json.JSONDecodeError, KeyError) as e:
        return {"score": 1, "passed": False, "error": str(e), "raw": result_text}
```

---

## Judge Bias: What Goes Wrong

LLM judges are systematically biased in ways that can corrupt your evaluation results if unaddressed. Knowing the biases is the first step to mitigating them.

### The Six Known Biases

```python
JUDGE_BIASES = {
    "position_bias": {
        "description": "In pairwise comparison, judges prefer the response that appears first",
        "evidence": "~30% of preferences flip when order is reversed",
        "mitigation": "Always run pairwise comparisons in both orders; use the consistent result",
    },
    "verbosity_bias": {
        "description": "Longer responses score higher regardless of quality",
        "evidence": "Adding padding sentences to a response increases its score by 0.4-0.8 points",
        "mitigation": "Add explicit conciseness criterion: penalize unnecessary length",
    },
    "self_preference": {
        "description": "GPT-4 rates GPT-4 outputs higher; Claude rates Claude outputs higher",
        "evidence": "~10-15% higher scores when judging own family outputs",
        "mitigation": "Use a judge from a different model family when comparing models",
    },
    "leniency_bias": {
        "description": "Judges cluster scores in the middle, avoiding extremes",
        "evidence": "Score distributions cluster at 3-4 on a 1-5 scale; 1s and 5s are rare",
        "mitigation": "Use binary pass/fail for critical criteria; reserve scale for nuance",
    },
    "authority_bias": {
        "description": "Well-formatted, confident-sounding responses score higher even when wrong",
        "evidence": "Adding LaTeX formatting to an incorrect answer increases score by 0.5 points",
        "mitigation": "Add explicit fact-checking sub-criteria; require citation of source",
    },
    "anchoring_bias": {
        "description": "Scores in batch evaluation are influenced by earlier scores in the batch",
        "evidence": "Score on the 50th case in a batch correlates with the 49th case's score",
        "mitigation": "Evaluate each case independently; shuffle case order between runs",
    },
}
```

### Mitigating Position Bias in Pairwise Comparison

```python
def pairwise_compare_with_debiasing(
    response_a: str,
    response_b: str,
    question: str,
    context: str,
    judge_model: str = "gpt-4.1",
    llm_client = None,
) -> dict:
    """
    Compare two responses using position swapping to eliminate order bias.
    A reliable winner is consistent across both orderings.
    An inconsistent result (ties both runs) indicates the responses are equivalent.
    """
    PAIRWISE_RUBRIC = """
    You are comparing two AI responses to the same customer question.

    Question: {question}
    Context: {context}

    RESPONSE A:
    {response_a}

    RESPONSE B:
    {response_b}

    Which response is better? Consider: accuracy, completeness, tone, actionability.
    Return JSON: {{"winner": "A" or "B" or "tie", "reason": "one sentence"}}
    """

    def judge_pair(r1, r2, label_1="A", label_2="B"):
        prompt = PAIRWISE_RUBRIC.format(
            question=question, context=context,
            response_a=r1, response_b=r2,
        )
        raw = llm_client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        result = json.loads(raw.choices[0].message.content)
        return result.get("winner", "tie"), result.get("reason", "")

    # Run 1: A first, B second
    winner_1, reason_1 = judge_pair(response_a, response_b)
    # Run 2: B first, A second (position swapped)
    winner_2_raw, reason_2 = judge_pair(response_b, response_a)
    # Flip the winner label to align with original naming
    winner_2 = "A" if winner_2_raw == "B" else ("B" if winner_2_raw == "A" else "tie")

    # Only declare a winner if consistent across both orderings
    if winner_1 == winner_2 and winner_1 != "tie":
        confident_winner = winner_1
    else:
        confident_winner = "tie"    # Inconsistent = effectively tied

    return {
        "winner": confident_winner,
        "confident": confident_winner != "tie",
        "run_1": {"winner": winner_1, "reason": reason_1},
        "run_2": {"winner": winner_2, "reason": reason_2},
        "judge_model": judge_model,
    }
```

### Choosing the Right Judge Model

| Strategy | When to Use | Trade-offs |
|----------|-------------|-----------|
| **Same model judges itself** | Fast iteration in development | Self-preference bias; not for final evals |
| **Stronger judges weaker** | GPT-4.1 judges GPT-4.1-mini | Reasonable; some family bias remains |
| **Different family** | Claude judges GPT outputs | Reduces self-preference; may have calibration differences |
| **Ensemble of judges** | High-stakes decisions | Most reliable; 2-3x cost |
| **Smaller judge for screening** | First pass to filter obvious failures | Fast and cheap; use stronger judge on borderline cases |

For most production use cases: use GPT-4.1 as your default judge if you're primarily using GPT models. For model comparison A/B tests, use a neutral judge (Claude, or ideally an ensemble).

---

## Calibration Against Human Labels

An uncalibrated judge is dangerous. It gives you false confidence in metrics that don't correlate with actual user experience. Calibration measures how well LLM judge scores correlate with human judgments.

### The Calibration Workflow

```python
from scipy.stats import pearsonr, spearmanr

def calibrate_judge(
    judge_scores: list[float],
    human_scores: list[float],
    binary_threshold: float = 4.0,   # Scores ≥ threshold = "pass"
) -> dict:
    """
    Measure agreement between LLM judge scores and human reviewer scores.

    Target metrics:
    - Pearson r ≥ 0.70: Judge rank-ordering correlates with human rank-ordering
    - Binary agreement ≥ 0.80: Pass/fail decisions match humans ≥80% of the time
    - Spearman r ≥ 0.70: Robust to outliers (non-parametric correlation)
    """
    n = len(judge_scores)
    assert n == len(human_scores), "Score lists must have equal length"
    assert n >= 30, f"Need ≥30 pairs for meaningful calibration; have {n}"

    pearson_r, pearson_p = pearsonr(judge_scores, human_scores)
    spearman_r, spearman_p = spearmanr(judge_scores, human_scores)

    judge_binary = [1 if s >= binary_threshold else 0 for s in judge_scores]
    human_binary = [1 if s >= binary_threshold else 0 for s in human_scores]
    binary_agreement = sum(j == h for j, h in zip(judge_binary, human_binary)) / n

    # False positive rate: judge says pass, human says fail
    false_positives = sum(1 for j, h in zip(judge_binary, human_binary)
                         if j == 1 and h == 0)
    fpr = false_positives / max(sum(h == 0 for h in human_binary), 1)

    # False negative rate: judge says fail, human says pass
    false_negatives = sum(1 for j, h in zip(judge_binary, human_binary)
                         if j == 0 and h == 1)
    fnr = false_negatives / max(sum(h == 1 for h in human_binary), 1)

    calibrated = pearson_r >= 0.70 and binary_agreement >= 0.80

    return {
        "pearson_r": round(float(pearson_r), 3),
        "pearson_p": round(float(pearson_p), 4),
        "spearman_r": round(float(spearman_r), 3),
        "binary_agreement": round(binary_agreement, 3),
        "false_positive_rate": round(fpr, 3),
        "false_negative_rate": round(fnr, 3),
        "n_pairs": n,
        "calibrated": calibrated,
        "verdict": (
            "Judge is calibrated and suitable for quality gates." if calibrated
            else f"Judge NOT calibrated (r={pearson_r:.2f}, agreement={binary_agreement:.0%}). "
                 "Revise rubric or use stronger judge model."
        ),
    }
```

### The Calibration Process Step by Step

1. **Select 100–200 cases** from your golden set that span the difficulty range (easy, medium, hard)
2. **Get two human reviewers** to score each case independently using the same rubric
3. **Resolve inter-rater disagreements**: For score differences > 1 point, discuss and document consensus
4. **Run the LLM judge** on the exact same 200 cases
5. **Compute calibration metrics**: Pearson r, binary agreement, FPR, FNR
6. **Iterate if needed**: If r < 0.70, revise the rubric (add concrete examples, add binary sub-checks), switch to a stronger judge model, or add more context to the prompt
7. **Re-calibrate quarterly**: Model updates can shift judge behavior; re-calibrate every 3 months or after any judge model change

```python
# DeepEval implementation with custom G-Eval criteria
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

support_quality_metric = GEval(
    name="Customer Support Quality",
    criteria="""
    Evaluate the customer support response against these criteria:

    1. ACCURACY (weight: 40%)
       - Is every stated policy, price, and feature verifiable from the context?
       - Score 0 on this criterion if any fabricated information is present.

    2. COMPLETENESS (weight: 30%)
       - Does the response address all parts of the user's question?
       - Score 0 if a sub-question is completely ignored.

    3. TONE (weight: 15%)
       - Is the tone professional, empathetic, and solution-focused?
       - Score 0 if dismissive or condescending.

    4. ACTIONABILITY (weight: 15%)
       - Does the user have a clear next step?
       - Score 0 if no next step is provided for an actionable request.
    """,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    threshold=0.70,
    model="gpt-4.1",
)
```

---

## When NOT to Use LLM-as-Judge

LLM-as-judge is a powerful tool, but it has real limits. Misapplying it wastes money and provides false confidence.

### The Evaluation Layer Stack

```
Use the cheapest, most reliable evaluator for each criterion:

Layer 1: Deterministic checks (free, instant, 100% reliable)
  ✓ JSON schema validation (jsonschema library)
  ✓ Required field presence (key in dict)
  ✓ Regex pattern matching (format compliance)
  ✓ String exact match (for constrained outputs)
  ✓ Code execution (unit tests for generated code)

Layer 2: Retrieval-based checks (cheap, fast)
  ✓ Faithfulness via RAGAS (grounded in retrieved context)
  ✓ Citation verification (stated source is in retrieved chunks)

Layer 3: LLM-as-judge (moderate cost, scalable)
  ✓ Subjective quality (helpfulness, tone, completeness)
  ✓ Reasoning quality (is the explanation coherent?)
  ✓ Style compliance (matches brand voice)
  ✗ NOT for: factual verification, code correctness, structured output

Layer 4: Human review (expensive, authoritative)
  ✓ Calibration baseline
  ✓ Edge cases the rubric doesn't cover
  ✓ Final arbitration on close cases
  ✓ High-stakes decisions (medical, legal, financial)
```

```python
# Decision matrix: which evaluator to use for each criterion
EVALUATOR_SELECTION = {
    "json_schema_valid":        "deterministic",   # jsonschema.validate()
    "required_fields_present":  "deterministic",   # key in response dict
    "response_length":          "deterministic",   # len(response.split()) <= N
    "code_passes_tests":        "deterministic",   # subprocess / pytest
    "grounded_in_context":      "ragas",           # FaithfulnessMetric
    "sources_cited":            "ragas",           # ContextualPrecisionMetric
    "response_is_helpful":      "llm_judge",       # Subjective; scale 1-5
    "tone_is_empathetic":       "llm_judge",       # Subjective
    "factual_accuracy":         "human",           # Judges hallucinate; can't verify facts
    "code_is_correct":          "deterministic",   # Execution, not LLM judgment
    "safe_for_minors":          "human",           # Safety-critical; no LLM shortcuts
    "legal_advice_absent":      "deterministic",   # Regex for legal disclaimers
}
```

!!! warning "Never Use LLM-as-Judge for Factual Verification"
    LLM judges hallucinate when verifying facts, just like the evaluated model. A judge cannot reliably determine whether "the refund policy allows returns within 45 days" is factually correct if it doesn't know the correct policy. Use retrieval-based checks (is the claim grounded in the retrieved context?) instead of asking the judge to verify external facts.

---

## Worked Example: Building a Complete Judge Pipeline

Here's a complete eval pipeline that layers all three non-human evaluator types:

```python
def evaluate_rag_response_complete(
    question: str,
    retrieved_chunks: list[str],
    response: str,
    expected_json_schema: dict | None = None,
    llm_client = None,
) -> dict:
    """
    Complete evaluation using all three layers: deterministic, RAGAS, LLM judge.
    """
    results = {}

    # Layer 1: Deterministic checks
    results["valid_json"] = True
    if expected_json_schema:
        try:
            import jsonschema
            parsed = json.loads(response)
            jsonschema.validate(parsed, expected_json_schema)
        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            results["valid_json"] = False
            results["json_error"] = str(e)

    results["within_length"] = len(response.split()) <= 300
    results["no_pii"] = not bool(re.search(r'\b\d{3}-\d{2}-\d{4}\b', response))

    # Layer 2: RAGAS (faithfulness)
    ragas = evaluate_rag_response(question, retrieved_chunks, response)
    results["faithfulness_score"] = ragas["faithfulness"]["score"]
    results["faithfulness_passed"] = ragas["faithfulness"]["passed"]

    # Layer 3: LLM-as-judge (quality)
    context = "\n\n".join(retrieved_chunks)
    judge_result = llm_judge(question, response, context, llm_client=llm_client)
    results["quality_score"] = judge_result["score"]
    results["quality_passed"] = judge_result["passed"]
    results["reasoning"] = judge_result.get("reasoning", "")

    # Overall pass/fail
    hard_checks = [results.get("valid_json", True), results["no_pii"],
                   results["faithfulness_passed"]]
    soft_checks = [results["within_length"], results["quality_passed"]]

    results["passed"] = all(hard_checks) and sum(soft_checks) >= 1
    results["hard_failures"] = [k for k, v in zip(
        ["valid_json", "no_pii", "faithfulness"], hard_checks
    ) if not v]

    return results
```

---

## Key Takeaways

- LLM-as-judge scales subjective evaluation 1,000× compared to human review, at 1–2% of the cost; calibrate quarterly to keep scores trustworthy
- Anchored rubrics with binary sub-checks and structured JSON output produce the most reliable judge scores; vague rubrics produce noisy, unactionable scores
- Judges are systematically biased (position, verbosity, self-preference, leniency); mitigate with position swapping, explicit conciseness criteria, and cross-family judge selection
- Calibrate before trusting judge scores in quality gates: target Pearson r ≥ 0.70 and binary agreement ≥ 80% against human labels
- Layer your evaluators: deterministic checks first (free, reliable), RAGAS for retrieval quality, LLM judge for subjective quality, humans for calibration and safety-critical decisions
- Never use LLM-as-judge for factual verification, code correctness, or high-stakes decisions (medical, legal, financial)

---

## Further Reading

- [Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges](https://arxiv.org/abs/2406.12624) — Comprehensive analysis of judge bias types and magnitudes
- [Large Language Models Are Not Robust Multiple Choice Selectors](https://arxiv.org/abs/2309.03882) — Research on position bias in LLM evaluation
- [G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — The original G-Eval paper introducing structured LLM evaluation
- [MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — Zheng et al.; foundational work on using LLMs to evaluate conversational quality

---

## Next Lesson

**Lesson 4: Agent Trajectory Evals** — Evaluate multi-step agent behavior at the step level and outcome level, including tool call correctness and forbidden sequence detection.
