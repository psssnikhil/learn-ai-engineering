---
title: Reasoning Models and Test-Time Compute
description: >-
  How o1, o3, and DeepSeek-R1 think before they answer — chain-of-thought,
  inference scaling, and when to pay the compute premium
duration: 55 min
difficulty: advanced
has_code: true
module: module-07
---

# Reasoning Models and Test-Time Compute

## Prerequisites

- [Transformers to LLMs](02-transformers-to-llms.md) — decoder-only generation
- [RLHF](08-rlhf.md) — reward modeling, PPO
- [Instruction Tuning](07-instruction-tuning.md) — SFT baselines

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand what test-time compute is and why it scales | 10 min | Intermediate |
| Trace through chain-of-thought mechanics step by step | 10 min | Intermediate |
| Distinguish reasoning models (o1/o3/R1) from standard LLMs | 10 min | Advanced |
| Know the cost/benefit trade-off and when *not* to use reasoning models | 10 min | Advanced |
| Implement a reasoning-aware wrapper with budget control | 15 min | Advanced |

---

## Intuition First

Standard LLMs are **reflex agents**: give them a prompt, they immediately start
predicting the next token. For "What is 2 + 2?" this is fine. For "Prove that
√2 is irrational using contradiction," the first token they predict commits them
to a reasoning path that may be completely wrong — and they have no mechanism to
backtrack.

Humans solve hard problems differently. A chess grandmaster doesn't blurt out the
first move that comes to mind; they simulate several candidate lines, prune the bad
ones, and only then commit. A mathematician drafts a proof, notices the contradiction
on line 7, crosses it out, and tries a different lemma.

**Test-time compute** is the insight that you can give language models a similar
budget: instead of one forward pass → answer, let the model generate *thinking
tokens* — scratch-pad reasoning that is never shown to the user — before it
produces the final response. The model trades **tokens** for **correctness**, and
both are measurable quantities.

The key empirical finding (OpenAI o1 technical report, 2024): **accuracy on hard
benchmarks scales predictably with the log of inference-time compute**, much like
training-time scaling laws (Chinchilla). You get two independent knobs.

---

## Core Theory

### 1. Scaling Laws: Training-Time vs. Test-Time

The classic Chinchilla scaling law says: given a compute budget `C`, split it
roughly evenly between model parameters `N` and training tokens `D`. More compute
→ better model, but you can only spend it once (at training time).

Test-time compute opens a **second axis**:

```
Loss = f(N, D, T)
```

where `T` is the number of tokens generated *at inference*. OpenAI's o1 paper
showed that plotting pass@1 on AIME math problems against log(inference tokens)
gives a near-linear curve — the same shape as the training compute curve.

This matters economically: a smaller, cheaper model given a large inference budget
can match a much larger model with a tiny budget on hard tasks. A 7B parameter
reasoning model with 8,000 thinking tokens can outscore a 70B standard model on
competition math.

### 2. Chain-of-Thought (CoT): The Mechanism

Chain-of-thought prompting (Wei et al., 2022) was the first demonstration that
**showing intermediate steps in the prompt** causes the model to produce them too,
and that those steps improve accuracy. Few-shot CoT:

```
Q: Roger has 5 tennis balls. He buys 2 cans, each with 3 balls. How many?
A: Roger starts with 5. 2 cans × 3 balls = 6 new balls. 5 + 6 = 11. Answer: 11.

Q: The cafeteria had 23 apples. They used 20 to make lunch, then bought 6 more. How many?
A: <model continues in the same step-by-step style>
```

Zero-shot CoT (Kojima et al., 2022) showed the three-word magic: **"Let's think
step by step"** is enough to trigger chain-of-thought in sufficiently large models.

**Why does it work?** Autoregressive generation is sequential computation. Each
token can attend to all prior tokens. By generating intermediate reasoning tokens,
the model is effectively doing **distributed computation across the sequence** —
each reasoning token becomes a reusable intermediate result that later tokens can
condition on. This is qualitatively different from trying to compress the full
reasoning into a single token.

!!! note "The serial computation argument"
    A transformer with `L` layers has fixed depth of computation per token. But
    generating `T` reasoning tokens before the answer means the model performs
    `L × T` layers of computation total. More tokens = more serial depth = more
    expressive computation, even with fixed model size.

### 3. Process Reward Models (PRMs) vs. Outcome Reward Models (ORMs)

Standard RLHF trains on **outcome rewards**: did the final answer match the gold
label? This is fine for short tasks but fails for multi-step reasoning — the model
can arrive at the right answer via flawed reasoning (or fail for the wrong reason).

**Process Reward Models** assign scores to *intermediate reasoning steps*, not just
final answers. OpenAI's "Let's Verify Step by Step" (Lightman et al., 2023) showed
that:

1. Human raters can reliably identify which intermediate steps are wrong.
2. Training a verifier on step-level labels (PRM800K dataset) dramatically
   outperforms outcome-only reward models.
3. Best-of-N search using a PRM beats greedy decoding by a large margin.

```
ORM: judge(final_answer) → {0, 1}
PRM: judge(step_1), judge(step_2), ..., judge(step_k) → product of step scores
```

The PRM score for a complete chain is typically the product (or minimum) of step
scores, making it sensitive to any wrong intermediate step.

### 4. Search Strategies at Inference

Given a PRM, you can do structured search over reasoning paths:

| Strategy | Description | Cost |
|----------|-------------|------|
| Greedy | Generate one chain, take it | 1× |
| Best-of-N (BoN) | Generate N chains independently, take highest PRM score | N× |
| Beam search | Keep top-k partial chains at each step, expand | k× per step |
| MCTS | Monte Carlo Tree Search over reasoning steps | Variable |
| Self-consistency | Generate N chains, majority vote on final answer | N× |

**Best-of-N** is the simplest and most widely used. For N samples:

```
accuracy(N) ≈ 1 - (1 - p)^N
```

where `p` is the probability any single chain is correct. This gives diminishing
returns as N grows large, but is remarkably effective in practice.

**Self-consistency** (Wang et al., 2022) doesn't need a PRM — it just takes
the most common final answer across N chains. This works because diverse reasoning
paths that agree on the answer are likely all correct.

### 5. o1 and o3: OpenAI's Reasoning Models

OpenAI's o-series models integrate these ideas into a unified system:

- **Hidden chain-of-thought**: The model generates a "thinking" block (sometimes
  thousands of tokens) that is not shown to the user. This lets the model explore
  dead ends without confusing the user.
- **Reinforcement learning on reasoning**: The model is trained with RL where
  the reward signal comes from verified final answers (math, code, logic), not
  human preference ratings.
- **Effort levels**: o1-mini, o1, o1-pro, o3-mini, o3 — progressively more
  inference compute, traded against cost.

The o3 model on ARC-AGI (a benchmark testing novel problem-solving) scored 87.5%
at "high compute" setting versus ~5% for GPT-4o. The compute budget was estimated
at $17–$1000 per task — orders of magnitude above standard inference.

### 6. DeepSeek-R1: Open-Weight Reasoning

DeepSeek-R1 (January 2025) showed that strong reasoning can emerge from:

1. **Cold-start SFT** on a small set of chain-of-thought examples
2. **GRPO** (Group Relative Policy Optimization) — a simpler RL alternative to PPO
   that doesn't need a separate value network
3. **Rule-based rewards** for math/code (exact match, compiler pass/fail) rather
   than a learned reward model

The emergent behavior: the model spontaneously developed **self-correction
mid-reasoning** — producing phrases like "Wait, let me reconsider..." when it
detected inconsistency. This was not explicitly trained; it emerged from the RL
process.

R1's weights are open-source (MIT license), enabling distillation: Qwen-7B and
LLaMA-8B fine-tuned on R1's reasoning traces approach R1 performance on many
benchmarks.

### 7. The Inference Scaling Curve

```
         Accuracy (%)
100 |                                          ●  o3-high
    |                                  ●  o3-low
 80 |                          ●  o1
    |                 ●  o1-mini
 60 |          ●  GPT-4o
    |    ●  GPT-3.5
 40 |─────────────────────────────────────────────────────
    0.001    0.01     0.1       1        10      100
                Inference compute (relative units, log scale)
```

*Illustrative curve based on published AIME 2024 results.*

The curve shows:
- Standard models plateau early — adding more inference compute (temperature
  sampling, top-k) doesn't help much
- Reasoning models have a steeper slope — they efficiently use additional compute
- There's a crossover point: below it, standard models are more efficient; above
  it, reasoning models win

---

## Worked Example: Step-by-Step CoT on a Logic Puzzle

**Problem**: Alice, Bob, and Carol each have a different pet (cat, dog, fish).
Alice doesn't have the cat. Bob doesn't have the dog. Carol has the fish.
Who has the dog?

**Standard LLM (greedy)**: might output "Alice has the dog" — wrong without
careful tracking.

**Chain-of-thought approach**:

```python
import openai

client = openai.OpenAI()

puzzle = """
Alice, Bob, and Carol each have a different pet (cat, dog, fish).
Clues:
1. Alice doesn't have the cat.
2. Bob doesn't have the dog.
3. Carol has the fish.
Who has the dog?
"""

# Standard approach — single forward pass
response_standard = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": puzzle}],
    max_tokens=20
)
print("Standard:", response_standard.choices[0].message.content)

# Chain-of-thought — trigger reasoning before answer
response_cot = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": puzzle + "\n\nThink step by step before giving the answer."
    }],
    max_tokens=300
)
print("CoT:", response_cot.choices[0].message.content)
```

**Expected CoT trace**:
```
Step 1: Carol has the fish (given). So cat and dog are split between Alice and Bob.
Step 2: Alice doesn't have the cat. So Alice has the dog.
Step 3: Bob gets the cat.
Answer: Alice has the dog.
```

The CoT trace creates intermediate facts ("cat and dog split between Alice and Bob")
that make the final deduction trivial. Without the trace, the model must do all
this reasoning implicitly in a single forward pass.

---

## Implementation

### Reasoning-Aware API Wrapper

Different models expose reasoning differently. Here's a unified wrapper:

```python
from dataclasses import dataclass
from typing import Optional
import openai
import time

@dataclass
class ReasoningResponse:
    thinking: Optional[str]   # visible only if model exposes it
    answer: str
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    latency_seconds: float

def call_reasoning_model(
    prompt: str,
    model: str = "o3-mini",
    effort: str = "medium",   # "low" | "medium" | "high" for o-series
) -> ReasoningResponse:
    """
    Wrapper around OpenAI reasoning models.
    effort maps to reasoning_effort parameter.
    """
    client = openai.OpenAI()
    start = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        reasoning_effort=effort,  # o3-mini and o1 support this
    )
    latency = time.time() - start

    usage = response.usage
    return ReasoningResponse(
        thinking=None,  # o-series hides the thinking block
        answer=response.choices[0].message.content,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        reasoning_tokens=getattr(usage, "completion_tokens_details", {}).get(
            "reasoning_tokens", 0
        ),
        latency_seconds=latency,
    )


def call_with_explicit_cot(
    prompt: str,
    model: str = "gpt-4o",
) -> ReasoningResponse:
    """
    Elicit chain-of-thought from a standard model using prompting.
    Useful when you don't have access to a reasoning model.
    """
    client = openai.OpenAI()
    start = time.time()

    system = (
        "You are a careful problem solver. Before giving your final answer, "
        "work through the problem step by step inside <thinking> tags. "
        "Then give your final answer after </thinking>."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1500,
    )
    latency = time.time() - start
    content = response.choices[0].message.content

    # Parse out thinking vs answer
    thinking = ""
    answer = content
    if "<thinking>" in content and "</thinking>" in content:
        import re
        m = re.search(r"<thinking>(.*?)</thinking>(.*)", content, re.DOTALL)
        if m:
            thinking = m.group(1).strip()
            answer = m.group(2).strip()

    usage = response.usage
    return ReasoningResponse(
        thinking=thinking,
        answer=answer,
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        reasoning_tokens=0,
        latency_seconds=latency,
    )
```

### Best-of-N Self-Consistency

```python
from collections import Counter
import concurrent.futures
import re

def extract_final_answer(text: str) -> str:
    """Extract the numeric/symbolic final answer from a CoT response."""
    # Matches "Answer: X" or "= X" at end of response
    patterns = [
        r"[Aa]nswer:\s*(.+?)\.?\s*$",
        r"=\s*(\d+[\d,\.]*)\s*$",
        r"the answer is\s+(.+?)\.?\s*$",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.MULTILINE)
        if m:
            return m.group(1).strip()
    # Fallback: last line
    return text.strip().split("\n")[-1].strip()


def best_of_n_self_consistency(
    prompt: str,
    model: str = "gpt-4o",
    n: int = 5,
    temperature: float = 0.7,
) -> dict:
    """
    Generate N independent chains, take majority vote on final answer.
    Returns the winner and vote distribution.
    """
    client = openai.OpenAI()

    def single_sample(_):
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt + "\n\nThink step by step."
            }],
            temperature=temperature,
            max_tokens=800,
        )
        return resp.choices[0].message.content

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as exe:
        chains = list(exe.map(single_sample, range(n)))

    answers = [extract_final_answer(c) for c in chains]
    vote_counts = Counter(answers)
    winner = vote_counts.most_common(1)[0][0]

    return {
        "winner": winner,
        "votes": dict(vote_counts),
        "chains": chains,
        "confidence": vote_counts[winner] / n,
    }


# Usage
result = best_of_n_self_consistency(
    "A train travels 120 km at 60 km/h, then 80 km at 40 km/h. What is the average speed?",
    n=5,
)
print(f"Answer: {result['winner']} (confidence: {result['confidence']:.0%})")
print(f"Vote distribution: {result['votes']}")
```

### Compute Budget Controller

```python
from dataclasses import dataclass
from enum import Enum

class TaskComplexity(Enum):
    SIMPLE = "simple"       # factual lookup, format conversion
    MODERATE = "moderate"   # multi-step arithmetic, short code
    HARD = "hard"           # competition math, complex debugging
    RESEARCH = "research"   # novel problem-solving, formal proofs

@dataclass
class ModelConfig:
    model: str
    reasoning_effort: Optional[str]
    max_output_tokens: int
    approx_cost_per_1k_output: float  # USD

ROUTING_TABLE = {
    TaskComplexity.SIMPLE: ModelConfig(
        model="gpt-4o-mini",
        reasoning_effort=None,
        max_output_tokens=512,
        approx_cost_per_1k_output=0.0006,
    ),
    TaskComplexity.MODERATE: ModelConfig(
        model="gpt-4o",
        reasoning_effort=None,
        max_output_tokens=2000,
        approx_cost_per_1k_output=0.010,
    ),
    TaskComplexity.HARD: ModelConfig(
        model="o3-mini",
        reasoning_effort="medium",
        max_output_tokens=8000,
        approx_cost_per_1k_output=0.044,
    ),
    TaskComplexity.RESEARCH: ModelConfig(
        model="o3",
        reasoning_effort="high",
        max_output_tokens=16000,
        approx_cost_per_1k_output=0.120,
    ),
}


def routed_call(prompt: str, complexity: TaskComplexity) -> str:
    cfg = ROUTING_TABLE[complexity]
    client = openai.OpenAI()

    kwargs = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": cfg.max_output_tokens,
    }
    if cfg.reasoning_effort:
        kwargs["reasoning_effort"] = cfg.reasoning_effort

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content
```

---

## Edge Cases & Misconceptions

### "Reasoning models are always better"

**Wrong.** For simple tasks — factual Q&A, summarization, translation — reasoning
models are dramatically over-engineered. An o3 call costs 10–100× more than GPT-4o
and takes longer. Use complexity routing (see above).

!!! warning "Latency trap"
    o1/o3 models with `reasoning_effort="high"` can take 30–120 seconds for
    a single response. User-facing applications with SLA requirements cannot
    afford this for most queries. Use asynchronous jobs or queue-based patterns.

### "Longer thinking is always better"

Not always. On easy tasks, additional reasoning tokens can introduce errors by
**over-thinking** — the model considers edge cases that don't apply and confuses
itself. OpenAI observed this in o1-preview on simple math problems.

### "Chain-of-thought is just prompting"

CoT prompting works on large models (≥7B parameters) but fails on small ones.
Below ~1B parameters, generating reasoning tokens *hurts* — the model doesn't
have enough capacity to reason coherently, so the reasoning trace is nonsense
that poisons the final answer.

### "Self-consistency always improves accuracy"

Self-consistency helps when errors are random. When errors are *systematic*
(all samples make the same conceptual mistake), majority voting locks in the
wrong answer. You need a PRM or verifier to detect systematic errors.

### "Visible CoT is the same as hidden CoT"

OpenAI explicitly states that o1's hidden thinking is *different* from what it
outputs to the user. The thinking block may contain reasoning that is
"more raw" than the polished response. You cannot reconstruct o1's actual
reasoning from its API output.

---

## Production Connection

### When to use reasoning models

| Use case | Recommendation |
|----------|----------------|
| Competition math / formal proofs | o3 high — worth every penny |
| Complex code debugging (segfaults, race conditions) | o3-mini medium |
| Multi-hop research questions | o1 or o3-mini |
| Agentic planning with long task horizons | o1 / o3-mini |
| Simple classification / extraction | GPT-4o-mini — avoid reasoning overhead |
| Streaming chat (latency sensitive) | Standard models with CoT prompting |
| High-volume batch (cost sensitive) | Best-of-N on standard models |

### Cost modeling

Before deploying a reasoning model, estimate costs:

```python
def estimate_reasoning_cost(
    prompts_per_day: int,
    avg_input_tokens: int,
    avg_reasoning_tokens: int,   # hidden thinking tokens
    avg_output_tokens: int,
    input_cost_per_1m: float = 15.0,   # o3 pricing (illustrative)
    output_cost_per_1m: float = 60.0,  # reasoning + output tokens
) -> float:
    """Returns daily cost in USD."""
    daily_input = prompts_per_day * avg_input_tokens
    daily_output = prompts_per_day * (avg_reasoning_tokens + avg_output_tokens)
    return (
        daily_input / 1_000_000 * input_cost_per_1m
        + daily_output / 1_000_000 * output_cost_per_1m
    )

# Example: 1000 o3 calls/day, 500 input tokens, 2000 reasoning, 300 output
cost = estimate_reasoning_cost(1000, 500, 2000, 300)
print(f"Daily cost: ${cost:.2f}")  # ~$138/day — substantial
```

### Async pattern for long-running reasoning

```python
import asyncio
import openai

async def reasoning_with_timeout(
    prompt: str,
    timeout_seconds: float = 60.0,
    model: str = "o3-mini",
) -> Optional[str]:
    client = openai.AsyncOpenAI()
    try:
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                reasoning_effort="medium",
            ),
            timeout=timeout_seconds,
        )
        return response.choices[0].message.content
    except asyncio.TimeoutError:
        # Fall back to standard model for user-facing response
        return None
```

---

## Key Takeaways

- **Test-time compute** is a second independent scaling axis: accuracy grows with
  log(inference tokens), complementary to training compute scaling.
- **Chain-of-thought** works by using generated tokens as distributed serial
  computation — each reasoning token is a reusable intermediate result.
- **Process Reward Models** (PRMs) score intermediate steps, enabling structured
  search (best-of-N, beam, MCTS) that dramatically outperforms greedy decoding.
- **o1/o3** use hidden chain-of-thought + RL on verifiable outcomes; they don't
  expose the reasoning trace to users.
- **DeepSeek-R1** showed open-source GRPO-based training produces comparable
  reasoning, and distillation into smaller models is effective.
- **Self-consistency** (majority vote across N chains) is a powerful, zero-cost
  alternative to PRMs when you can afford N× inference.
- **Routing is critical**: use simple models for simple tasks, reserve reasoning
  models for genuinely hard problems where their cost premium is justified.
- **Latency is the silent killer**: reasoning models can take minutes — always
  build async patterns and fallback routes into production systems.

---

## Further Reading

- [OpenAI o1 System Card](https://openai.com/index/openai-o1-system-card/) — official capability benchmarks
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — Lightman et al., PRM800K and process rewards
- [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) — Wang et al.
- [Chain-of-Thought Prompting Elicits Reasoning](https://arxiv.org/abs/2201.11903) — Wei et al., the original CoT paper
- [DeepSeek-R1 Technical Report](https://arxiv.org/abs/2501.12948) — open-weight reasoning
- [Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314) — DeepMind analysis
- [Large Language Monkeys](https://arxiv.org/abs/2407.21787) — coverage vs. accuracy trade-offs

---

## Next Lesson

You've completed the LLM Foundations module. Proceed to the **Build** phase:

→ [Module 09: RAG — Retrieval Augmented Generation](../../../build/module-09-rag-retrieval-augmented-generation/index.md)
