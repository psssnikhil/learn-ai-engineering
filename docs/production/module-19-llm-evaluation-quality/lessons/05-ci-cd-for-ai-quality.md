---
title: CI/CD for AI Quality
description: >-
  Integrate eval suites into pull request checks, implement canary deployments,
  and use shadow traffic to validate AI changes before production rollout
duration: 50 min
difficulty: advanced
has_code: true
module: module-19
---
# CI/CD for AI Quality

## Prerequisites

- Completed Lessons 1–4 (full evaluation pipeline: evals, golden sets, LLM-as-judge, agent evals)
- Familiarity with GitHub Actions or similar CI/CD platforms
- Understanding of canary deployment concepts from Module 10 Lesson 7

---

## What You'll Learn

| Objective | Outcome |
|-----------|---------|
| Explain why traditional CI/CD is insufficient for AI quality | Can articulate the green CI problem and its production consequences |
| Integrate offline eval suites into PR checks | Can configure a GitHub Actions workflow that blocks merges on quality regression |
| Design canary deployment gates with automated promotion/rollback | Can write evaluation criteria that safely promote AI changes to 100% traffic |
| Implement shadow traffic for zero-risk pre-production validation | Can run new AI versions against real production data without user exposure |
| Manage CI eval cost without sacrificing coverage | Can configure tiered eval strategies that keep CI fast and cheap |

---

## Intuition First: The Code Quality Gap

Traditional CI/CD answers one question: "Does the code work?" It runs unit tests, integration tests, and type checks. A passing build means the application functions correctly.

AI applications require a second question: "Does the *AI* work?" The code can be perfect—imports clean, API calls correct, JSON parsed properly—while the AI system degrades significantly. A prompt change, model update, or retrieval config modification does not break unit tests. It breaks user experience.

```
Traditional CI: Green build = code works
AI quality CI:  Green build + green evals = code works AND AI works

Without eval gates:
  Prompt change → Unit tests: PASS (they test code, not AI) → Deploy
  → Quality drops 15% → Users notice 3 days later → Manual investigation
  → 6 hours to identify prompt change as root cause

With eval gates:
  Prompt change → Unit tests: PASS → Eval suite: FAIL (8 cases regressed)
  → PR blocked → Dev sees which cases failed and why → Fix before merge
  → Quality maintained
```

The goal is to make AI quality regressions as visible and catchable as null pointer exceptions.

---

## Eval in Pull Request Checks

The PR gate is the first line of defense. It runs offline evals on every AI-related change before the code touches any environment.

### Detecting AI-Related Changes

Not every PR needs an eval run. Running evals on dependency updates, documentation changes, or styling fixes wastes money and slows developers.

```python
# eval/change_detector.py
import subprocess
from pathlib import Path

AI_CHANGE_PATTERNS = [
    "prompts/**/*.yaml",           # Prompt template changes
    "prompts/**/*.jinja",
    "config/models.yaml",          # Model configuration
    "config/retrieval.yaml",       # RAG configuration
    "src/agents/**/*.py",          # Agent logic
    "src/rag/**/*.py",             # RAG pipeline
    "src/prompts/**/*.py",         # Prompt construction logic
    "eval/golden-set/**/*.json",   # Golden set updates
    "eval/golden-set/**/*.yaml",
    "requirements.txt",            # Dep changes may affect model behavior
    "pyproject.toml",
]

def get_changed_files(base_branch: str = "main") -> list[str]:
    """Get list of files changed in current branch vs base."""
    result = subprocess.run(
        ["git", "diff", "--name-only", f"origin/{base_branch}...HEAD"],
        capture_output=True, text=True,
    )
    return result.stdout.strip().split("\n")

def should_run_eval(changed_files: list[str]) -> tuple[bool, list[str]]:
    """
    Returns (should_run, matched_patterns).
    True if any changed file matches an AI-related pattern.
    """
    import fnmatch
    matched = []
    for pattern in AI_CHANGE_PATTERNS:
        if any(fnmatch.fnmatch(f, pattern) for f in changed_files):
            matched.append(pattern)
    return len(matched) > 0, matched

if __name__ == "__main__":
    files = get_changed_files()
    run_eval, matches = should_run_eval(files)
    print(f"AI-related changes: {run_eval}")
    if matches:
        print(f"Matched patterns: {matches}")
```

### GitHub Actions Integration

```yaml
# .github/workflows/ai-quality-gate.yml
name: AI Quality Gate

on:
  pull_request:
    branches: [main]

jobs:
  detect-ai-changes:
    runs-on: ubuntu-latest
    outputs:
      has_ai_changes: ${{ steps.detect.outputs.has_ai_changes }}
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0    # Need full history for diff
      - id: detect
        run: |
          result=$(python eval/change_detector.py)
          echo "has_ai_changes=$([[ "$result" == *"True"* ]] && echo "true" || echo "false")" >> $GITHUB_OUTPUT

  eval-quality-gate:
    needs: detect-ai-changes
    if: needs.detect-ai-changes.outputs.has_ai_changes == 'true'
    runs-on: ubuntu-latest
    timeout-minutes: 15    # Fail fast if eval hangs

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements.txt promptfoo deepeval

      - name: Load eval cache
        uses: actions/cache@v4
        with:
          path: .eval-cache/
          key: eval-cache-${{ hashFiles('eval/golden-set/**') }}
          restore-keys: eval-cache-

      - name: Run offline eval suite
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          python eval/run_eval_suite.py \
            --golden-set eval/golden-set/ \
            --output eval/results/pr-${{ github.event.number }}.json \
            --cache-dir .eval-cache/ \
            --max-cost 2.00     # Budget guard: fail if evals would cost > $2

      - name: Run quality gates
        run: |
          python eval/quality_gate.py \
            --results eval/results/pr-${{ github.event.number }}.json \
            --baseline eval/baselines/main.json \
            --max-regression 5.0

      - name: Post results as PR comment
        if: always()    # Post even on failure
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const results = JSON.parse(fs.readFileSync(
              'eval/results/pr-${{ github.event.number }}.json'
            ));
            const body = formatEvalReport(results);
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: body,
            });
```

### Promptfoo Configuration for PR Evals

```yaml
# eval/promptfooconfig.yaml
description: "PR quality gate eval"

providers:
  - openai:gpt-4o-mini

prompts:
  - file://prompts/support_agent/v{{ env.PROMPT_VERSION }}.yaml

tests:
  - file://eval/golden-set/billing.yaml
  - file://eval/golden-set/technical.yaml
  - file://eval/golden-set/safety.yaml

defaultTest:
  assert:
    - type: llm-rubric
      value: "The response directly addresses the user's question without fabricating policies or prices."
      threshold: 0.8
    - type: javascript
      value: |
        const data = JSON.parse(output);
        return data.hasOwnProperty('action') && data.hasOwnProperty('next_step');
      threshold: 1.0    # Format compliance: 100% required

sharing: false    # Don't share results to promptfoo.app in CI
```

### PR Gate Best Practices

| Practice | Why |
|----------|-----|
| Run on every AI-related PR, not every PR | Cost control; focus attention where it matters |
| Post results as a PR comment | Reviewers see quality impact without running locally |
| Compare against main branch baseline | Relative regression detection, not just absolute thresholds |
| Cache eval results for unchanged cases | Skip re-evaluation for cases unaffected by the change |
| Set a time limit (15 min max) | Slow evals kill developer velocity |
| Budget cap per PR ($2–5 max) | Prevents runaway eval cost from large golden sets |
| Allow emergency override with 2-engineer approval | Critical hotfixes cannot wait for eval fixes |

---

## Canary Deployments for AI Changes

A canary routes a small percentage of production traffic to the new version while monitoring for quality differences. For AI applications, "quality" must be measured—not just error rates.

### Canary Architecture

```python
import hashlib
import asyncio
from dataclasses import dataclass

@dataclass
class AIVersion:
    version_id: str
    prompt_name: str
    prompt_version: int
    model: str
    retrieval_config: dict | None = None


class AICanaryRouter:
    """
    Routes requests to stable or canary AI version.
    Deterministic per request_id: same request always goes to same version.
    """

    def __init__(self, stable: AIVersion, canary: AIVersion,
                 canary_percentage: float = 5.0):
        self.stable = stable
        self.canary = canary
        self._canary_pct = canary_percentage
        self._metrics: dict[str, list[float]] = {
            "stable_quality": [], "canary_quality": [],
            "stable_latency_ms": [], "canary_latency_ms": [],
            "stable_cost_usd": [], "canary_cost_usd": [],
        }

    def route(self, request_id: str) -> tuple[AIVersion, str]:
        """Deterministically route to stable or canary."""
        bucket = int(hashlib.sha256(request_id.encode()).hexdigest(), 16) % 100
        if bucket < self._canary_pct:
            return self.canary, "canary"
        return self.stable, "stable"

    def record_metric(self, env: str, metric: str, value: float):
        key = f"{env}_{metric}"
        if key in self._metrics:
            self._metrics[key].append(value)

    def evaluate_canary_health(self, min_samples: int = 200) -> dict:
        """
        Determine if the canary is healthy enough to promote or needs rollback.
        """
        canary_quality = self._metrics["canary_quality"]
        stable_quality = self._metrics["stable_quality"]
        canary_latency = self._metrics["canary_latency_ms"]
        stable_latency = self._metrics["stable_latency_ms"]

        if len(canary_quality) < min_samples:
            return {
                "action": "hold",
                "reason": f"Insufficient canary samples: {len(canary_quality)}/{min_samples}",
            }

        canary_q = sum(canary_quality) / len(canary_quality)
        stable_q = sum(stable_quality) / len(stable_quality)
        quality_regression = (stable_q - canary_q) / max(stable_q, 0.01)

        canary_lat = sum(canary_latency) / len(canary_latency)
        stable_lat = sum(stable_latency) / len(stable_latency)
        latency_increase = (canary_lat - stable_lat) / max(stable_lat, 1)

        failures = []
        if quality_regression > 0.05:     # > 5% quality regression
            failures.append(f"Quality regression: {quality_regression:.1%}")
        if latency_increase > 0.20:       # > 20% latency increase
            failures.append(f"Latency increase: {latency_increase:.1%}")

        return {
            "action": "rollback" if failures else "promote",
            "failures": failures,
            "canary_quality": round(canary_q, 3),
            "stable_quality": round(stable_q, 3),
            "canary_latency_ms": round(canary_lat, 1),
            "stable_latency_ms": round(stable_lat, 1),
            "n_canary_samples": len(canary_quality),
        }
```

### Automated Canary Promotion Schedule

```python
CANARY_STAGES = [
    {"pct": 1,   "min_duration_hours": 2,  "min_samples": 50,
     "criteria": ["error_rate", "timeout_rate"]},
    {"pct": 5,   "min_duration_hours": 8,  "min_samples": 200,
     "criteria": ["quality_score", "error_rate"]},
    {"pct": 20,  "min_duration_hours": 24, "min_samples": 500,
     "criteria": ["quality_score", "latency_p99", "cost_per_request"]},
    {"pct": 50,  "min_duration_hours": 24, "min_samples": 1000,
     "criteria": ["all"]},
    {"pct": 100, "min_duration_hours": 0,  "min_samples": 0,
     "criteria": []},    # Final promotion
]

CANARY_GATES = {
    "quality_score": {
        "canary_min": 0.82,
        "max_regression_vs_stable": 0.05,    # At most 5% worse than stable
    },
    "error_rate": {
        "canary_max": 0.02,
        "max_increase_vs_stable": 0.01,      # At most 1pp higher than stable
    },
    "latency_p99_ms": {
        "canary_max": 6000,
        "max_increase_vs_stable": 0.20,      # At most 20% higher than stable
    },
    "cost_per_request_usd": {
        "canary_max": 0.05,
        "max_increase_vs_stable": 0.25,      # At most 25% more expensive
    },
    "faithfulness_score": {
        "canary_min": 0.80,
        "max_regression_vs_stable": 0.05,
    },
}

def evaluate_canary_gates(canary_metrics: dict, stable_metrics: dict) -> dict:
    """
    Check all canary gate criteria.
    Returns {"action": "promote" | "rollback" | "hold", "failures": [...]}
    """
    failures = []

    for metric, gates in CANARY_GATES.items():
        canary_val = canary_metrics.get(metric)
        stable_val = stable_metrics.get(metric)

        if canary_val is None:
            continue  # Metric not available yet; skip

        if "canary_min" in gates and canary_val < gates["canary_min"]:
            failures.append(f"{metric}: {canary_val:.3f} below minimum {gates['canary_min']}")

        if "canary_max" in gates and canary_val > gates["canary_max"]:
            failures.append(f"{metric}: {canary_val:.3f} above maximum {gates['canary_max']}")

        if stable_val and stable_val > 0:
            if "max_regression_vs_stable" in gates:
                regression = (stable_val - canary_val) / stable_val
                if regression > gates["max_regression_vs_stable"]:
                    failures.append(
                        f"{metric}: {regression:.1%} regression vs stable "
                        f"({canary_val:.3f} vs {stable_val:.3f})"
                    )
            if "max_increase_vs_stable" in gates:
                increase = (canary_val - stable_val) / stable_val
                if increase > gates["max_increase_vs_stable"]:
                    failures.append(
                        f"{metric}: {increase:.1%} increase vs stable "
                        f"({canary_val:.3f} vs {stable_val:.3f})"
                    )

    return {
        "action": "rollback" if failures else "promote",
        "failures": failures,
        "gates_checked": list(CANARY_GATES.keys()),
    }
```

---

## Shadow Traffic

Shadow traffic (also called dark launching) sends a copy of production requests to the new version without returning its response to the user. The user always receives the stable response; you evaluate the new version's output offline.

Shadow traffic is the safest way to validate a new AI version against real production data:
- **Zero user risk**: The new version cannot degrade user experience
- **Real production data**: Evaluates on actual query distributions, not just golden sets
- **Latency-insensitive**: The shadow version can be slower without affecting users
- **Run before any canary exposure**: Validate for days before routing real users

```python
import asyncio
import time
from dataclasses import dataclass

@dataclass
class ShadowComparison:
    request_id: str
    input: str
    stable_response: str
    candidate_response: str
    stable_latency_ms: float
    candidate_latency_ms: float
    output_changed: bool
    length_ratio: float    # candidate / stable
    quality_delta: float | None   # From LLM judge


class ShadowTrafficHandler:
    """
    Handles requests by returning stable response to user
    while evaluating candidate in the background.
    """

    def __init__(self, stable_handler, candidate_handler, eval_recorder):
        self.stable = stable_handler
        self.candidate = candidate_handler
        self.recorder = eval_recorder

    async def handle(self, request: dict) -> dict:
        """
        Returns stable response immediately.
        Fires-and-forgets candidate evaluation in background.
        """
        start = time.time()
        stable_response = await self.stable.handle(request)
        stable_latency = (time.time() - start) * 1000

        # Fire and forget — do not await
        asyncio.create_task(
            self._shadow_eval(request, stable_response, stable_latency)
        )

        return stable_response    # User always gets stable response

    async def _shadow_eval(
        self, request: dict, stable_response: dict, stable_latency: float
    ):
        """Run candidate and compare. Runs async after response is sent."""
        try:
            start = time.time()
            candidate_response = await self.candidate.handle(request)
            candidate_latency = (time.time() - start) * 1000

            comparison = self._compare_responses(
                request, stable_response, candidate_response,
                stable_latency, candidate_latency,
            )

            # Record for analysis (async, non-blocking)
            await self.recorder.log(comparison)

        except Exception as e:
            await self.recorder.log_error(
                request_id=request.get("id"),
                error=str(e),
                error_type=type(e).__name__,
            )

    def _compare_responses(
        self, request: dict, stable: dict, candidate: dict,
        stable_latency: float, candidate_latency: float,
    ) -> ShadowComparison:
        stable_text = stable.get("text", "")
        candidate_text = candidate.get("text", "")

        return ShadowComparison(
            request_id=request.get("id", ""),
            input=request.get("input", ""),
            stable_response=stable_text,
            candidate_response=candidate_text,
            stable_latency_ms=stable_latency,
            candidate_latency_ms=candidate_latency,
            output_changed=stable_text != candidate_text,
            length_ratio=len(candidate_text) / max(len(stable_text), 1),
            quality_delta=None,   # Run LLM judge async on sample of comparisons
        )
```

### Shadow Traffic Analysis

```python
def analyze_shadow_traffic(comparisons: list[ShadowComparison]) -> dict:
    """
    Aggregate shadow traffic comparison results.
    Run this daily to decide whether to proceed with canary or hold.
    """
    n = len(comparisons)
    if n == 0:
        return {"status": "no_data"}

    changed_pct = sum(1 for c in comparisons if c.output_changed) / n

    latency_deltas = [c.candidate_latency_ms - c.stable_latency_ms for c in comparisons]
    avg_latency_delta = sum(latency_deltas) / n
    p95_latency_delta = sorted(latency_deltas)[int(n * 0.95)]

    length_ratios = [c.length_ratio for c in comparisons]
    avg_length_ratio = sum(length_ratios) / n

    # Quality delta (only for cases where LLM judge ran)
    judged = [c for c in comparisons if c.quality_delta is not None]
    avg_quality_delta = (
        sum(c.quality_delta for c in judged) / len(judged)
        if judged else None
    )

    return {
        "n_comparisons": n,
        "output_changed_pct": round(changed_pct, 3),
        "avg_latency_delta_ms": round(avg_latency_delta, 1),
        "p95_latency_delta_ms": round(p95_latency_delta, 1),
        "avg_length_ratio": round(avg_length_ratio, 3),
        "avg_quality_delta": round(avg_quality_delta, 3) if avg_quality_delta else None,
        "n_judged": len(judged),
        "recommendation": (
            "Proceed to canary" if avg_quality_delta and avg_quality_delta >= -0.05
            else "Hold: quality degradation detected in shadow traffic"
        ),
    }
```

---

## The Complete CI/CD Pipeline

```
PR Stage:
  Code change
    → Unit tests (existing)
    → Detect AI-related changes
    → Run offline eval suite against golden set (skip if no AI changes)
    → Compare vs main branch baseline
    → Post results as PR comment
    → Block merge if hard gates fail

Staging Stage:
  Deploy to staging
    → Run full regression suite vs prior baseline
    → Run shadow traffic against staging for 48 hours
    → Run load test to verify latency under simulated traffic

Canary Stage:
  1% traffic → monitor error rate and timeouts for 2 hours
  5% traffic → monitor quality score for 8 hours
  20% traffic → full quality + cost + latency for 24 hours
  50% traffic → statistical comparison for 24 hours
  → Automated promotion or rollback based on gates

Production:
  100% traffic
    → Continuous monitoring (Lesson 6)
    → Feedback loop → new golden set cases
    → Weekly automated eval report
```

### Cost Management in CI/CD

| Strategy | Savings |
|----------|---------|
| Only run evals on AI-related PRs | Skip 70–80% of PR eval runs |
| Cache results for unchanged golden cases | Skip re-evaluation for cases not affected by the change |
| Use gpt-4o-mini for screening, gpt-4.1 for failures only | 10× cost reduction for first-pass evaluation |
| Nightly full suite, PR runs critical 50 cases only | PR costs $0.25 instead of $2.50 |
| Budget cap per PR ($2–5) | Alert and skip if eval would exceed budget |
| Batch API for nightly full suite | 50% discount on overnight runs |

---

## Edge Cases and Misconceptions

**"Evals in CI slow down development too much."**
A 50-case eval suite with gpt-4o-mini takes under 3 minutes and costs under $0.50. The slowdown is comparable to running integration tests. The alternative—discovering a quality regression 3 days after deploy—costs far more in investigation time and user impact.

**"Shadow traffic requires maintaining two production-grade versions."**
Shadow traffic runs the candidate version as a background process after the response is already sent. The candidate does not need to meet production reliability standards—it only needs to produce responses you can evaluate. Run it on a single secondary instance.

**"Canary is too conservative—we can validate with A/B testing instead."**
A/B testing deliberately exposes both versions to users to measure a difference. Canary deployment is about *validating* one version before full exposure. They serve different purposes. Use canary for deployment safety; use A/B testing for product decisions (Lesson 5 of Module 10).

**"Our golden set covers everything; we don't need shadow traffic."**
No golden set covers everything. Production users ask questions you never anticipated, use phrasing that differs from your test cases, and mix intents in ways your test suite doesn't model. Shadow traffic is specifically valuable because it tests on the real production distribution.

---

## Key Takeaways

- AI CI/CD requires eval gates at every stage: PR gate catches regressions before merge; staging gate prevents bad deploys; canary gate validates on real traffic before full exposure
- Change detection ensures evals run only on AI-relevant PRs, keeping CI fast and budget under control
- Canary deployments for AI applications require quality measurement (LLM judge or user feedback) in addition to standard error rate and latency monitoring
- Shadow traffic is the safest pre-canary validation: evaluates against real production data with zero user exposure risk
- Manage CI eval costs with change detection, result caching, tiered models, and per-PR budget caps
- Close the loop: production failures and negative feedback flow back into the golden set, making future eval suites more predictive

---

## Further Reading

- [Continuous Integration for Machine Learning](https://arxiv.org/abs/2006.10594) — Systematic treatment of CI/CD patterns adapted for ML systems
- [Reliable Machine Learning](https://arxiv.org/abs/2004.11698) — Google's framework for ML system reliability including deployment gates
- [The ML Test Score: A Rubric for ML Production Readiness](https://arxiv.org/abs/1911.00617) — Comprehensive production readiness checklist for ML/AI systems
- [Promptfoo CI/CD documentation](https://promptfoo.dev/docs/integrations/github-action/) — Official GitHub Actions integration for Promptfoo eval gates

---

## Next Lesson

**Lesson 6: Production Monitoring & Alerts** — Set up continuous quality monitoring, detect distribution drift, build user feedback loops, and design alerting systems that catch problems before users notice.
