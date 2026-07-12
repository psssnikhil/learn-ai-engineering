---
title: Agent Trajectory Evals
description: >-
  Evaluate multi-step agent behavior with step-level and outcome-level metrics,
  tool call correctness, and reasoning chain validation
duration: 40 min
difficulty: advanced
has_code: false
module: module-19
---
# Agent Trajectory Evals

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Distinguish step-level vs outcome-level evaluation | 40 min | Advanced |
| Validate tool call correctness | | |
| Score agent reasoning trajectories | | |
| Build eval suites for multi-step agents | | |

---

## Why Agent Evals Are Different

A chatbot eval checks one input and one output. An **agent eval** checks an entire trajectory: a sequence of thoughts, tool calls, observations, and decisions that may span dozens of steps.

```
Chatbot eval:
  Input: "What's the weather in Tokyo?"
  Output: "It's 22°C and sunny."
  Check: Is the answer correct?

Agent eval:
  Input: "Book me the cheapest flight to Tokyo next Friday"
  Trajectory:
    Step 1: search_flights(origin="SFO", dest="NRT", date="2026-07-18")
    Step 2: OBSERVE → 3 results: $890, $1,200, $750
    Step 3: get_user_preferences() → economy, no layovers > 2
    Step 4: filter_results(max_layovers=2) → 2 results: $890, $750
    Step 5: book_flight(flight_id="UA837", price=750)
    Step 6: OBSERVE → Booking confirmed
  Check: Did it call the right tools? In the right order? With correct args? Did it achieve the goal?
```

A agent can produce the correct final answer via a completely wrong trajectory (lucky hallucination) or fail despite reasonable steps (bad luck with an API). You need to evaluate both.

---

## Step-Level vs Outcome-Level Evaluation

### Outcome-Level Evaluation

Did the agent achieve the goal? This is the minimum bar.

| Metric | Description |
|--------|-------------|
| **Task success rate** | Did the final state match the expected outcome? |
| **Answer correctness** | Is the final response factually correct? |
| **User goal completion** | Would a user consider their request fulfilled? |

```python
def outcome_eval(trajectory: dict, expected: dict) -> dict:
    """Evaluate whether the agent achieved the desired outcome."""
    final_state = trajectory["steps"][-1]

    checks = {
        "task_completed": final_state.get("status") == "success",
        "correct_answer": expected["answer"] in final_state.get("output", ""),
        "no_errors": all(s.get("error") is None for s in trajectory["steps"]),
    }

    return {
        "passed": all(checks.values()),
        "checks": checks,
    }
```

Outcome-level eval is necessary but insufficient. An agent that succeeds 90% of the time but calls `delete_database` on 5% of runs is not production-ready.

### Step-Level Evaluation

Did each step in the trajectory follow correct reasoning and tool usage?

| Metric | Description |
|--------|-------------|
| **Tool selection accuracy** | Did the agent call the right tool at each step? |
| **Argument correctness** | Were tool arguments valid and complete? |
| **Step ordering** | Were tools called in a logical sequence? |
| **Efficiency** | Did the agent solve the task in the minimum necessary steps? |
| **Recovery** | Did the agent handle errors and retry appropriately? |

```python
def step_level_eval(trajectory: dict, expected_steps: list[dict]) -> dict:
    """Compare actual trajectory steps against expected tool call sequence."""
    actual_tools = [s["tool"] for s in trajectory["steps"] if s.get("type") == "tool_call"]
    expected_tools = [s["tool"] for s in expected_steps]

    step_results = []
    for i, (actual, expected) in enumerate(zip(actual_tools, expected_tools)):
        step_results.append({
            "step": i + 1,
            "tool_match": actual == expected["tool"],
            "args_valid": validate_tool_args(actual, trajectory["steps"][i].get("args", {}), expected),
        })

    return {
        "tool_sequence_accuracy": sum(r["tool_match"] for r in step_results) / max(len(expected_tools), 1),
        "args_accuracy": sum(r["args_valid"] for r in step_results) / max(len(step_results), 1),
        "extra_steps": len(actual_tools) - len(expected_tools),
        "step_details": step_results,
    }
```

### When to Use Each

| Scenario | Outcome | Step-Level |
|----------|---------|------------|
| Simple Q&A agent | Primary | Optional |
| Tool-using agent (API calls) | Necessary | Primary |
| Multi-agent orchestration | Necessary | Primary |
| Autonomous coding agent | Necessary | Primary |
| Customer support agent | Both equally important | Both equally important |

---

## Tool Call Correctness

Tool call correctness is the most objective agent eval metric. Unlike answer quality, you can verify it deterministically.

### Three Dimensions of Tool Call Eval

```python
def evaluate_tool_call(actual: dict, expected: dict) -> dict:
    """Evaluate a single tool call across three dimensions."""
    return {
        # 1. Was the right tool selected?
        "tool_selection": actual["tool"] == expected["tool"],

        # 2. Were the arguments correct?
        "args_correct": all(
            actual["args"].get(k) == v
            for k, v in expected["args"].items()
        ),

        # 3. Was the call necessary? (no redundant or harmful calls)
        "necessary": actual["tool"] not in expected.get("forbidden_tools", []),
    }
```

### Argument Validation Strategies

| Strategy | When to Use | Example |
|----------|-------------|---------|
| **Exact match** | Deterministic args (IDs, dates) | `flight_id == "UA837"` |
| **Schema validation** | Structured args with types | `isinstance(price, int) and price > 0` |
| **Semantic match** | Natural language args | LLM judge: "Is this search query reasonable?" |
| **Range check** | Numeric params | `0 < temperature < 1` |
| **Enum check** | Fixed option sets | `status in ["pending", "confirmed", "cancelled"]` |

```python
TOOL_SCHEMAS = {
    "search_flights": {
        "required": ["origin", "destination", "date"],
        "types": {"origin": str, "destination": str, "date": str},
        "validators": {
            "date": lambda d: re.match(r"\d{4}-\d{2}-\d{2}", d) is not None,
            "origin": lambda o: len(o) == 3,  # IATA code
        },
    },
    "book_flight": {
        "required": ["flight_id", "passenger_name"],
        "types": {"flight_id": str, "passenger_name": str},
        "forbidden_combinations": [],  # e.g., never book without search first
    },
}

def validate_tool_args(tool_name: str, args: dict) -> list[str]:
    """Return list of validation errors, empty if valid."""
    schema = TOOL_SCHEMAS.get(tool_name)
    if not schema:
        return [f"Unknown tool: {tool_name}"]

    errors = []
    for field in schema["required"]:
        if field not in args:
            errors.append(f"Missing required arg: {field}")

    for field, expected_type in schema["types"].items():
        if field in args and not isinstance(args[field], expected_type):
            errors.append(f"{field}: expected {expected_type.__name__}, got {type(args[field]).__name__}")

    for field, validator in schema.get("validators", {}).items():
        if field in args and not validator(args[field]):
            errors.append(f"{field}: failed validation")

    return errors
```

### Forbidden Tool Sequences

Some tool call orderings are dangerous regardless of outcome:

```python
FORBIDDEN_SEQUENCES = [
    {"pattern": ["delete_*"], "without_prior": ["confirm_deletion"]},
    {"pattern": ["send_email"], "without_prior": ["draft_email", "review_draft"]},
    {"pattern": ["execute_sql"], "without_prior": ["validate_query"]},
]

def check_forbidden_sequences(trajectory: list[dict]) -> list[str]:
    """Detect dangerous tool call patterns in agent trajectory."""
    violations = []
    tools_called = [s["tool"] for s in trajectory if s.get("type") == "tool_call"]

    for rule in FORBIDDEN_SEQUENCES:
        for i, tool in enumerate(tools_called):
            if any(tool.startswith(p.replace("*", "")) for p in rule["pattern"]):
                prior_tools = tools_called[:i]
                if not any(p in prior_tools for p in rule["without_prior"]):
                    violations.append(f"{tool} called without prior {rule['without_prior']}")

    return violations
```

---

## Trajectory Scoring

Combine step-level and outcome-level metrics into a single trajectory score.

### Trajectory Eval Test Case

```python
agent_eval_case = {
    "id": "flight-booking-007",
    "input": "Book the cheapest economy flight from SFO to Tokyo next Friday",
    "expected_outcome": {
        "status": "success",
        "booking_confirmed": True,
        "max_price": 900,
    },
    "expected_trajectory": {
        "required_tools": ["search_flights", "book_flight"],
        "tool_order": ["search_flights", "book_flight"],
        "max_steps": 8,
        "forbidden_tools": ["cancel_all_bookings", "delete_user"],
    },
    "scoring_weights": {
        "outcome": 0.50,
        "tool_selection": 0.25,
        "arg_correctness": 0.15,
        "efficiency": 0.10,
    },
}
```

### Composite Score Calculation

```python
def score_trajectory(trajectory: dict, test_case: dict) -> dict:
    weights = test_case["scoring_weights"]

    # Outcome score
    outcome = outcome_eval(trajectory, test_case["expected_outcome"])
    outcome_score = 1.0 if outcome["passed"] else 0.0

    # Step-level scores
    step_eval = step_level_eval(trajectory, test_case["expected_trajectory"]["required_tools"])
    tool_score = step_eval["tool_sequence_accuracy"]
    args_score = step_eval["args_accuracy"]

    # Efficiency: penalize unnecessary steps
    max_steps = test_case["expected_trajectory"]["max_steps"]
    actual_steps = len([s for s in trajectory["steps"] if s.get("type") == "tool_call"])
    efficiency_score = max(0, 1 - (actual_steps - max_steps) / max_steps) if actual_steps > max_steps else 1.0

    # Forbidden sequence check (instant fail)
    violations = check_forbidden_sequences(trajectory["steps"])
    if violations:
        return {"composite": 0.0, "failed": True, "reason": "forbidden_tool_sequence", "violations": violations}

    composite = (
        weights["outcome"] * outcome_score
        + weights["tool_selection"] * tool_score
        + weights["arg_correctness"] * args_score
        + weights["efficiency"] * efficiency_score
    )

    return {
        "composite": composite,
        "passed": composite >= 0.8,
        "breakdown": {
            "outcome": outcome_score,
            "tool_selection": tool_score,
            "arg_correctness": args_score,
            "efficiency": efficiency_score,
        },
    }
```

---

## Building Agent Eval Suites

Use DeepEval's `ToolCorrectnessMetric` or Promptfoo JavaScript assertions on `toolsCalled` to validate agent behavior in CI.

Store trajectories alongside eval scores — when a case fails, the trace shows exactly where the agent went wrong.

---

## Key Takeaways

- Agent evals require both **outcome-level** (did it succeed?) and **step-level** (was each step correct?) evaluation
- **Tool call correctness** is the most objective agent metric — validate tool selection, arguments, and ordering
- Define **forbidden tool sequences** for safety-critical operations
- Use composite scoring with weighted breakdowns across outcome, tool selection, args, and efficiency
- Record full trajectories for debugging — the final answer alone hides reasoning failures
- DeepEval's `ToolCorrectnessMetric` and Promptfoo's JavaScript assertions cover most agent eval needs

---

## Next Lesson

**Lesson 5: CI/CD for AI Quality** — Integrate eval suites into PR checks, implement canary deployments, and use shadow traffic to validate changes before they reach users.
