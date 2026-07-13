---
title: Permissions and Safety in the Harness
description: >-
  Implement allowlists, human-in-the-loop approval gates, and token/cost/step
  budgets as harness policies that run before any tool executes
duration: 50 min
difficulty: advanced
has_code: true
module: module-18
---
# Permissions and Safety in the Harness

## Prerequisites

- [Lesson 2 — Agent Loop and State](02-agent-loop-and-state.md): harness loop and step budget
- [Lesson 3 — Tools and Function Calling](03-tools-and-function-calling.md): tool registry and sandbox
- Familiarity with Python dataclasses, enums, and basic OS path operations
- General awareness of prompt injection as a threat category

---

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand why safety is a harness concern, not a model alignment concern | 50 min | Advanced |
| Implement tool allowlists and denylists with a default-deny posture | | |
| Build human-in-the-loop approval gates with actionable UI context | | |
| Apply step, token, cost, and wall-clock budgets as termination policies | | |
| Validate tool arguments — not just tool names — to catch path traversal and command injection | | |
| Design a layered defense-in-depth policy that runs before, during, and after execution | | |

---

## Intuition First

Safety guardrails in model alignment training reduce the probability that a model *wants* to do something harmful. But they do not stop an agent from *accidentally* deleting the wrong file if the tool is available and the context suggests it. Intent alignment is not execution control.

Consider: an agent is asked to "clean up old files in the temp folder." The model has been trained to be helpful. A misunderstood instruction leads it to call `delete_files("/data/production/")` instead of `/tmp/`. No amount of RLHF prevents this if the delete tool is registered and allowed.

This is why safety is a **harness concern**: the harness controls which tools can execute, under what conditions, with what argument constraints, and at what cost. The model decides *what* to do; the harness decides *whether it may*.

Defense in depth means you have multiple independent gates:

```
User prompt → [Input policy] → [Tool allowlist] → [Human approval] → [Budget check] → [Arg validation] → Execute
```

Each gate is independently effective. All gates together make it very hard for any single failure to cause harm.

---

## Safety Is a Harness Job

Model alignment and safety training reduce harmful outputs — but they do not stop an agent from **deleting a file** if the tool is available and the prompt says so. Runtime safety belongs in the **harness**, not in the system prompt.

```
                    ┌─────────────────────────────────┐
  User prompt ─────▶│  Layer 1: Input policy          │
                    │  (block jailbreaks, PII rules)  │
                    └───────────────┬─────────────────┘
                                    ▼
                    ┌─────────────────────────────────┐
                    │  Layer 2: Tool allowlist          │
                    │  (what CAN be called)             │
                    └───────────────┬─────────────────┘
                                    ▼
                    ┌─────────────────────────────────┐
                    │  Layer 3: Argument validation     │
                    │  (path scope, command blocklist)  │
                    └───────────────┬─────────────────┘
                                    ▼
                    ┌─────────────────────────────────┐
                    │  Layer 4: Human approval          │
                    │  (sensitive actions)            │
                    └───────────────┬─────────────────┘
                                    ▼
                    ┌─────────────────────────────────┐
                    │  Layer 5: Budget limits           │
                    │  (steps, tokens, cost, time)      │
                    └───────────────┬─────────────────┘
                                    ▼
                              Tool executes
```

[M16 · AI Safety](../../../production/module-16-ai-safety-ethics/index.md) covers ethics and red teaming. This lesson covers the **runtime controls** you implement in code.

---

## Allowlists and Denylists

The simplest effective guard: **default deny** for tools.

```python
from dataclasses import dataclass, field

@dataclass
class PermissionSet:
    allowed_tools: set[str] = field(default_factory=set)
    denied_tools: set[str] = field(default_factory=set)
    require_approval: set[str] = field(default_factory=set)

    def can_invoke(self, tool_name: str) -> tuple[bool, str | None]:
        if tool_name in self.denied_tools:
            return False, f"Tool '{tool_name}' is explicitly denied."
        if self.allowed_tools and tool_name not in self.allowed_tools:
            return False, f"Tool '{tool_name}' is not in the allowlist."
        if tool_name in self.require_approval:
            return False, "APPROVAL_REQUIRED"
        return True, None

# Read-only research agent
READ_ONLY = PermissionSet(
    allowed_tools={"search_web", "read_file", "list_directory"},
    denied_tools={"write_file", "run_terminal", "send_email"},
)

# Coding agent with gated writes
CODING_AGENT = PermissionSet(
    allowed_tools={"read_file", "write_file", "run_terminal", "search_web"},
    require_approval={"write_file", "run_terminal"},
)

# Support agent — can look up and reply, but not refund without approval
SUPPORT_AGENT = PermissionSet(
    allowed_tools={"lookup_account", "search_kb", "send_reply", "issue_refund"},
    require_approval={"send_reply", "issue_refund"},
)
```

| Agent profile | Allowed | Requires approval |
|---------------|---------|-----------------|
| **Research** | search, read | — |
| **Coding** | read, write, terminal | write, terminal |
| **Support** | lookup_account, send_reply, refund | send_reply, refund |

!!! tip "Denylist wins over allowlist"
    If a tool appears in both sets, deny. Explicit blocks prevent accidents when someone later expands the allowlist.

### Worked Example — Allowlist Check

```python
perms = CODING_AGENT

# read_file is allowed, no approval needed
allowed, reason = perms.can_invoke("read_file")
print(allowed, reason)  # True, None

# write_file requires approval
allowed, reason = perms.can_invoke("write_file")
print(allowed, reason)  # False, "APPROVAL_REQUIRED"

# send_email is not in allowlist at all
allowed, reason = perms.can_invoke("send_email")
print(allowed, reason)  # False, "Tool 'send_email' is not in the allowlist."
```

---

## Human-in-the-Loop

When the harness returns `APPROVAL_REQUIRED`, pause the loop, surface the pending action to the user, and resume only on explicit consent.

```python
import os
import time
from dataclasses import dataclass
from enum import Enum

class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"

@dataclass
class PendingAction:
    tool_name: str
    arguments: dict
    reason: str
    status: ApprovalStatus = ApprovalStatus.PENDING

class HumanInTheLoop:
    def __init__(self, approval_callback):
        self.approval_callback = approval_callback  # UI hook
        self.pending: PendingAction | None = None

    def request(self, tool_name: str, arguments: dict, reason: str) -> bool:
        self.pending = PendingAction(tool_name, arguments, reason)
        approved = self.approval_callback(self.pending)
        self.pending.status = (
            ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        )
        return approved

def execute_with_permissions(
    registry, sandbox, permissions: PermissionSet,
    hitl: HumanInTheLoop, name: str, args: dict,
) -> str:
    allowed, reason = permissions.can_invoke(name)
    if not allowed:
        if reason == "APPROVAL_REQUIRED":
            summary = f"Agent wants to call {name}({args})"
            if not hitl.request(name, args, summary):
                return registry._error(
                    "USER_REJECTED",
                    f"User declined {name}. Try a different approach.",
                )
        else:
            return registry._error("NOT_ALLOWED", reason)

    return sandbox.run(registry, name, args)
```

### What to Show the User

The approval prompt must give the user enough context to make an informed decision. Vague prompts ("Agent wants to call write_file") cause approval fatigue — users click approve without reading.

**Good approval prompt for a file write:**

```
⚠ Agent wants to write a file

File: /workspace/src/auth.py
Action: MODIFY (not create — file exists)
Preview of changes:
  Line 47: - raise ValueError("unauthenticated")
  Line 47: + return AnonymousUser()

Reason given by agent: "Replacing exception with anonymous user pattern
as requested."

[Approve] [Reject] [Approve all file writes for this session]
```

**Good approval prompt for a terminal command:**

```
⚠ Agent wants to run a terminal command

Command: git push origin feature/auth-refactor

This will: push local commits to the remote branch.
Note: This cannot be undone without force-pushing.

[Approve] [Reject]
```

### UX Patterns

| Pattern | Use when |
|---------|----------|
| **Per-action prompt** | Destructive or irreversible ops (delete, send, purchase) |
| **Session grant** | "Allow terminal commands for this task" (one-time broad grant) |
| **Diff preview** | File writes — show the patch before applying |
| **Timeout → deny** | No response within 30s → auto-reject, not auto-approve |
| **Rate limit prompts** | Cap at N approval prompts per session to prevent prompt fatigue |

!!! warning "Auto-approve is the most dangerous UX decision"
    Users who see too many approval prompts start clicking "approve" without reading. Limit approval prompts to genuinely sensitive actions, and clearly explain consequences. A "cancel" button that stops the entire agent run is always better than silent auto-approval.

---

## Budget Limits

Budgets are **termination policies** (Lesson 1) enforced on every step:

```python
@dataclass
class RunBudget:
    max_steps: int = 20
    max_tokens: int = 200_000
    max_cost_usd: float = 1.00
    max_wall_seconds: float = 600.0

class BudgetEnforcer:
    def __init__(self, budget: RunBudget):
        self.budget = budget
        self.start_time = time.time()

    def check(self, state) -> tuple[bool, str | None]:
        elapsed = time.time() - self.start_time

        if state.step >= self.budget.max_steps:
            return False, f"Step budget exhausted ({self.budget.max_steps} steps)"
        if state.total_tokens >= self.budget.max_tokens:
            return False, f"Token budget exhausted ({self.budget.max_tokens:,} tokens)"
        if state.total_cost_usd >= self.budget.max_cost_usd:
            return False, f"Cost budget exhausted (${self.budget.max_cost_usd:.2f})"
        if elapsed >= self.budget.max_wall_seconds:
            return False, f"Wall-clock timeout ({self.budget.max_wall_seconds:.0f}s)"

        return True, None

    def remaining(self, state) -> dict:
        return {
            "steps_left": self.budget.max_steps - state.step,
            "cost_left_usd": round(self.budget.max_cost_usd - state.total_cost_usd, 4),
            "tokens_left": self.budget.max_tokens - state.total_tokens,
            "time_left_s": round(self.budget.max_wall_seconds - (time.time() - self.start_time), 1),
        }
```

Inject budget status into perceive so the model paces itself:

```python
def perceive_with_budget(self, state, enforcer: BudgetEnforcer):
    messages = self.perceive(state)
    remaining = enforcer.remaining(state)
    budget_note = (
        f"\n\nBudget remaining: {remaining['steps_left']} steps, "
        f"${remaining['cost_left_usd']:.4f} cost, "
        f"{remaining['time_left_s']:.0f}s wall time."
    )
    messages[0]["content"] += budget_note
    return messages
```

**Worked example — cost accumulation:**

```
Step 1: gpt-4o, 1,800 input + 200 output → ~$0.0050
Step 2: gpt-4o, 2,400 input + 150 output → ~$0.0064
Step 3: gpt-4o, 3,100 input + 300 output → ~$0.0086
...
Step 10: accumulated ~$0.05 for a 10-step run
```

At 10,000 runs/day, a $0.05 avg cost = $500/day. Budget enforcement prevents runaway runs (e.g., infinite tool-retry loops) from inflating that average.

!!! warning "Silent budget exhaustion frustrates users"
    When a budget trips, return a **partial answer** with what was accomplished and which limit was hit — not a generic error or empty response.

---

## Scoped Permissions (Filesystem and Network)

Beyond tool names, constrain **what arguments are valid**:

```python
import os
from pathlib import Path

ALLOWED_PATHS = ["/workspace", "/tmp/agent"]
BLOCKED_COMMANDS = ["rm -rf", "curl | sh", "sudo", "chmod 777", "dd if="]

def validate_path(path: str, allowed_roots: list[str] = ALLOWED_PATHS) -> bool:
    """Prevent path traversal attacks — e.g., ../../etc/passwd."""
    try:
        resolved = str(Path(path).resolve())
    except (ValueError, OSError):
        return False
    return any(resolved.startswith(os.path.realpath(root)) for root in allowed_roots)

def validate_command(command: str, blocked: list[str] = BLOCKED_COMMANDS) -> bool:
    """Block high-risk shell patterns."""
    cmd_lower = command.lower()
    return not any(b.lower() in cmd_lower for b in blocked)

def validate_tool_args(name: str, args: dict) -> str | None:
    """Return an error string if arguments violate policy; None if valid."""
    if name == "read_file" and "path" in args:
        if not validate_path(args["path"]):
            return (
                f"Path '{args['path']}' is outside allowed directories "
                f"{ALLOWED_PATHS}. Use a path within the workspace."
            )

    if name == "write_file" and "path" in args:
        if not validate_path(args["path"]):
            return f"Cannot write outside {ALLOWED_PATHS}."

    if name == "run_terminal" and "command" in args:
        if not validate_command(args["command"]):
            return (
                f"Command blocked by policy. Blocked patterns: {BLOCKED_COMMANDS}. "
                "Use safer alternatives."
            )

    return None  # no policy violation
```

Argument-level policy catches cases where the tool is allowed but the **target** is not — e.g., `read_file("/etc/passwd")`.

### Input Policy (Layer 1)

Before the model even processes the user's message, filter for jailbreak patterns and PII:

```python
import re

JAILBREAK_PATTERNS = [
    r"ignore (all )?previous instructions",
    r"you are now (in )?DAN mode",
    r"pretend you (have no|are not)",
    r"system prompt override",
]

PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),   # SSN
    re.compile(r"\b4[0-9]{12}(?:[0-9]{3})?\b"),  # Visa card
]

def check_input_policy(user_message: str) -> str | None:
    """Return an error if input violates policy; None if clean."""
    msg_lower = user_message.lower()
    for pattern in JAILBREAK_PATTERNS:
        if re.search(pattern, msg_lower):
            return "Input contains a pattern associated with prompt injection. Please rephrase."

    for pattern in PII_PATTERNS:
        if pattern.search(user_message):
            return "Input appears to contain sensitive personal data. Please remove before proceeding."

    return None
```

---

## Putting It Together

```python
def harness_step(state, permissions, enforcer, hitl, registry, sandbox):
    # Layer 5: budget check first (fastest)
    ok, reason = enforcer.check(state)
    if not ok:
        state.status = "finished"
        state.scratchpad["stop_reason"] = reason
        state.scratchpad["partial_note"] = (
            "Run terminated before completion. "
            f"Limit reached: {reason}. "
            "Here is what was accomplished so far: see tool_results."
        )
        return state

    response = model.complete(perceive_with_budget(state, enforcer))

    for call in response.tool_calls or []:
        # Layer 3: argument-level validation
        arg_error = validate_tool_args(call.name, call.arguments)
        if arg_error:
            state.append_tool_result(ToolResult(
                tool_call_id=call.id, name=call.name,
                arguments=call.arguments,
                output=registry._error("POLICY_VIOLATION", arg_error),
                duration_ms=0, success=False,
            ))
            continue

        # Layers 2 + 4: allowlist + human approval
        output = execute_with_permissions(
            registry, sandbox, permissions, hitl, call.name, call.arguments
        )
        state.append_tool_result(ToolResult(
            tool_call_id=call.id, name=call.name,
            arguments=call.arguments, output=output,
            duration_ms=0, success='"ok": true' in output,
        ))

    state.step += 1
    return state
```

---

## Audit Logging

Every permissions check and approval decision should be logged for compliance and post-incident review:

```python
import json
import logging
from datetime import datetime, timezone

audit_log = logging.getLogger("agent.audit")

def log_permission_decision(
    run_id: str,
    tool_name: str,
    arguments: dict,
    decision: str,  # "APPROVED", "DENIED", "APPROVAL_REQUIRED", "USER_REJECTED"
    policy_reason: str | None,
    user_id: str | None = None,
):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "tool": tool_name,
        "decision": decision,
        "policy_reason": policy_reason,
        "user_id": user_id,
        # Redact sensitive arg values; keep keys for debugging
        "argument_keys": list(arguments.keys()),
    }
    audit_log.info(json.dumps(entry))
```

Audit logs answer: *Who approved this? When? What arguments were passed? Which policy rule triggered?* — essential for any regulated industry deployment.

---

## Common Misconceptions

**"The system prompt can enforce safety."** System prompts influence the model's intent but have no authority over tool execution. A model that decides to call `delete_file` will do so regardless of what the system prompt says, unless the harness blocks it.

**"Allowlists are too restrictive."** Default deny is not about limiting functionality — it's about explicit intentionality. If a tool is needed, add it to the allowlist. If you're not sure whether a tool is needed, it shouldn't be there.

**"Human approval breaks the user experience."** Human approval breaks the experience when the prompts are vague, frequent, or badly timed. Well-designed approval prompts (with previews, clear consequences, session grants) integrate naturally into agent workflows and build user trust.

**"Budget limits are just for cost control."** Budgets also prevent security issues. An agent caught in an infinite tool-call loop (often from prompt injection exploiting a retrieval tool) will exhaust its step budget rather than running forever.

**"Argument validation is redundant with schema validation."** JSON Schema validation ensures arguments are the right *type*. Argument policy validation ensures they are the right *value* — a string path like `/etc/passwd` passes type validation but fails policy validation.

---

## Production Tips

- **Start with the most restrictive permissions and relax.** It's much easier to add tools to an allowlist after validating they're needed than to remove tools after users have built workflows depending on them.
- **Make approval prompts actionable.** Show the tool name, the exact arguments, a human-readable summary of what the action does, and the expected consequence. "Agent wants to do something" is never enough.
- **Log approval decisions durably.** Store approval events in a separate, write-once log — not just application logs that may be rotated. Compliance requirements often mandate this.
- **Test all policy paths explicitly.** Write unit tests for: tool allowed, tool denied, tool requires approval (approved), tool requires approval (rejected), arg policy violation, budget exceeded. These are off the happy path and easy to break silently.
- **Model the threat:** Who are your users? What tools are available? What's the worst-case misuse? For each answer, there's a harness policy that mitigates it.

---

## Key Takeaways

- **Harness policies** enforce safety before tools run — prompts alone are insufficient, model alignment alone is insufficient
- Use **allowlists** (default deny), **denylists** (explicit blocks), and **approval gates** for sensitive tools
- **Human-in-the-loop** pauses the loop with a clear action summary; well-designed prompts build trust, vague prompts cause fatigue
- **Budgets** on steps, tokens, cost, and wall-clock time prevent runaway runs and limit the blast radius of prompt injection
- Validate **tool arguments** (paths, commands) — not just tool names — to prevent path traversal and command injection
- **Audit every permission decision** for compliance, forensics, and incident review

---

## Related Papers

| Paper | Year | Key contribution |
|-------|------|-----------------|
| [Prompt Injection Attacks against LLM-integrated Applications](https://arxiv.org/abs/2306.05499) | 2023 | Taxonomy of prompt injection leading to unauthorized tool use |
| [AgentDojo: A Dynamic Environment to Evaluate Attacks and Defenses for LLM Agents](https://arxiv.org/abs/2406.13352) | 2024 | Benchmark for agent security — defense evaluation framework |
| [INJECAGENT: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents](https://arxiv.org/abs/2403.02691) | 2024 | Indirect injection via tool results (retrieval, web search) |
| [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) | 2022 | Model-level alignment as one layer in defense-in-depth |

---

## Further Reading

- [Awesome Harness Engineering](https://github.com/ai-boost/awesome-harness-engineering) — permission and policy patterns
- [Agents Towards Production](https://github.com/NirDiamant/agents-towards-production) — guardrails in deployed agents
- [M16 · Prompt Injection](../../../production/module-16-ai-safety-ethics/lessons/04-lesson-04.md) — input-side threats

---

## Next Lesson

**[Lesson 6: Observability in the Harness](06-observability-in-the-harness.md)** — Traces, spans, and structured logging for every agent step; integrating Langfuse and OpenTelemetry; debugging failed runs with trace replay.
