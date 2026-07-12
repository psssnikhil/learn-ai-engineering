---
title: Permissions and Safety in the Harness
description: >-
  Implement allowlists, human-in-the-loop approval gates, and token/cost/step
  budgets as harness policies that run before any tool executes
duration: 35 min
difficulty: advanced
has_code: true
module: module-18
---
# Permissions and Safety in the Harness

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Enforce tool allowlists and denylists in the harness | 35 min | Advanced |
| Implement human-in-the-loop approval for sensitive actions | | |
| Apply step, token, and cost budgets as termination policies | | |
| Layer defense in depth: policy before model, before tool, after tool | | |

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
                    │  Layer 3: Human approval          │
                    │  (sensitive actions)            │
                    └───────────────┬─────────────────┘
                                    ▼
                    ┌─────────────────────────────────┐
                    │  Layer 4: Budget limits           │
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
```

| Agent profile | Allowed | Requires approval |
|---------------|---------|-----------------|
| **Research** | search, read | — |
| **Coding** | read, write, terminal | write, terminal |
| **Support** | lookup_account, send_reply | send_reply, refund |

!!! tip "Denylist wins over allowlist"
    If a tool appears in both sets, deny. Explicit blocks prevent accidents when someone expands the allowlist later.

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

### UX patterns that work

| Pattern | Use when |
|---------|----------|
| **Per-action prompt** | Destructive ops (delete, send, purchase) |
| **Session grant** | "Allow terminal for this task" |
| **Diff preview** | File writes — show patch before apply |
| **Timeout → deny** | No silent auto-approve after 30s |

Cursor uses per-action prompts for terminal and MCP writes. Claude Desktop prompts before filesystem mutations. Your harness should expose enough context (tool name, args, file path) for an informed decision.

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
            return False, "Step budget exhausted"
        if state.total_tokens >= self.budget.max_tokens:
            return False, "Token budget exhausted"
        if state.total_cost_usd >= self.budget.max_cost_usd:
            return False, "Cost budget exhausted"
        if elapsed >= self.budget.max_wall_seconds:
            return False, "Wall-clock timeout"

        return True, None

    def remaining(self, state) -> dict:
        return {
            "steps_left": self.budget.max_steps - state.step,
            "cost_left_usd": round(self.budget.max_cost_usd - state.total_cost_usd, 4),
            "tokens_left": self.budget.max_tokens - state.total_tokens,
        }
```

Inject budget status into perceive so the model paces itself:

```python
def perceive_with_budget(self, state, enforcer: BudgetEnforcer):
    messages = self.perceive(state)
    remaining = enforcer.remaining(state)
    messages[0]["content"] += (
        f"\n\nBudget remaining: {remaining['steps_left']} steps, "
        f"${remaining['cost_left_usd']} cost."
    )
    return messages
```

!!! warning "Silent budget exhaustion frustrates users"
    When a budget trips, return a **partial answer** with what was accomplished and which limit was hit — not a generic error.

---

## Scoped Permissions (Filesystem and Network)

Beyond tool names, constrain **what arguments are valid**:

```python
def validate_path(path: str, allowed_roots: list[str]) -> bool:
    resolved = os.path.realpath(path)
    return any(resolved.startswith(os.path.realpath(root)) for root in allowed_roots)

def validate_tool_args(name: str, args: dict, policy: dict) -> str | None:
    if name == "read_file" and "path" in args:
        if not validate_path(args["path"], policy["allowed_paths"]):
            return f"Path '{args['path']}' is outside allowed directories."
    if name == "run_terminal" and "command" in args:
        blocked = ["rm -rf", "curl | sh", "sudo"]
        cmd = args["command"]
        if any(b in cmd for b in blocked):
            return f"Command blocked by policy: {cmd[:50]}"
    return None
```

Argument-level policy catches cases where the tool is allowed but the **target** is not — e.g., `read_file("/etc/passwd")`.

---

## Putting It Together

```python
def harness_step(state, permissions, enforcer, hitl, registry, sandbox):
    ok, reason = enforcer.check(state)
    if not ok:
        state.status = "finished"
        state.scratchpad["stop_reason"] = reason
        return state

    response = model.complete(perceive_with_budget(state, enforcer))

    for call in response.tool_calls or []:
        arg_error = validate_tool_args(call.name, call.arguments, PATH_POLICY)
        if arg_error:
            state.append_tool_result(ToolResult(
                tool_call_id=call.id, name=call.name,
                arguments=call.arguments, output=registry._error("POLICY", arg_error),
                duration_ms=0, success=False,
            ))
            continue

        output = execute_with_permissions(
            registry, sandbox, permissions, hitl, call.name, call.arguments
        )
        # ... append result, checkpoint, etc.

    return state
```

---

## Key Takeaways

- **Harness policies** enforce safety before tools run — prompts alone are insufficient
- Use **allowlists** (default deny), **denylists** (explicit blocks), and **approval gates** for sensitive tools
- **Human-in-the-loop** pauses the loop with a clear action summary; timeout defaults to deny
- **Budgets** on steps, tokens, cost, and wall-clock time prevent runaway runs
- Validate **tool arguments** (paths, commands) — not just tool names

---

## Further Reading

- [Awesome Harness Engineering](https://github.com/ai-boost/awesome-harness-engineering) — permission and policy patterns
- [Agents Towards Production](https://github.com/NirDiamant/agents-towards-production) — guardrails in deployed agents
- [M16 · Prompt Injection](../../../production/module-16-ai-safety-ethics/lessons/04-lesson-04.md) — input-side threats

---

## Next Lesson

**Lesson 6: Observability in the Harness** — Traces, spans, and structured logging for every agent step.
