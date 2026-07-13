---
title: Agent Architectures
description: >-
  Explore the major architectural patterns for building AI agents, from simple
  ReAct loops to advanced planning and reflection systems
duration: 60 min
difficulty: advanced
has_code: true
module: module-11
---

# Agent Architectures

## Prerequisites

- **Lesson 01 — Introduction to Agents** — agent loop, core components, tool calling basics
- **Python intermediate** — classes, generators, async familiarity helpful
- **Awareness of LLM cost/latency tradeoffs** — you'll need this when choosing patterns

---

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Map the landscape of agent patterns from simplest to most complex | 10 min | Intermediate |
| Implement and compare Simple Tool Use, ReAct, Plan-and-Execute, and Reflection | 25 min | Advanced |
| Understand when each pattern is appropriate | 10 min | Advanced |
| Combine patterns for real-world hybrid agents | 15 min | Advanced |

---

## Intuition First: Choosing a Structure for Autonomous Work

When you give a person a complex task, the work structure they use depends on the task:

- **Quick lookup**: Grab the reference, answer. No plan needed.
- **Research report**: Read multiple sources, integrate, write. A linear loop works.
- **Software project**: Break into tasks, assign order, adapt as things fail. You need an explicit plan.
- **Writing an essay for a deadline**: Write a draft, critique it, revise, critique again. Iterative refinement.

The same logic applies to agent architectures. Simple tasks need simple structures. Complex tasks benefit from explicit planning. Output-quality tasks need self-critique. Using Plan-and-Execute for a simple lookup is wasteful; using Simple Tool Use for a 20-step research project will fail.

This lesson maps four patterns to their appropriate task types and shows you how to implement each.

---

## The Architecture Landscape

```
Simplicity                                          Capability
|──────────────────────────────────────────────────|
│                                                   │
│  Simple       ReAct         Plan-and-    Reflect  │
│  Tool Use     (Workhorse)   Execute      -ion     │
│                                                   │
│  1 LLM call   Loop until    Plan first,  Draft →  │
│  + 1 tool     done          execute each critique │
│  + synthesis               step          → revise │
│                                                   │
│  LLM calls:   LLM calls:   LLM calls:   LLM calls:│
│  2            2–10         N steps × 2  2–6       │
│                            + 1 planner            │
```

Complexity should be introduced only when simpler patterns fail. Resist the temptation to reach immediately for Plan-and-Execute.

---

## Pattern 1: Simple Tool Use (Router)

The simplest agentic pattern: the LLM decides which tool to call, calls it once, and synthesizes an answer.

```
User query
    │
    ▼
[LLM: which tool?]
    │
    ▼
[execute one tool]
    │
    ▼
[LLM: synthesize answer from tool result]
    │
    ▼
Final answer
```

This is technically two LLM calls, but there is no loop. It is the foundation of tool-augmented chatbots.

```python
from openai import OpenAI
import json

client = OpenAI()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city. Returns temp in Celsius and conditions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. 'Tokyo'"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get current stock price for a ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker, e.g. 'AAPL'"}
                },
                "required": ["ticker"],
            },
        },
    },
]

def simple_tool_agent(query: str) -> str:
    """Route a user query to the appropriate tool, then synthesize."""
    messages = [
        {"role": "system", "content": "Use the most appropriate tool to answer the query."},
        {"role": "user", "content": query},
    ]

    # First LLM call: decide which tool to call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS,
    )
    message = response.choices[0].message
    messages.append(message)

    if not message.tool_calls:
        # LLM answered directly — no tool needed
        return message.content

    # Execute the (single) tool call
    tc = message.tool_calls[0]
    args = json.loads(tc.function.arguments)

    # Simulate tool execution
    if tc.function.name == "get_weather":
        result = {"city": args["city"], "temp_c": 22, "conditions": "Partly cloudy"}
    elif tc.function.name == "get_stock_price":
        result = {"ticker": args["ticker"], "price_usd": 187.42}
    else:
        result = {"error": f"Unknown tool: {tc.function.name}"}

    messages.append({
        "role": "tool",
        "tool_call_id": tc.id,
        "content": json.dumps(result),
    })

    # Second LLM call: synthesize a natural-language answer
    synthesis = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
    )
    return synthesis.choices[0].message.content


print(simple_tool_agent("What's the weather like in Tokyo right now?"))
# → "The current weather in Tokyo is 22°C and partly cloudy."
```

**When to use**: Single-step lookups, FAQ bots, any task that requires at most one tool call.

**Limitations**: Cannot recover from a wrong tool choice. Cannot chain tools. Cannot iterate to refine an answer.

---

## Pattern 2: ReAct (Reason + Act) Loop

The workhorse pattern. The agent cycles through Thought → Action → Observation until it decides the goal is achieved. No upfront plan — each step is decided based on what was learned in previous steps.

```
User goal
    │
    ▼
[THINK: What do I need next?]
    │
    ▼
[ACT: call a tool]
    │
    ▼
[OBSERVE: tool result]
    │
    ▼
[THINK: What do I know now? What next?]
    │
    ▼
...repeat...
    │
    ▼
[THINK: I have enough. Final answer.]
    │
    ▼
Final answer
```

The critical insight from Yao et al. (2022): **interleaving reasoning with action** produces far better results than either pure chain-of-thought reasoning or blind tool use alone. The model thinks about why it's calling a tool, reads the result, and updates its understanding — just like a human researcher.

```python
def react_agent(goal: str, tools: list, max_steps: int = 10) -> str:
    """
    ReAct loop: reason → act → observe, repeat until done.
    The LLM stops calling tools when it's ready to give the final answer.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful agent. For each step:\n"
                "1. Think about what information you have and what you still need.\n"
                "2. If you need external information, use a tool.\n"
                "3. When you have enough information, give your final answer "
                "   WITHOUT calling any tool.\n"
                "Be concise in your reasoning. Don't repeat yourself."
            ),
        },
        {"role": "user", "content": goal},
    ]

    for step in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            temperature=0.2,
        )
        message = response.choices[0].message
        messages.append(message)

        # No tool calls = agent is done
        if not message.tool_calls:
            return message.content

        # Execute all requested tools (may be parallel)
        for tc in message.tool_calls:
            print(f"  [{step + 1}] {tc.function.name}({tc.function.arguments})")
            result = execute_tool(tc.function.name, tc.function.arguments)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": str(result),
            })

    # Forced final answer if max steps reached
    messages.append({
        "role": "user",
        "content": "Please give your best answer based on what you've gathered.",
    })
    final = client.chat.completions.create(model="gpt-4o", messages=messages)
    return final.choices[0].message.content
```

**When to use**: Research tasks, debugging, data gathering, most general-purpose agents.

**Limitations**: Can loop if confused; context window fills up on long tasks; no upfront structure for complex multi-step workflows.

---

## Pattern 3: Plan-and-Execute

For complex tasks, an upfront plan reduces wasted steps and makes failures easier to diagnose. The LLM first generates a structured plan, then an executor handles each step.

```
User goal
    │
    ▼
[PLANNER LLM]
    │  "Step 1: ... Step 2: ... Step 3: ..."
    ▼
Plan (list of steps)
    │
    ├─► [EXECUTOR: run step 1 as a mini-ReAct agent]
    │         │ result 1
    ├─► [EXECUTOR: run step 2, given result 1]
    │         │ result 2
    ├─► [EXECUTOR: run step 3, given results 1 & 2]
    │         │ result 3
    ▼
[SYNTHESIZER LLM]
    │  Combines all results into final answer
    ▼
Final answer
```

The planner and executor can be different LLMs. A cheap model (GPT-4o-mini) handles simple execution steps; a capable model (GPT-4o) handles complex planning and final synthesis.

```python
from dataclasses import dataclass

@dataclass
class StepResult:
    step: str
    result: str
    success: bool

def plan_and_execute_agent(goal: str, tools: list) -> str:
    """
    Phase 1: Generate a step-by-step plan.
    Phase 2: Execute each step with a mini-ReAct agent.
    Phase 3: Synthesize results into a final answer.
    """

    # ── Phase 1: Planning ────────────────────────────────────────
    plan_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strategic planner. Given a goal, produce a numbered "
                    "step-by-step plan. Each step should be a concrete, actionable task. "
                    "Output ONLY the numbered list, one step per line."
                ),
            },
            {"role": "user", "content": f"Goal: {goal}"},
        ],
        temperature=0.1,
    )
    plan_text = plan_response.choices[0].message.content
    steps = [
        line.strip()
        for line in plan_text.splitlines()
        if line.strip() and line.strip()[0].isdigit()
    ]
    print(f"Plan ({len(steps)} steps):")
    for s in steps:
        print(f"  {s}")

    # ── Phase 2: Execute each step ───────────────────────────────
    step_results: list[StepResult] = []

    for i, step in enumerate(steps, start=1):
        prior_context = "\n".join(
            f"Step {j}: {r.result}" for j, r in enumerate(step_results, start=1)
        )
        task = (
            f"Execute this step: {step}\n\n"
            f"Prior results:\n{prior_context}" if prior_context else
            f"Execute this step: {step}"
        )

        print(f"\nExecuting step {i}: {step}")
        result = react_agent(task, tools, max_steps=5)
        step_results.append(StepResult(step=step, result=result, success=True))
        print(f"  Result: {result[:100]}...")

    # ── Phase 3: Synthesize ──────────────────────────────────────
    results_summary = "\n\n".join(
        f"Step {i}: {r.step}\nResult: {r.result}"
        for i, r in enumerate(step_results, start=1)
    )

    synthesis = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Synthesize the step results into a coherent final answer."},
            {
                "role": "user",
                "content": f"Original goal: {goal}\n\n{results_summary}",
            },
        ],
    )
    return synthesis.choices[0].message.content


# Example
result = plan_and_execute_agent(
    goal="Write a brief competitive analysis of OpenAI, Anthropic, and Google DeepMind",
    tools=TOOL_SCHEMAS,
)
print(f"\nFinal answer:\n{result}")
```

**When to use**: Complex multi-step tasks, report generation, workflows where step order matters and each step depends on prior results.

**Limitations**: Planning takes extra tokens and latency. Initial plan may become invalid mid-execution. Re-planning adds significant complexity.

!!! note "Re-planning: when to revise the plan"
    Add a simple check after each step: if the result indicates the step failed (e.g., contains "error" or "not found"), generate a revised plan for the remaining steps. This makes the system more robust but adds one LLM call per failure.

---

## Pattern 4: Reflection (Generate → Critique → Revise)

For tasks where output *quality* matters more than external information gathering, the reflection pattern iteratively improves a draft by having the LLM critique its own work.

```
Task
 │
 ▼
[GENERATE initial draft]
 │
 ▼
[CRITIQUE: What is wrong, missing, or improvable?]
 │
 ├─ "Looks good, no major issues" → return draft
 │
 └─ [REVISE based on critique]
         │
         ▼
     [CRITIQUE again]
         │
     ...repeat up to N times...
```

This is the pattern behind "Constitutional AI" (Anthropic) and self-refinement research.

```python
def reflection_agent(
    task: str,
    max_iterations: int = 3,
    quality_threshold: str = "no significant issues",
) -> str:
    """
    Generate, critique, and iteratively revise until the critique
    says the output meets quality standards.
    """

    # Initial draft
    draft = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Complete the task to the best of your ability."},
            {"role": "user", "content": task},
        ],
    ).choices[0].message.content

    print(f"Initial draft:\n{draft[:200]}...\n")

    for i in range(max_iterations):
        # Critique
        critique = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict editor. Critique the draft below critically. "
                        "Identify: factual errors, logical gaps, missing information, "
                        "poor structure, or unclear writing. "
                        "If the draft is high quality with no significant issues, "
                        "say 'no significant issues'. Be specific and actionable."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Task: {task}\n\nDraft:\n{draft}",
                },
            ],
            temperature=0.3,
        ).choices[0].message.content

        print(f"Critique {i + 1}: {critique[:150]}...")

        # Check if quality threshold met
        if quality_threshold.lower() in critique.lower():
            print(f"Quality threshold met after {i + 1} critique(s).")
            return draft

        # Revise based on critique
        draft = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "Revise the draft to address all the critique points. Keep what works.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Task: {task}\n\n"
                        f"Draft:\n{draft}\n\n"
                        f"Critique:\n{critique}\n\n"
                        "Produce an improved draft."
                    ),
                },
            ],
        ).choices[0].message.content

        print(f"Revised draft:\n{draft[:200]}...\n")

    return draft   # Return best effort after max_iterations


result = reflection_agent(
    task="Write a 200-word executive summary of the benefits and risks of RAG systems.",
    max_iterations=2,
)
print(f"\nFinal:\n{result}")
```

**When to use**: Content creation, code generation, structured documents, any task where iterative refinement demonstrably improves quality.

**Limitations**: Each critique/revise cycle adds 2 LLM calls. Three iterations = 7 LLM calls total (1 draft + 3 critiques + 3 revisions). Use sparingly.

---

## Choosing the Right Architecture

| Task | Pattern | Cost | Latency |
|------|---------|------|---------|
| Weather lookup, single-step Q&A | Simple Tool Use | $ | Low |
| Web research, bug fixing, data analysis | ReAct | $$ | Medium |
| Multi-step report, project planning | Plan-and-Execute | $$$ | High |
| Writing, code generation, structured docs | Reflection | $$ | Medium |
| Research + polished report | Plan-and-Execute + Reflection | $$$$ | High |

**Decision flowchart**:

```
Can the task be answered with one tool call?
    Yes → Simple Tool Use
    No  ↓
Does the task require >3 sequential steps with dependencies?
    Yes → Plan-and-Execute
    No  ↓
Does the task require external data or computation?
    Yes → ReAct
    No  ↓
Does the task produce a text artifact that benefits from self-critique?
    Yes → Reflection
    No  → Simple LLM call (no agent needed)
```

---

## Combining Patterns: Hybrid Agents

Production agents rarely use a single pure pattern. Common combinations:

**Plan-and-Execute with ReAct execution**: The planner generates a high-level plan; each step is executed by a mini-ReAct agent. This is the most reliable pattern for complex multi-step tasks.

**ReAct with Reflection review**: After the ReAct agent produces a draft answer, run one reflection pass. Adds ~30% cost for significantly better output quality.

**Hierarchical agents**: A supervisor agent routes tasks to specialist sub-agents (e.g., a "code agent", a "research agent", a "writing agent"). The supervisor uses Simple Tool Use; the sub-agents use ReAct. This is the "multi-agent" architecture pattern.

```python
def hybrid_agent(goal: str, tools: list) -> str:
    """
    Plan-and-Execute with final Reflection pass.
    Combines planning + iterative execution + quality refinement.
    """
    # Phase 1 & 2: Plan and execute
    raw_answer = plan_and_execute_agent(goal, tools)

    # Phase 3: Reflect and refine the synthesized answer
    refined = reflection_agent(
        task=f"Improve this answer for the goal: {goal}\n\nDraft answer:\n{raw_answer}",
        max_iterations=1,  # just one critique pass
    )
    return refined
```

---

## Edge Cases & Misconceptions

**Misconception: Plan-and-Execute is always better than ReAct.**
Plans become stale. If step 2 fails unexpectedly, steps 3–5 built on step 2 are now wrong. ReAct adapts organically; Plan-and-Execute must explicitly re-plan. Use Plan-and-Execute when the task structure is known in advance and steps are relatively independent.

**Misconception: Reflection always improves output quality.**
The LLM critiques its own work — but it has the same knowledge biases as the drafting LLM. If the initial draft contains a subtle factual error, the critique may not catch it. Reflection is most effective for structure, clarity, and completeness; less reliable for factual accuracy.

**Edge case: Plan-and-Execute with 20+ steps.**
Passing all prior step results to each new step grows the context linearly. For very long plans, summarize prior results before passing them forward, or use a memory store rather than in-context accumulation.

**Edge case: Reflection loops that don't converge.**
If the critique never returns "no significant issues," the loop runs to max_iterations regardless of quality. Add a fallback: if the draft didn't improve between iterations (measure by similarity or a "better/worse" judge LLM call), exit early.

---

## Production Connection

Framework implementations of these patterns:

- **LangGraph** — graph-based agent orchestration; nodes are functions or LLM calls, edges are conditional transitions. Excellent for Plan-and-Execute and hybrid agents.
- **AutoGen (Microsoft)** — multi-agent conversation framework; supervisor + specialist agents with automatic handoffs.
- **CrewAI** — role-based multi-agent system with task assignment and sequential/parallel execution modes.

For most production use cases, start with a direct implementation (as shown here) before reaching for a framework. Frameworks add abstraction that makes debugging harder; understand the patterns first.

---

## Key Takeaways

- Four primary patterns: Simple Tool Use (1 tool call), ReAct (iterative loop), Plan-and-Execute (plan first, execute each step), Reflection (generate + critique + revise).
- Start simple: ReAct solves most general-purpose agent tasks without the complexity of Plan-and-Execute.
- Plan-and-Execute is appropriate when the task has clear sequential steps and each step's output feeds the next.
- Reflection improves output quality at the cost of 2–4 extra LLM calls per iteration; most effective for writing and structured document generation.
- Production agents typically combine patterns — Plan-and-Execute with ReAct execution is the most robust combination for complex tasks.
- LLM calls grow multiplicatively with complexity; always model the expected LLM call count before committing to a pattern.

---

## Related Papers

| Paper | What it contributes | Link |
|-------|-------------------|------|
| Yao et al. (2022) — *ReAct: Synergizing Reasoning and Acting* | Introduces the ReAct pattern; shows reasoning traces improve tool-use accuracy | [arxiv.org/abs/2210.03629](https://arxiv.org/abs/2210.03629) |
| Wang et al. (2023) — *Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning* | Explicit plan-before-solve prompting strategy and its benefits | [arxiv.org/abs/2305.04091](https://arxiv.org/abs/2305.04091) |
| Madaan et al. (2023) — *Self-Refine: Iterative Refinement with Self-Feedback* | Iterative critique and revision without training; the Reflection pattern | [arxiv.org/abs/2303.17651](https://arxiv.org/abs/2303.17651) |
| Wu et al. (2023) — *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation* | Multi-agent framework; supervisor/specialist architecture | [arxiv.org/abs/2308.08155](https://arxiv.org/abs/2308.08155) |

---

## Further Reading

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/) — graph-based agent orchestration
- [AutoGen GitHub](https://github.com/microsoft/autogen) — Microsoft's multi-agent framework
- [Anthropic: Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — real-world patterns and pitfalls

---

## Next Lesson

**[Lesson 3: The ReAct Pattern](03-ReAct-Pattern.md)** — Deep dive into the ReAct implementation: tool descriptions, error handling, token management, and production optimization.
