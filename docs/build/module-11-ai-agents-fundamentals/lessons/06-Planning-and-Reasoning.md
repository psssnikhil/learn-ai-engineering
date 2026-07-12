---
title: Planning & Reasoning
description: >-
  Learn how AI agents plan multi-step tasks, decompose problems, and reason
  about their actions
duration: 35 min
difficulty: advanced
has_code: false
module: module-11
objectives:
  - Explain task decomposition and why agents need planning
  - Implement a plan-and-execute agent pattern
  - Build a self-reflection loop that improves agent outputs
  - Compare planning strategies for different task types
---
# Planning & Reasoning

## Learning Objectives

By the end of this lesson, you will be able to:
- Understand why LLMs need explicit planning for complex tasks
- Implement task decomposition that breaks goals into steps
- Build plan-and-execute agents that follow structured plans
- Add self-reflection to catch and correct mistakes

---

## Why Agents Need Planning

LLMs generate text token-by-token. Without planning, they tend to:
- Start executing before thinking through the full approach
- Get stuck in local optima (first idea, not best idea)
- Lose track of progress on multi-step tasks
- Miss dependencies between subtasks

Planning forces the agent to think before acting.

---

## 1. Task Decomposition

The simplest planning strategy: ask the LLM to break a complex task into steps before executing any of them.

```python
from openai import OpenAI
import json

client = OpenAI()

def decompose_task(task: str) -> list[str]:
    """Break a complex task into ordered subtasks."""
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{
            "role": "user",
            "content": f"""Break this task into a numbered list of concrete steps.
Each step should be a single, actionable item.
Include only steps that are necessary.

Task: {task}

Return JSON: {{"steps": ["step 1", "step 2", ...]}}"""
        }],
        temperature=0,
        response_format={"type": "json_object"},
    )
    result = json.loads(response.choices[0].message.content)
    return result["steps"]

# Example
steps = decompose_task(
    "Research the top 3 Python web frameworks, compare their performance, "
    "and write a recommendation report"
)
# Returns:
# ["Research Flask: features, performance benchmarks, community size",
#  "Research Django: features, performance benchmarks, community size",
#  "Research FastAPI: features, performance benchmarks, community size",
#  "Create comparison table across key criteria",
#  "Write recommendation with justification"]
```

---

## 2. Plan-and-Execute Pattern

Separate planning from execution. One LLM call creates the plan, then individual calls execute each step.

```python
class PlanAndExecuteAgent:
    def __init__(self, tools: dict):
        self.tools = tools
        self.plan = []
        self.results = []

    def create_plan(self, goal: str) -> list[dict]:
        """Generate a structured plan for achieving the goal."""
        tools_desc = "
".join(
            f"- {name}: {func.__doc__}" for name, func in self.tools.items()
        )

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{
                "role": "user",
                "content": f"""Create a plan to achieve this goal.
Available tools: {tools_desc}

Goal: {goal}

Return JSON: {{"plan": [
  {{"step": 1, "action": "tool_name or 'think'", "input": "...", "depends_on": []}},
  ...
]}}"""
            }],
            temperature=0,
            response_format={"type": "json_object"},
        )
        self.plan = json.loads(response.choices[0].message.content)["plan"]
        return self.plan

    def execute_plan(self) -> list[dict]:
        """Execute each step of the plan in order."""
        for step in self.plan:
            print(f"Step {step['step']}: {step['action']} - {step['input']}")

            if step["action"] == "think":
                result = self._think(step["input"])
            elif step["action"] in self.tools:
                result = self.tools[step["action"]](step["input"])
            else:
                result = f"Unknown action: {step['action']}"

            self.results.append({
                "step": step["step"],
                "action": step["action"],
                "result": result,
            })
            print(f"  Result: {result[:200]}...")

        return self.results

    def _think(self, question: str) -> str:
        """Use the LLM to reason about intermediate results."""
        context = "
".join(
            f"Step {r['step']} ({r['action']}): {r['result'][:500]}"
            for r in self.results
        )
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{
                "role": "user",
                "content": f"Given these results so far:
{context}

Answer: {question}"
            }],
        )
        return response.choices[0].message.content
```

---

## 3. Self-Reflection

After generating an output, the agent evaluates its own work and iterates to improve it.

```python
def reflect_and_improve(task: str, output: str, max_iterations: int = 3) -> str:
    """Iteratively improve output through self-reflection."""
    current = output

    for i in range(max_iterations):
        # Step 1: Critique the current output
        critique = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{
                "role": "user",
                "content": f"""Critically evaluate this output for the given task.

Task: {task}
Output: {current}

List specific problems, inaccuracies, or areas for improvement.
If the output is good enough, respond with "APPROVED".

Return JSON: {{"approved": true/false, "issues": ["issue 1", ...]}}"""
            }],
            temperature=0,
            response_format={"type": "json_object"},
        )

        review = json.loads(critique.choices[0].message.content)

        if review["approved"]:
            print(f"Approved after {i + 1} iteration(s)")
            return current

        # Step 2: Revise based on the critique
        issues = "
".join(f"- {issue}" for issue in review["issues"])
        revision = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{
                "role": "user",
                "content": f"""Revise this output to fix the identified issues.

Task: {task}
Current output: {current}

Issues to fix:
{issues}

Provide the improved output only."""
            }],
        )
        current = revision.choices[0].message.content
        print(f"Iteration {i + 1}: Fixed {len(review['issues'])} issues")

    return current
```

---

## 4. Adaptive Replanning

When a step fails or new information changes the plan, the agent replans:

```python
class AdaptivePlanAgent:
    def __init__(self, tools: dict):
        self.tools = tools
        self.plan = []
        self.completed = []

    def run(self, goal: str) -> str:
        self.plan = self._make_plan(goal)

        while self.plan:
            step = self.plan.pop(0)
            result = self._execute_step(step)

            if result["success"]:
                self.completed.append(result)
            else:
                # Replan based on failure
                print(f"Step failed: {step}. Replanning...")
                remaining_desc = "
".join(
                    f"- {s}" for s in self.plan
                )
                completed_desc = "
".join(
                    f"- {r['step']}: {r['result'][:100]}"
                    for r in self.completed
                )

                self.plan = self._replan(
                    goal, completed_desc, step, result["error"], remaining_desc
                )

        return self._synthesize(goal)

    def _make_plan(self, goal: str) -> list[str]:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{
                "role": "user",
                "content": f"Create a step-by-step plan for: {goal}
Return JSON: {{\"steps\": [...]}}"
            }],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)["steps"]

    def _replan(self, goal, completed, failed_step, error, remaining):
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{
                "role": "user",
                "content": f"""Goal: {goal}
Completed: {completed}
Failed step: {failed_step}
Error: {error}
Remaining plan: {remaining}

Create a revised plan for the remaining work. Return JSON: {{"steps": [...]}}"""
            }],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content)["steps"]

    def _execute_step(self, step: str) -> dict:
        try:
            # Simplified: use LLM to execute the step
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": f"Execute this step: {step}"}],
            )
            return {"step": step, "result": response.choices[0].message.content, "success": True}
        except Exception as e:
            return {"step": step, "result": "", "error": str(e), "success": False}

    def _synthesize(self, goal: str) -> str:
        results_text = "
".join(
            f"Step: {r['step']}
Result: {r['result'][:300]}"
            for r in self.completed
        )
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{
                "role": "user",
                "content": f"Synthesize a final answer for '{goal}' from these results:
{results_text}"
            }],
        )
        return response.choices[0].message.content
```

---

## Choosing a Planning Strategy

| Strategy | Best For | Overhead | Reliability |
|----------|----------|----------|-------------|
| **No planning** (direct prompt) | Simple, single-step tasks | None | Low for complex tasks |
| **Task decomposition** | Known task types with clear steps | 1 extra LLM call | Medium |
| **Plan-and-execute** | Multi-tool tasks with dependencies | 1 planning + N execution calls | High |
| **Self-reflection** | Quality-critical outputs (writing, code) | 2-6 extra LLM calls | High |
| **Adaptive replanning** | Uncertain environments, long tasks | Variable | Highest |

---

## Key Takeaways

- Planning separates "what to do" from "how to do it," improving agent reliability
- Task decomposition is the simplest and most cost-effective planning strategy
- Self-reflection catches errors by having the agent critique its own output
- Adaptive replanning handles failures gracefully by revising the plan mid-execution
- More planning overhead means higher cost but better results on complex tasks

## Resources

- [Plan-and-Solve Prompting paper](https://arxiv.org/abs/2305.04091) -- Research on planning prompts
- [Reflexion paper](https://arxiv.org/abs/2303.11366) -- Self-reflection for language agents
- [Tree of Thoughts paper](https://arxiv.org/abs/2305.10601) -- Deliberate problem solving with LLMs

---

Next: Building an Agent from Scratch
