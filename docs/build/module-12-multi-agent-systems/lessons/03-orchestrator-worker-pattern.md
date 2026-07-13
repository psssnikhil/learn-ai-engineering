---
title: Orchestrator-Worker Pattern
description: >-
  Learn how to build centralized coordination systems where one agent manages
  multiple worker agents
duration: 45 min
difficulty: intermediate
has_code: false
module: module-12
youtube: 'https://www.youtube.com/watch?v=cJOxQqZQ7AE'
---
# Orchestrator-Worker Pattern

## Learning Objectives

| What You'll Learn | Time | Difficulty |
|-------------------|------|------------|
| Understand orchestrator-worker architecture | 45 min | Intermediate |
| Implement dynamic task delegation | | |
| Handle worker failures gracefully | | |
| Build production-ready orchestrators | | |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python 3.10+** | With `asyncio` and `openai` packages |
| **Module 5: AI Agents** | Agent fundamentals, tool use |
| **Module 7: Prompt Engineering** | System prompts for specialized workers |
| **Module 12, Lessons 1-2** | Multi-agent concepts and communication |

```bash
pip install openai pydantic python-dotenv
```

---

## What Is the Orchestrator-Worker Pattern?

The **orchestrator-worker pattern** is a multi-agent architecture where a central agent (orchestrator) breaks down complex tasks, delegates to specialized worker agents, and synthesizes results. The orchestrator never does the work itself — it plans, assigns, monitors, and combines.

This is the most common pattern in production multi-agent systems because it provides clear accountability, predictable workflows, and a single point for quality control.

### Architecture

```
                    ┌──────────────────┐
                    │   Orchestrator   │
                    │   (Coordinator)  │
                    │                  │
                    │  - Plans steps   │
                    │  - Assigns work  │
                    │  - Monitors      │
                    │  - Synthesizes   │
                    └────────┬─────────┘
                             │
            ┌────────────────┼────────────────┐
            ↓                ↓                ↓
      ┌──────────┐     ┌──────────┐    ┌──────────┐
      │ Worker 1 │     │ Worker 2 │    │ Worker 3 │
      │(Research)│     │(Analysis)│    │(Writing) │
      └──────────┘     └──────────┘    └──────────┘
```

### Key Responsibilities

**Orchestrator:**
- Task decomposition — break complex goals into steps
- Work assignment — route each step to the right worker
- Result aggregation — combine worker outputs
- Error handling — retry, fallback, or abort on failure
- Progress tracking — monitor step completion and latency

**Workers:**
- Execute assigned tasks with domain expertise
- Report results in a structured format
- Request clarifications when input is ambiguous
- Provide status updates for long-running tasks

---

## When to Use This Pattern

### Perfect For

| Use Case | Why It Works |
|----------|--------------|
| **Complex workflows** | Clear task decomposition into known steps |
| **Specialized agents** | Each worker has distinct expertise and prompts |
| **Predictable process** | Steps are known in advance (even if content varies) |
| **Quality control needed** | Central validation point before final output |
| **Resource management** | Central allocation of work and rate limiting |

### Not Ideal For

- Simple, single-step tasks (use a single agent)
- Highly dynamic, unpredictable workflows (consider peer-to-peer)
- Systems requiring peer collaboration without hierarchy
- Ultra-low latency requirements (orchestration adds overhead)

---

## What You'll Build

By the end of this lesson, you will have:

- [ ] A working orchestrator that plans and executes multi-step workflows
- [ ] Three specialized worker agents (research, analysis, writing)
- [ ] Failure handling with retry and fallback strategies
- [ ] LLM-powered dynamic planning (not just hardcoded workflows)
- [ ] A complete research report pipeline as a demonstration
- [ ] Structured logging of each step for debugging

---

## Step 1: Define the Worker Base Class

Workers are specialized agents with focused system prompts and a standard interface.

```python
# src/workers/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from openai import OpenAI
import time

client = OpenAI()

@dataclass
class WorkerResult:
    worker_name: str
    output: str
    success: bool
    latency_ms: float
    error: str = ""

class Worker(ABC):
    def __init__(self, name: str, system_prompt: str, model: str = "gpt-4.1-mini"):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model

    async def execute(self, task_input: str) -> WorkerResult:
        start = time.time()
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": task_input},
                ],
                temperature=0.3,
            )
            output = response.choices[0].message.content
            latency = (time.time() - start) * 1000
            return WorkerResult(
                worker_name=self.name,
                output=output,
                success=True,
                latency_ms=latency,
            )
        except Exception as e:
            latency = (time.time() - start) * 1000
            return WorkerResult(
                worker_name=self.name,
                output="",
                success=False,
                latency_ms=latency,
                error=str(e),
            )

    @abstractmethod
    def get_capabilities(self) -> str:
        pass
```

---

## Step 2: Implement Specialized Workers

```python
# src/workers/specialists.py
from src.workers.base import Worker

class ResearchWorker(Worker):
    def __init__(self):
        super().__init__(
            name="research_agent",
            system_prompt=(
                "You are a research specialist. Given a topic or question, "
                "provide comprehensive background information, key facts, "
                "relevant statistics, and important context. "
                "Structure your response with clear sections. "
                "Cite sources when possible. Be thorough but concise."
            ),
        )

    def get_capabilities(self) -> str:
        return "Background research, fact gathering, context building"

class AnalysisWorker(Worker):
    def __init__(self):
        super().__init__(
            name="analysis_agent",
            system_prompt=(
                "You are an analysis specialist. Given research data, "
                "identify patterns, trends, key insights, and implications. "
                "Compare different perspectives. Highlight strengths and weaknesses. "
                "Provide data-driven conclusions."
            ),
        )

    def get_capabilities(self) -> str:
        return "Pattern analysis, trend identification, insight extraction"

class WritingWorker(Worker):
    def __init__(self):
        super().__init__(
            name="writing_agent",
            system_prompt=(
                "You are a writing specialist. Given research and analysis, "
                "produce a well-structured, clear, and engaging written report. "
                "Use professional tone. Include an executive summary, "
                "main sections, and conclusions. Format with headers."
            ),
        )

    def get_capabilities(self) -> str:
        return "Report writing, summarization, content structuring"
```

---

## Step 3: Build the Orchestrator

The orchestrator plans workflows, delegates to workers, handles failures, and synthesizes results.

```python
# src/orchestrator.py
import asyncio
import json
from typing import Any
from openai import OpenAI
from src.workers.base import Worker, WorkerResult

client = OpenAI()

class Orchestrator:
    def __init__(self):
        self.workers: dict[str, Worker] = {}
        self.step_log: list[dict] = []

    def register_worker(self, worker: Worker):
        self.workers[worker.name] = worker
        print(f"Registered worker: {worker.name} ({worker.get_capabilities()})")

    async def execute_workflow(self, task: dict) -> dict:
        workflow_id = task.get("id", "workflow-1")
        print(f"Starting workflow: {workflow_id}")

        plan = await self.plan_workflow(task)
        print(f"Plan: {len(plan)} steps")

        results: dict[str, WorkerResult] = {}
        for step in plan:
            resolved_input = self._resolve_input(step["input"], results)
            step_result = await self.execute_step(step, resolved_input)
            results[step["name"]] = step_result
            self.step_log.append({
                "step": step["name"],
                "worker": step["worker"],
                "success": step_result.success,
                "latency_ms": step_result.latency_ms,
            })

        final = await self.synthesize_results(task, results)
        print(f"Workflow complete: {workflow_id}")
        return final

    async def plan_workflow(self, task: dict) -> list[dict]:
        """Use LLM to dynamically plan workflow steps."""
        worker_descriptions = "\n".join(
            f"- {name}: {w.get_capabilities()}"
            for name, w in self.workers.items()
        )

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{
                "role": "user",
                "content": f"""Plan a workflow to complete this task.

Task: {task.get('description', task.get('query', ''))}
Task type: {task.get('type', 'general')}

Available workers:
{worker_descriptions}

Return a JSON array of steps:
[
  {{"name": "step_name", "worker": "worker_name", "input": "what to pass to the worker"}},
  ...
]

Use {{{{step_name}}}} in input to reference a previous step's output.
Keep plans to 2-5 steps. Order steps logically.""",
            }],
            response_format={"type": "json_object"},
        )

        plan_data = json.loads(response.choices[0].message.content)
        steps = plan_data.get("steps", plan_data.get("plan", []))
        return steps

    def _resolve_input(self, input_template: str, results: dict[str, WorkerResult]) -> str:
        """Replace {{step_name}} placeholders with actual step outputs."""
        resolved = input_template
        for step_name, result in results.items():
            placeholder = "{{" + step_name + "}}"
            if placeholder in resolved:
                resolved = resolved.replace(placeholder, result.output)
        return resolved

    async def execute_step(self, step: dict, resolved_input: str) -> WorkerResult:
        worker_name = step["worker"]
        worker = self.workers.get(worker_name)

        if not worker:
            return WorkerResult(
                worker_name=worker_name,
                output="",
                success=False,
                error=f"Worker not found: {worker_name}",
                latency_ms=0,
            )

        print(f"  Executing: {step['name']} -> {worker_name}")
        result = await worker.execute(resolved_input)

        if not result.success:
            return await self.handle_step_failure(step, resolved_input, result)

        print(f"  Completed: {step['name']} ({result.latency_ms:.0f}ms)")
        return result

    async def handle_step_failure(
        self, step: dict, resolved_input: str, failed_result: WorkerResult,
    ) -> WorkerResult:
        retries = step.get("retries", 0)
        max_retries = 3

        if retries < max_retries:
            print(f"  Retrying: {step['name']} (attempt {retries + 1}/{max_retries})")
            step["retries"] = retries + 1
            await asyncio.sleep(2 ** retries)
            return await self.execute_step(step, resolved_input)

        fallback = step.get("fallback_worker")
        if fallback and fallback in self.workers:
            print(f"  Fallback: {step['name']} -> {fallback}")
            step["worker"] = fallback
            step["retries"] = 0
            return await self.execute_step(step, resolved_input)

        print(f"  Failed: {step['name']} after {max_retries} retries")
        return failed_result

    async def synthesize_results(self, task: dict, results: dict[str, WorkerResult]) -> dict:
        """Combine worker results into a final output."""
        successful = {k: v for k, v in results.items() if v.success}
        failed = {k: v for k, v in results.items() if not v.success}

        if failed and not successful:
            return {
                "status": "failed",
                "error": f"All steps failed: {list(failed.keys())}",
                "step_log": self.step_log,
            }

        combined = "\n\n".join(
            f"## {name}\n{result.output}"
            for name, result in successful.items()
        )

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{
                "role": "user",
                "content": f"""Synthesize these worker outputs into a final deliverable.

Original task: {task.get('description', '')}

Worker outputs:
{combined}

Produce a polished final output that integrates all worker contributions.""",
            }],
            temperature=0.3,
        )

        return {
            "status": "success" if not failed else "partial",
            "final_output": response.choices[0].message.content,
            "steps_completed": len(successful),
            "steps_failed": len(failed),
            "step_log": self.step_log,
            "total_latency_ms": sum(s["latency_ms"] for s in self.step_log),
        }
```

---

## Step 4: Run a Complete Workflow

```python
# demo.py
import asyncio
from src.orchestrator import Orchestrator
from src.workers.specialists import ResearchWorker, AnalysisWorker, WritingWorker

async def main():
    orchestrator = Orchestrator()
    orchestrator.register_worker(ResearchWorker())
    orchestrator.register_worker(AnalysisWorker())
    orchestrator.register_worker(WritingWorker())

    task = {
        "id": "report-001",
        "type": "research_report",
        "description": "Write a report on the impact of AI agents on software engineering productivity",
        "query": "How are AI coding agents changing software development?",
    }

    result = await orchestrator.execute_workflow(task)

    print(f"\nStatus: {result['status']}")
    print(f"Steps: {result['steps_completed']} completed, {result.get('steps_failed', 0)} failed")
    print(f"Total latency: {result['total_latency_ms']:.0f}ms")
    print(f"\n--- Final Output ---\n{result['final_output'][:500]}...")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Step 5: Hardcoded Workflows (Alternative)

For predictable pipelines, skip LLM planning and use predefined workflows:

```python
HARDCODED_PLANS = {
    "research_report": [
        {"name": "research", "worker": "research_agent", "input": "{query}"},
        {"name": "analyze", "worker": "analysis_agent", "input": "{{research}}"},
        {"name": "write", "worker": "writing_agent", "input": "Research:\n{{research}}\n\nAnalysis:\n{{analyze}}"},
    ],
    "code_review": [
        {"name": "security", "worker": "security_agent", "input": "{code}"},
        {"name": "quality", "worker": "quality_agent", "input": "{code}"},
        {"name": "synthesize", "worker": "writing_agent",
         "input": "Security review:\n{{security}}\n\nQuality review:\n{{quality}}"},
    ],
}

async def plan_hardcoded(self, task: dict) -> list[dict]:
    plan_template = HARDCODED_PLANS.get(task["type"], [])
    query = task.get("query", task.get("description", ""))
    return [
        {**step, "input": step["input"].replace("{query}", query).replace("{code}", task.get("code", ""))}
        for step in plan_template
    ]
```

Use hardcoded plans when the workflow is stable. Use LLM planning when task types vary or steps depend on task content.

---

## Testing Your Build

### Unit Tests

```python
# tests/test_orchestrator.py
import pytest
import asyncio
from src.orchestrator import Orchestrator
from src.workers.base import Worker, WorkerResult

class MockWorker(Worker):
    def __init__(self, name: str, response: str = "mock output", should_fail: bool = False):
        super().__init__(name=name, system_prompt="test")
        self._response = response
        self._should_fail = should_fail

    async def execute(self, task_input: str) -> WorkerResult:
        if self._should_fail:
            return WorkerResult(self.name, "", False, 10, error="mock failure")
        return WorkerResult(self.name, self._response, True, 10)

    def get_capabilities(self) -> str:
        return f"Mock {self.name}"

@pytest.fixture
def orchestrator():
    o = Orchestrator()
    o.register_worker(MockWorker("research_agent", "Research findings here"))
    o.register_worker(MockWorker("analysis_agent", "Analysis results here"))
    o.register_worker(MockWorker("writing_agent", "Final report here"))
    return o

@pytest.mark.asyncio
async def test_workflow_completes(orchestrator):
    result = await orchestrator.execute_workflow({
        "type": "research_report",
        "query": "Test topic",
        "description": "Test report",
    })
    assert result["status"] in ("success", "partial")

@pytest.mark.asyncio
async def test_worker_failure_handled(orchestrator):
    orchestrator.register_worker(MockWorker("failing_agent", should_fail=True))
    result = await orchestrator.execute_workflow({
        "type": "general",
        "description": "Test with failing worker",
    })
    assert "step_log" in result

def test_input_resolution():
    o = Orchestrator()
    results = {
        "research": WorkerResult("research_agent", "Research data", True, 10),
    }
    resolved = o._resolve_input("Based on: {{research}}", results)
    assert "Research data" in resolved
```

### Manual Testing Checklist

- [ ] Orchestrator registers workers and lists capabilities
- [ ] LLM planning produces valid step sequences
- [ ] Steps execute in order with correct input passing
- [ ] `{{step_name}}` placeholders resolve to previous outputs
- [ ] Failed steps retry up to 3 times
- [ ] Final synthesis produces coherent combined output
- [ ] Step log captures latency and success for each step

---

## Deployment Notes

### Production Considerations

| Concern | Development | Production |
|---------|-------------|------------|
| Planning | LLM-generated (flexible) | Hardcoded for known workflows, LLM for novel tasks |
| Worker calls | Sequential | Parallel where steps are independent |
| Failure handling | Retry 3x | Retry + circuit breaker + dead letter queue |
| Observability | Print statements | Structured logging + tracing (Langfuse/LangSmith) |
| Cost control | No limits | Per-workflow token budget, worker timeouts |

### Parallel Execution

When steps are independent, run them concurrently:

```python
async def execute_parallel_steps(self, steps: list[dict]) -> dict[str, WorkerResult]:
    tasks = [self.execute_step(step, step["input"]) for step in steps]
    results = await asyncio.gather(*tasks)
    return {steps[i]["name"]: results[i] for i in range(len(steps))}
```

### Framework Alternatives

| Framework | Orchestrator Pattern | Best For |
|-----------|---------------------|----------|
| **LangGraph** | State machine orchestrator | Complex branching workflows |
| **CrewAI** | Manager + crew members | Role-based agent teams |
| **AutoGen** | Group chat manager | Conversational multi-agent |
| **Custom (this lesson)** | Full control | Learning and simple pipelines |

---

## Extensions and Challenges

- **Dynamic worker creation**: Orchestrator spawns new workers based on task needs
- **Human-in-the-loop**: Pause workflow for human approval before synthesis
- **Worker specialization via fine-tuning**: Fine-tune each worker on domain-specific data
- **Cost tracking**: Log token usage per worker per workflow for budgeting
- **Workflow templates**: Save and reuse successful plans for similar tasks
- **LangGraph migration**: Reimplement the orchestrator as a LangGraph state machine

---

## Key Takeaways

- The orchestrator-worker pattern provides centralized control with specialized execution
- The orchestrator plans, delegates, monitors, and synthesizes — workers execute
- Use LLM planning for flexible workflows, hardcoded plans for predictable pipelines
- Always implement retry, fallback, and structured logging for production reliability
- Input resolution between steps (`{{step_name}}`) is the key mechanism for chaining workers
- Frameworks like LangGraph and CrewAI implement this pattern with additional features

---

## Next Lesson

**Lesson 4: Peer-to-Peer Agent Collaboration** — Learn patterns where agents communicate directly without a central orchestrator.
