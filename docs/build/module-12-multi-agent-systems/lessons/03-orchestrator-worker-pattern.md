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

## 🎯 Learning Objectives

| What You'll Learn | Time | Difficulty |
|-------------------|------|------------|
| Understand orchestrator-worker architecture | 45 min | Intermediate |
| Implement dynamic task delegation | | |
| Handle worker failures gracefully | | |
| Build production-ready orchestrators | | |

---

## 📚 What Is Orchestrator-Worker Pattern?

The **orchestrator-worker pattern** is where a central agent (orchestrator) breaks down complex tasks, delegates to specialized worker agents, and synthesizes results.

### Architecture

```
                    ┌──────────────────┐
                    │   Orchestrator   │
                    │   (Coordinator)  │
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
- 🎯 Task decomposition
- 📋 Work assignment
- 🔄 Result aggregation
- ⚠️ Error handling
- 📊 Progress tracking

**Workers:**
- ⚡ Execute assigned tasks
- 📤 Report results
- 🔄 Request clarifications
- ⏱️ Provide status updates

---

## 🏗️ When to Use This Pattern

### ✅ Perfect For:

| Use Case | Why It Works |
|----------|--------------|
| **Complex workflows** | Clear task decomposition |
| **Specialized agents** | Each worker has expertise |
| **Predictable process** | Known steps in advance |
| **Quality control needed** | Central validation point |
| **Resource management** | Central allocation of work |

### ❌ Not Ideal For:

- Simple, single-step tasks
- Highly dynamic, unpredictable workflows
- Systems requiring peer collaboration
- Ultra-low latency requirements

---

## 💻 Implementation: Basic Orchestrator

```python
from typing import List, Dict, Any
import asyncio

class Orchestrator:
    def __init__(self):
        self.workers = {}
        self.task_queue = asyncio.Queue()
        self.results = {}
    
    def register_worker(self, name: str, worker):
        """Register a worker agent"""
        self.workers[name] = worker
        print(f"✅ Registered worker: {name}")
    
    async def execute_workflow(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main orchestration logic
        Breaks down task and coordinates workers
        """
        workflow_id = task.get("id")
        print(f"🎯 Starting workflow: {workflow_id}")
        
        # Step 1: Plan the workflow
        plan = await self.plan_workflow(task)
        
        # Step 2: Execute each step
        results = {}
        for step in plan:
            step_result = await self.execute_step(step)
            results[step["name"]] = step_result
        
        # Step 3: Synthesize final result
        final_result = await self.synthesize_results(results)
        
        print(f"✅ Workflow complete: {workflow_id}")
        return final_result
    
    async def plan_workflow(self, task: Dict[str, Any]) -> List[Dict]:
        """
        Decide what steps are needed and in what order
        This is where LLM can help with dynamic planning
        """
        task_type = task.get("type")
        
        if task_type == "research_report":
            return [
                {"name": "research", "worker": "research_agent", "input": task["query"]},
                {"name": "analyze", "worker": "analysis_agent", "input": "{{research}}"},
                {"name": "write", "worker": "writing_agent", "input": "{{analyze}}"},
            ]
        elif task_type == "code_review":
            return [
                {"name": "security_check", "worker": "security_agent", "input": task["code"]},
                {"name": "quality_check", "worker": "quality_agent", "input": task["code"]},
                {"name": "performance_check", "worker": "perf_agent", "input": task["code"]},
            ]
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    async def execute_step(self, step: Dict) -> Any:
        """Execute a single workflow step"""
        worker_name = step["worker"]
        worker = self.workers.get(worker_name)
        
        if not worker:
            raise ValueError(f"Worker not found: {worker_name}")
        
        print(f"  ⚙️  Executing step: {step['name']} with {worker_name}")
        
        try:
            result = await worker.execute(step["input"])
            print(f"  ✅ Step complete: {step['name']}")
            return result
        except Exception as e:
            print(f"  ❌ Step failed: {step['name']} - {e}")
            # Handle failure (retry, skip, abort)
            return await self.handle_step_failure(step, e)
    
    async def handle_step_failure(self, step: Dict, error: Exception):
        """Handle worker failures"""
        # Strategy 1: Retry
        retries = step.get("retries", 0)
        if retries < 3:
            print(f"  🔄 Retrying step: {step['name']}")
            step["retries"] = retries + 1
            await asyncio.sleep(1)  # Brief delay
            return await self.execute_step(step)
        
        # Strategy 2: Use fallback worker
        fallback = step.get("fallback_worker")
        if fallback and fallback in self.workers:
            print(f"  🔄 Using fallback worker: {fallback}")
            step["worker"] = fallback
            return await self.execute_step(step)
        
        # Strategy 3: Fail gracefully
        raise Exception(f"Step {step['name']} failed after retries: {error}")
    
    async def synthesize_results(self, results: Dict) -> Dict:
        """Combine worker results into final output"""
        return {
            "status": "success",
            "results": results,
            "timestamp": asyncio.get_event_loop().time()
        }

# Worker base class
class Worker:
    def __init__(self, name: str):
        self.name = name
    
    async def execute(self, task_input: Any) -> Any:
        """Override this in specific workers"""
        raise NotImplementedError

---

## 📹 Recommended Videos

- [Orchestrator-Worker Pattern](https://www.youtube.com/watch?v=cJOxQqZQ7AE) — Building centralized agent coordination
- [Multi-Agent Orchestration with LangGraph](https://www.youtube.com/watch?v=E2shqsYwxck) — Practical orchestration patterns
- [CrewAI Multi-Agent Systems](https://www.youtube.com/watch?v=sPzc6hMg7So) — Building agent teams with CrewAI

---

## 📚 Additional Resources

- [LangGraph Docs](https://langchain-ai.github.io/langgraph/) — Multi-agent orchestration framework
- [CrewAI Docs](https://docs.crewai.com/) — Framework for orchestrating role-playing AI agents
- [AutoGen](https://microsoft.github.io/autogen/) — Microsoft's multi-agent conversation framework
- [Agent Protocol](https://agentprotocol.ai/) — Standard interface for agent communication
