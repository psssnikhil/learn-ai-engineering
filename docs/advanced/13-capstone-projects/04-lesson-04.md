---
title: 'Project 3: Multi-Agent Research System'
description: >-
  Build a team of AI agents that collaborate to research topics — a planner,
  researcher, writer, and fact-checker working together via message passing
duration: 150 min
difficulty: advanced
has_code: false
youtube: 'https://www.youtube.com/watch?v=hvAPnpSfSGo'
objectives:
  - Design a multi-agent architecture with distinct roles
  - Implement message passing between agents
  - Build a research agent with web search capabilities
  - Orchestrate agents to produce a fact-checked research report
---
# Project 3: Multi-Agent Research System

## Project Overview

Build a multi-agent system where specialized AI agents collaborate to research a topic and produce a comprehensive, fact-checked report:

- **Planner Agent**: Breaks the research topic into sub-questions
- **Researcher Agent**: Searches for information and collects sources
- **Writer Agent**: Synthesizes research into a structured report
- **Fact-Checker Agent**: Verifies claims against sources

**Time estimate**: 15-20 hours
**Skills used**: Multi-Agent Systems, Orchestration, Tool Use, Prompt Engineering

---

## Architecture

```
User: "Research the current state of AI agents in production"
    |
    v
[Orchestrator]
    |
    |-- [Planner Agent]
    |     "Break this into 5 sub-questions"
    |     -> Q1: What frameworks exist?
    |     -> Q2: What are common architectures?
    |     -> Q3: What are the challenges?
    |     -> Q4: What companies use them?
    |     -> Q5: What's the future outlook?
    |
    |-- [Researcher Agent] (runs for each sub-question)
    |     Uses tools: web_search, read_url
    |     -> Findings + source URLs for each question
    |
    |-- [Writer Agent]
    |     Combines all research into structured report
    |     -> Draft report with citations
    |
    |-- [Fact-Checker Agent]
    |     Verifies each claim against sources
    |     -> Verified report + confidence ratings
    |
    v
Final Report (markdown with citations and confidence)
```

---

## Step 1: Define Agent Base Class

```python
from openai import OpenAI
import json

client = OpenAI()

class Agent:
    """Base class for all agents in the system."""
    
    def __init__(self, name: str, system_prompt: str, model: str = "gpt-4.1"):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.message_history = []
    
    def run(self, user_message: str, tools: list | None = None) -> str:
        """Run the agent with a message and optional tools."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        kwargs = {"model": self.model, "messages": messages, "temperature": 0.3}
        if tools:
            kwargs["tools"] = tools
        
        response = client.chat.completions.create(**kwargs)
        result = response.choices[0].message.content
        
        self.message_history.append({"input": user_message, "output": result})
        return result
```

---

## Step 2: Create Specialized Agents

```python
planner = Agent(
    name="Planner",
    system_prompt="""You are a research planner. Given a topic, break it into 
5-7 specific, researchable sub-questions. Each question should be focused enough 
to research independently but together they cover the topic comprehensively.

Respond as a JSON array of strings, each being a sub-question."""
)

researcher = Agent(
    name="Researcher",
    system_prompt="""You are a research analyst. Given a specific question, 
provide a thorough answer based on your knowledge. Include:
1. Key facts and findings
2. Specific examples and data points
3. Source references where possible

Be factual and specific. Clearly distinguish between established facts and 
your analysis. Respond in structured markdown."""
)

writer = Agent(
    name="Writer",
    system_prompt="""You are a technical writer. Given research findings on 
multiple sub-topics, synthesize them into a coherent, well-structured report.

Requirements:
1. Executive summary (3-4 sentences)
2. Structured sections with headers
3. Inline citations using [Source N] notation
4. Conclusion with key takeaways
5. References section at the end

Write in a professional, informative tone."""
)

fact_checker = Agent(
    name="Fact-Checker",
    system_prompt="""You are a fact-checker. Given a report and the source 
research it was based on, verify each major claim.

For each claim, provide:
1. The claim (quoted from the report)
2. Verdict: VERIFIED / UNVERIFIED / NEEDS CONTEXT
3. Confidence: HIGH / MEDIUM / LOW
4. Reasoning: Why you rated it this way

Respond as a JSON array of objects with these fields."""
)
```

---

## Step 3: Build the Orchestrator

```python
class ResearchOrchestrator:
    """Orchestrates the multi-agent research pipeline."""
    
    def __init__(self):
        self.planner = planner
        self.researcher = researcher
        self.writer = writer
        self.fact_checker = fact_checker
    
    def research(self, topic: str) -> dict:
        """Run the full research pipeline."""
        print(f"Researching: {topic}
")
        
        # Phase 1: Planning
        print("Phase 1: Planning sub-questions...")
        plan_result = self.planner.run(
            f"Break this research topic into sub-questions: {topic}"
        )
        sub_questions = json.loads(plan_result)
        print(f"  Generated {len(sub_questions)} sub-questions")
        
        # Phase 2: Research each sub-question
        print("
Phase 2: Researching each sub-question...")
        research_results = []
        for i, question in enumerate(sub_questions):
            print(f"  Researching Q{i+1}: {question[:60]}...")
            result = self.researcher.run(question)
            research_results.append({
                "question": question,
                "findings": result
            })
        
        # Phase 3: Write the report
        print("
Phase 3: Writing report...")
        research_context = "

---

".join(
            f"## Research on: {r['question']}

{r['findings']}"
            for r in research_results
        )
        report = self.writer.run(
            f"Write a comprehensive report on '{topic}' based on this research:

{research_context}"
        )
        
        # Phase 4: Fact-check
        print("
Phase 4: Fact-checking...")
        fact_check_input = (
            f"Report:
{report}

"
            f"Source Research:
{research_context}"
        )
        fact_check_result = self.fact_checker.run(fact_check_input)
        
        try:
            fact_checks = json.loads(fact_check_result)
        except json.JSONDecodeError:
            fact_checks = [{"raw": fact_check_result}]
        
        return {
            "topic": topic,
            "sub_questions": sub_questions,
            "research": research_results,
            "report": report,
            "fact_checks": fact_checks,
        }

# Usage
orchestrator = ResearchOrchestrator()
result = orchestrator.research("The current state of AI agents in production applications")
print(result["report"])
```

---

## Step 4: Add Quality Metrics

```python
def evaluate_research(result: dict) -> dict:
    """Evaluate the quality of the research output."""
    report = result["report"]
    fact_checks = result["fact_checks"]
    
    metrics = {
        "report_length": len(report.split()),
        "sections_count": report.count("##"),
        "citations_count": report.count("[Source"),
        "sub_questions": len(result["sub_questions"]),
    }
    
    # Fact-check summary
    if isinstance(fact_checks, list) and all(isinstance(fc, dict) for fc in fact_checks):
        verdicts = [fc.get("verdict", "unknown") for fc in fact_checks]
        metrics["verified_claims"] = verdicts.count("VERIFIED")
        metrics["unverified_claims"] = verdicts.count("UNVERIFIED")
        metrics["needs_context"] = verdicts.count("NEEDS CONTEXT")
        metrics["total_claims_checked"] = len(fact_checks)
        if metrics["total_claims_checked"] > 0:
            metrics["verification_rate"] = (
                metrics["verified_claims"] / metrics["total_claims_checked"]
            )
    
    return metrics
```

---

## Extension Ideas

- Add a real web search tool (Tavily, Brave Search, or SerpAPI)
- Implement agent-to-agent debate for controversial findings
- Add a revision loop where the writer incorporates fact-checker feedback
- Build a Streamlit UI for interactive research sessions
- Add caching to avoid re-researching the same questions
- Implement parallel research with asyncio

---

## Resources

- **LangGraph Documentation**: Framework for multi-agent orchestration
- **CrewAI**: Multi-agent framework for reference
- **Autogen by Microsoft**: Another multi-agent framework approach
- **"Multi-Agent Debate" Paper**: Improving factual accuracy through agent debate

---

## Next Project

**Project 4: AI-Powered Content Platform** — Build a full-stack application that generates, edits, and manages content using RAG and fine-tuned models.
