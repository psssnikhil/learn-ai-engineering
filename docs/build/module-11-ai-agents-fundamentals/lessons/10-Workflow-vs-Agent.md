---
title: Workflow vs Agent Design
description: >-
  Learn when to use deterministic workflows vs autonomous agents, and how to
  combine both for robust production AI systems
duration: 35 min
difficulty: advanced
has_code: false
module: module-11
objectives:
  - Explain the key differences between workflows and agents
  - Identify use cases best suited for workflows vs agents
  - Implement a simple workflow-based LLM pipeline
  - Design a hybrid system that combines workflow and agent patterns
  - Describe the reliability and cost trade-offs of each approach
---
# Workflow vs Agent Design

## What You'll Learn

By the end of this lesson, you'll understand:
- The fundamental difference between workflows and agents
- When deterministic workflows outperform autonomous agents
- When agents are the right choice
- Hybrid architectures that combine both approaches
- Production reliability considerations

**Time to Complete**: 35 minutes
**Difficulty**: Advanced

---

## Workflows vs Agents: The Core Difference

**Workflow**: A pre-defined sequence of steps where the developer controls the flow. The LLM is a component within a deterministic pipeline.

**Agent**: An autonomous system where the LLM decides what to do next. The developer defines tools and goals, but the LLM controls the flow.

```
WORKFLOW (Developer controls flow)          AGENT (LLM controls flow)
┌──────┐  ┌──────┐  ┌──────┐              ┌──────────────────────┐
│Step 1│→ │Step 2│→ │Step 3│              │ LLM decides next step│
│(LLM) │  │(Code)│  │(LLM) │              │     ↓          ↑     │
└──────┘  └──────┘  └──────┘              │  Execute    Observe   │
                                           │     ↓          ↑     │
Fixed path, predictable                   │  Tool/Action → Result │
                                           └──────────────────────┘
                                           Dynamic path, flexible
```

---

## When to Use Workflows

Workflows excel when the task is well-understood and the steps are predictable.

### Example: Document Processing Pipeline

```python
class DocumentProcessor:
    """A workflow: fixed steps, predictable execution."""

    def __init__(self, llm_client):
        self.client = llm_client

    def process(self, document: str) -> dict:
        # Step 1: Extract key information (LLM)
        extracted = self._extract(document)

        # Step 2: Validate format (code -- no LLM needed)
        validated = self._validate(extracted)

        # Step 3: Classify document type (LLM)
        classification = self._classify(document)

        # Step 4: Generate summary (LLM)
        summary = self._summarize(document, classification)

        # Step 5: Store results (code)
        return {
            "extracted_data": validated,
            "classification": classification,
            "summary": summary
        }

    def _extract(self, document: str) -> dict:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Extract the following fields from this document: title, date, author, key_points.

Document:
{document}

Return JSON."
            }],
            temperature=0
        )
        import json
        return json.loads(response.choices[0].message.content)

    def _validate(self, data: dict) -> dict:
        """Deterministic validation -- no LLM needed."""
        required_fields = ["title", "date", "author", "key_points"]
        for field in required_fields:
            if field not in data:
                data[field] = "MISSING"
        return data

    def _classify(self, document: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Classify this document as one of: REPORT, MEMO, PROPOSAL, OTHER.

{document[:1000]}"
            }],
            temperature=0
        )
        return response.choices[0].message.content.strip()

    def _summarize(self, document: str, doc_type: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"Summarize this {doc_type} in 3 bullet points:

{document}"
            }],
            temperature=0
        )
        return response.choices[0].message.content
```

### Workflow Advantages

- **Predictable**: Same input always follows the same path
- **Debuggable**: Easy to identify which step failed
- **Cost-controlled**: Fixed number of LLM calls per execution
- **Testable**: Each step can be unit tested independently
- **Fast**: No decision overhead between steps

---

## When to Use Agents

Agents excel when the task is open-ended or the steps cannot be predicted in advance.

### Example: Research Agent

```python
class ResearchAgent:
    """An agent: LLM decides what to do next."""

    def __init__(self, llm_client, tools: dict):
        self.client = llm_client
        self.tools = tools

    def research(self, question: str) -> str:
        messages = [{
            "role": "system",
            "content": (
                "You are a research agent. Use your tools to find information "
                "and answer the question thoroughly. When you have enough "
                "information, provide your final answer."
            )
        }, {
            "role": "user",
            "content": question
        }]

        for _ in range(10):  # Max iterations
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=self._format_tools()
            )

            message = response.choices[0].message

            if not message.tool_calls:
                return message.content  # Agent is done

            # Execute tools the agent chose
            messages.append(message)
            for call in message.tool_calls:
                result = self.tools[call.function.name](call.function.arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result
                })

        return "Research could not be completed in the allowed steps."
```

### Agent Advantages

- **Flexible**: Can handle novel, unpredictable tasks
- **Adaptive**: Adjusts approach based on intermediate results
- **Capable**: Can solve complex problems humans did not anticipate
- **Composable**: Add new tools without changing the core logic

---

## The Decision Framework

```
                         Is the task predictable?
                        /                        \
                      YES                         NO
                       |                           |
            Are steps known in advance?    Does it need tools?
              /                \              /           \
            YES                NO           YES            NO
             |                  |             |              |
          WORKFLOW         HYBRID          AGENT       REACTIVE
          (pipeline)    (workflow +      (autonomous     (single
                         agent for       tool use)      LLM call)
                         unknowns)
```

| Factor | Workflow | Agent |
|--------|----------|-------|
| Predictability | High | Low |
| Cost control | Easy | Hard (variable LLM calls) |
| Debugging | Simple (step-by-step) | Complex (non-deterministic) |
| Flexibility | Low (fixed path) | High (dynamic path) |
| Reliability | High | Medium (can get stuck) |
| Development speed | Fast for known tasks | Fast for novel tasks |

---

## Hybrid Architecture

The best production systems often combine both patterns.

```python
class HybridSystem:
    """Workflow for known steps, agent for uncertain ones."""

    def __init__(self, llm_client, tools: dict):
        self.client = llm_client
        self.agent = ResearchAgent(llm_client, tools)

    def process_request(self, request: dict) -> dict:
        request_type = request.get("type")

        if request_type == "summarize":
            # WORKFLOW: known, predictable steps
            return self._summarize_workflow(request["content"])

        elif request_type == "analyze":
            # WORKFLOW with AGENT fallback
            result = self._analyze_workflow(request["content"])
            if result.get("needs_research"):
                # Hand off to agent for the uncertain part
                research = self.agent.research(result["research_question"])
                result["research"] = research
            return result

        elif request_type == "research":
            # AGENT: open-ended, unpredictable
            return {"answer": self.agent.research(request["question"])}

        else:
            # REACTIVE: simple response
            return {"answer": self._simple_response(request["content"])}

    def _summarize_workflow(self, content: str) -> dict:
        """Fixed 2-step workflow."""
        # Step 1: Summarize
        summary = self._llm_call("Summarize in 3 bullets:", content)
        # Step 2: Extract keywords
        keywords = self._llm_call("Extract 5 keywords:", content)
        return {"summary": summary, "keywords": keywords}

    def _analyze_workflow(self, content: str) -> dict:
        """Workflow that may need agent help."""
        analysis = self._llm_call("Analyze this content. If external research is needed, say NEEDS_RESEARCH: [question]", content)
        if "NEEDS_RESEARCH:" in analysis:
            question = analysis.split("NEEDS_RESEARCH:")[-1].strip()
            return {"analysis": analysis, "needs_research": True, "research_question": question}
        return {"analysis": analysis, "needs_research": False}

    def _llm_call(self, instruction: str, content: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"{instruction}

{content}"}],
            temperature=0
        )
        return response.choices[0].message.content

    def _simple_response(self, content: str) -> str:
        return self._llm_call("Respond helpfully:", content)
```

---

## Production Recommendations

1. **Start with workflows** for every task you can predict
2. **Add agents** only where flexibility is required
3. **Put guardrails on agents**: max steps, timeouts, cost limits
4. **Log everything**: agent decisions are harder to debug than workflow steps
5. **Have a human fallback**: when the agent gets stuck, escalate to a person

---

## Resources

- **Anthropic's "Building Effective Agents"** -- Practical guide on when to use workflows vs agents
- **Andrew Ng's Agentic Patterns** -- Four patterns for agentic AI design
- **LangGraph** -- Framework that supports both workflow and agent patterns

---

## Key Takeaways

1. **Workflows are deterministic pipelines** where the developer controls flow -- use them for predictable tasks
2. **Agents are autonomous** where the LLM controls flow -- use them for open-ended tasks
3. **Hybrid is usually best** in production: workflow for the known parts, agent for the uncertain parts
4. **Workflows are more reliable and cheaper** but less flexible
5. **Agents are more capable** but harder to debug and cost-control

## Module Complete!

**Next Module**: Multi-Agent Systems
