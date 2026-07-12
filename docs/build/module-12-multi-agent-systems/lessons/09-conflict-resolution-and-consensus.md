---
title: Conflict Resolution and Consensus
description: >-
  Resolve disagreements between agents using debate, voting, judge patterns,
  and consensus mechanisms from AutoGen, LangGraph, and CrewAI
duration: 40 min
difficulty: intermediate
has_code: true
module: module-12
---
# Conflict Resolution and Consensus

## 🎯 Learning Objectives

| What You'll Learn | Time | Difficulty |
|-------------------|------|------------|
| Understand why agents produce conflicting outputs | 40 min | Intermediate |
| Apply debate, voting, and judge patterns | | |
| Implement consensus mechanisms in multi-agent workflows | | |
| Decide when conflict resolution justifies the cost | | |

---

## 📚 Why Agents Disagree

Multiple agents working on the same problem often produce **conflicting outputs**. Each LLM has stochastic behavior, different system prompts, and partial context. A security agent flags a pattern as dangerous while a developer agent calls it a standard practice. A research agent cites one study; another cites contradictory evidence.

Ignoring conflicts produces inconsistent final answers. Production systems need explicit **conflict resolution** strategies. Microsoft's [AI Agents for Beginners](https://github.com/microsoft/ai-agents-for-beginners) includes multi-agent debate as a quality improvement technique. The [agents-towards-production](https://github.com/NirDiamant/agents-towards-production) repo implements reviewer loops where a judge agent resolves disagreements.

```
    Problem
       ↓
  ┌────┴────┐
  ↓         ↓
Agent A   Agent B
 "Yes"     "No"
  └────┬────┘
       ↓
  ┌─────────┐
  │  Judge  │  → Final decision
  └─────────┘
```

---

## 🏛️ Resolution Patterns

### 1. Debate Pattern

Agents argue for their positions across multiple rounds. A moderator or judge synthesizes the strongest arguments. AutoGen excels here with multi-agent group chats where agents critique each other's responses.

```python
DEBATE_ROUNDS = 2

async def debate(agent_a, agent_b, topic: str) -> list[dict]:
    history = []
    position_a = await agent_a(f"Argue FOR: {topic}")
    position_b = await agent_b(f"Argue AGAINST: {topic}")
    history.extend([position_a, position_b])

    for round_num in range(DEBATE_ROUNDS):
        rebuttal_a = await agent_a(
            f"Rebut this argument: {position_b['content']}"
        )
        rebuttal_b = await agent_b(
            f"Rebut this argument: {position_a['content']}"
        )
        history.extend([rebuttal_a, rebuttal_b])
        position_a, position_b = rebuttal_a, rebuttal_b

    return history
```

**Best for:** High-stakes decisions, policy analysis, architectural choices.

### 2. Voting Pattern

Each agent casts a vote or score. The majority or weighted average wins. Fast and cheap but shallow — agents do not refine each other's reasoning.

```python
from collections import Counter

def majority_vote(votes: list[str]) -> str:
    counts = Counter(votes)
    winner, count = counts.most_common(1)[0]
    return winner

def weighted_vote(scores: list[dict]) -> str:
    totals: dict[str, float] = {}
    for entry in scores:
        option = entry["option"]
        weight = entry.get("confidence", 1.0)
        totals[option] = totals.get(option, 0) + weight
    return max(totals, key=totals.get)

votes = [
    {"agent": "security", "option": "reject", "confidence": 0.9},
    {"agent": "dev", "option": "approve", "confidence": 0.6},
    {"agent": "qa", "option": "reject", "confidence": 0.8},
]
# Weighted result: "reject"
```

**Best for:** Classification tasks, go/no-go decisions with clear options.

### 3. Judge Pattern

A dedicated **judge agent** (often a stronger model) reviews all proposals and picks or synthesizes the best. LangGraph implements this as a final node that receives all worker outputs.

```python
async def judge_agent(proposals: list[dict], criteria: list[str]) -> dict:
    prompt = f"""
    You are an impartial judge. Evaluate these proposals:
    {proposals}

    Criteria: {', '.join(criteria)}

    Return: chosen_proposal, reasoning, confidence (0-1)
    """
    return await llm_call(prompt)
```

**Best for:** Quality-critical outputs where one authoritative review suffices.

### 4. Consensus / Merge Pattern

Instead of picking one winner, a merger agent **combines** the best elements from each proposal. Useful for reports and code where partial contributions have value.

```python
async def merge_proposals(proposals: list[str]) -> str:
    prompt = f"""
    Merge these agent proposals into one coherent output.
    Resolve contradictions by favoring evidence-backed claims.
    Proposals: {proposals}
    """
    return await llm_call(prompt)
```

---

## 📊 Pattern Comparison

| Pattern | Quality | Cost | Latency | Use When |
|---------|---------|------|---------|----------|
| **Debate** | Highest | High | Slow | Critical decisions |
| **Voting** | Medium | Low | Fast | Clear binary choices |
| **Judge** | High | Medium | Medium | Single best answer needed |
| **Merge** | High | Medium | Medium | Composable outputs |
| **First-wins** | Low | Lowest | Fastest | Low-stakes, speed critical |

---

## 🔄 AutoGen and CrewAI Approaches

### AutoGen Group Debate

AutoGen's group chat lets agents respond to each other for multiple turns. Set `max_round` to cap debate length. A `GroupChatManager` selects speakers and can terminate when consensus language appears ("we agree", "final answer").

### CrewAI Review Tasks

In CrewAI, add a review task after worker tasks. The reviewer agent receives all prior outputs via `context` and produces an approved or revised version.

```python
from crewai import Task

draft_task = Task(
    description="Draft a security assessment for {system}",
    agent=security_analyst,
    expected_output="Security assessment draft",
)

review_task = Task(
    description="Review the draft. Flag conflicts with best practices.",
    agent=security_lead,
    context=[draft_task],
    expected_output="Approved assessment or revision list",
)

revision_task = Task(
    description="Revise based on review feedback",
    agent=security_analyst,
    context=[draft_task, review_task],
    expected_output="Final approved assessment",
)
```

### LangGraph Debate Subgraph

Build a subgraph with debate rounds as nodes, a conditional edge checking for agreement, and a judge node as the exit.

---

## 💻 Conflict Detector and Resolver

```python
from dataclasses import dataclass
from enum import Enum

class ResolutionStrategy(Enum):
    VOTE = "vote"
    JUDGE = "judge"
    MERGE = "merge"

@dataclass
class AgentProposal:
    agent: str
    content: str
    confidence: float

class ConflictResolver:
    def detect_conflict(self, proposals: list[AgentProposal]) -> bool:
        if len(proposals) < 2:
            return False
        # Simple heuristic: low pairwise similarity implies conflict
        stances = [p.content[:50].lower() for p in proposals]
        return len(set(stances)) > 1

    async def resolve(
        self,
        proposals: list[AgentProposal],
        strategy: ResolutionStrategy,
    ) -> str:
        if not self.detect_conflict(proposals):
            return proposals[0].content

        if strategy == ResolutionStrategy.VOTE:
            return weighted_vote([
                {"option": p.content, "confidence": p.confidence}
                for p in proposals
            ])
        elif strategy == ResolutionStrategy.JUDGE:
            result = await judge_agent(
                [{"agent": p.agent, "content": p.content} for p in proposals],
                criteria=["accuracy", "completeness", "actionability"],
            )
            return result["chosen_proposal"]
        else:
            return await merge_proposals([p.content for p in proposals])
```

---

## ⚠️ When NOT to Use Consensus

Consensus mechanisms multiply LLM calls. Skip them when:
- The task has a single correct answer verifiable by code (use a test runner, not a debate)
- Latency budgets are tight
- Agents operate on independent subtasks with no overlap
- A supervisor already reviews output (Lesson 5)

Add consensus when errors are costly: legal review, medical information, financial analysis, security assessments.

---

## 💡 Best Practices

1. **Detect before resolving** — Not every difference is a conflict. Use similarity checks first.
2. **Cap debate rounds** — Two to three rounds usually suffice.
3. **Use stronger models for judges** — The judge should be at least as capable as the debaters.
4. **Log dissent** — Record minority opinions for audit trails.
5. **Define resolution criteria** — Tell the judge what matters: safety, speed, cost, user preference.

---

## 🎓 Key Takeaways

```
✅ Agents disagree due to stochastic outputs, different roles, and partial context
✅ Debate, voting, judge, and merge patterns each suit different conflict types
✅ AutoGen group chat, CrewAI review tasks, and LangGraph judge nodes implement resolution
✅ Consensus improves quality but adds cost — reserve for high-stakes decisions
✅ Always cap rounds and log dissenting opinions
```

---

## 🚀 Next Lesson

**Lesson 10: Building a Multi-Agent System** — Put everything together in an end-to-end production build.

You'll learn:
- 🏗️ Architecture decisions for real projects
- 📋 Step-by-step system assembly
- 🧪 Testing and observability for agent teams
- 🚀 Deployment checklist for multi-agent apps

---

## 📚 Additional Resources

- 📄 [AutoGen Group Chat Patterns](https://microsoft.github.io/autogen/docs/user-guide/core-user-guide/design-patterns/group-chat)
- 💻 [CrewAI Task Review Patterns](https://docs.crewai.com/core-concepts/Tasks/)
- 📖 [Microsoft AI Agents for Beginners — Agent Collaboration](https://github.com/microsoft/ai-agents-for-beginners)
- 🔧 [agents-towards-production quality gates](https://github.com/NirDiamant/agents-towards-production)

---

*⏱️ Estimated time: 40 minutes | 📊 Difficulty: Intermediate | 💻 Includes code examples*
