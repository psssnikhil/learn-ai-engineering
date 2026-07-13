---
title: AI Governance and Regulation
description: >-
  Navigate the evolving landscape of AI regulation — EU AI Act, NIST framework,
  organizational governance structures, and compliance strategies
duration: 30 min
difficulty: intermediate
has_code: false
module: module-16
youtube: 'https://www.youtube.com/watch?v=xoVJKj8lcNQ'
objectives:
  - Identify the risk categories in the EU AI Act
  - Describe the NIST AI Risk Management Framework
  - Design an organizational AI governance structure
  - Map compliance requirements to engineering practices
---
# AI Governance and Regulation

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand major AI regulations and frameworks | 30 min | Intermediate |
| Map regulatory requirements to engineering practices | | |
| Design organizational AI governance | | |
| Build compliance into the development lifecycle | | |

---

## Prerequisites

| Requirement | Details |
|-------------|---------|
| **Module 16, Lessons 1-6** | Safety, bias, privacy, transparency fundamentals |
| **Basic understanding** | Of your organization's AI use cases |
| **No legal expertise required** | This lesson maps regulations to engineering — consult legal for compliance |

**Courses required:**
- Module 16: AI Safety & Ethics (full module recommended)
- Module 13: LLMOps (for monitoring and deployment governance)
- Module 17, Project 8: AI Safety Evaluation Suite (for compliance testing)

---

## The Regulatory Landscape (2025-2026)

AI regulation is evolving rapidly. As an AI engineer, you need to understand what is required — even if you are not a lawyer. Your job is to **implement** the technical controls that regulations demand.

### EU AI Act

The EU AI Act is the first comprehensive AI regulation. It classifies AI systems by risk level:

| Risk Level | Examples | Requirements |
|------------|----------|-------------|
| **Unacceptable** | Social scoring, real-time biometric surveillance | Banned |
| **High Risk** | Hiring tools, credit scoring, medical devices | Conformity assessment, documentation, human oversight |
| **Limited Risk** | Chatbots, AI-generated content | Transparency obligations (disclose AI use) |
| **Minimal Risk** | Spam filters, AI in games | No specific requirements |

### Key Engineering Implications

If your AI system falls under "high risk":

```
Required Technical Measures:
1. Risk management system (ongoing, documented)
2. Data governance (training data quality, bias testing)
3. Technical documentation (model cards, system architecture)
4. Record-keeping (logging, audit trails)
5. Transparency (user-facing explanations)
6. Human oversight (ability to override AI decisions)
7. Accuracy, robustness, cybersecurity measures
```

### US Regulatory Approach

The US does not have a single comprehensive AI law. Instead, regulation is fragmented:

| Framework | Scope | Status |
|-----------|-------|--------|
| **NIST AI RMF** | Voluntary risk management | Active, widely adopted |
| **Executive Order 14110** | Federal agency AI use | Active |
| **State laws** (CA, CO, IL) | Automated decision-making, deepfakes | Varies by state |
| **Sector-specific** (FDA, SEC, EEOC) | Healthcare, finance, hiring | Active enforcement |

---

## What You'll Build

By the end of this lesson, you will have:

- [ ] A risk classification for a sample AI application
- [ ] A model card documenting capabilities and limitations
- [ ] An AI review checklist for pre-deployment governance
- [ ] A compliance mapping from regulation to engineering controls
- [ ] A governance structure diagram for a small AI team

---

## Architecture

### NIST AI Risk Management Framework

```
Four Core Functions:

GOVERN  -->  MAP  -->  MEASURE  -->  MANAGE
  |           |          |            |
  v           v          v            v
Policies   Identify   Quantify    Mitigate
and        risks      risks       risks
oversight
```

**Govern**: Establish policies, roles, and accountability
**Map**: Identify and categorize AI risks for your context
**Measure**: Assess and monitor risks quantitatively
**Manage**: Implement controls and mitigations

### Governance Structure

```
[Executive Sponsor]
        |
        v
[AI Ethics Board / Review Committee]
        |
        +-- AI Ethics Lead (policy, review)
        +-- AI Engineers (implement controls)
        +-- Product Managers (impact assessment)
        +-- Legal/Compliance (interpret regulations)
        +-- Data Scientists (bias testing, fairness)
        |
        v
[AI Review Process]  (before every deployment)
        |
        +-- Impact Assessment
        +-- Technical Review (safety eval, bias test)
        +-- Documentation (model card, architecture)
        +-- Approval (sign-off for high-risk)
        |
        v
[Deployed AI System]
        |
        +-- Monitoring (drift, bias, errors)
        +-- Incident Response
        +-- Periodic Re-evaluation
```

---

## Step 1: Classify Your AI System's Risk Level

Use this decision tree to classify any AI application:

```python
# src/risk_classifier.py

RISK_QUESTIONS = [
    {
        "question": "Does the system make or influence decisions about people?",
        "yes_risk": "elevated",
        "categories": ["hiring", "credit", "medical", "legal", "education"],
    },
    {
        "question": "Could a wrong output cause physical harm or significant financial loss?",
        "yes_risk": "high",
        "categories": ["medical", "autonomous", "financial"],
    },
    {
        "question": "Does the system process biometric data or perform surveillance?",
        "yes_risk": "unacceptable",
        "categories": ["biometric", "surveillance"],
    },
    {
        "question": "Is the system a chatbot or content generator interacting with users?",
        "yes_risk": "limited",
        "categories": ["chatbot", "content_generation"],
    },
    {
        "question": "Does the system automate decisions without human review?",
        "yes_risk": "high",
        "categories": ["automated_decision"],
    },
]

def classify_risk(answers: dict) -> dict:
    """Classify AI system risk based on questionnaire answers."""
    risk_level = "minimal"
    applicable_regulations = []
    required_controls = []

    for q in RISK_QUESTIONS:
        if answers.get(q["question"], False):
            if q["yes_risk"] == "unacceptable":
                risk_level = "unacceptable"
                applicable_regulations.append("EU AI Act: BANNED")
                break
            elif q["yes_risk"] == "high" and risk_level != "unacceptable":
                risk_level = "high"
            elif q["yes_risk"] == "elevated" and risk_level in ("minimal", "limited"):
                risk_level = "elevated"
            elif q["yes_risk"] == "limited" and risk_level == "minimal":
                risk_level = "limited"

    if risk_level == "high":
        required_controls = [
            "conformity_assessment", "technical_documentation",
            "record_keeping", "human_oversight", "bias_testing",
            "accuracy_monitoring", "cybersecurity",
        ]
    elif risk_level == "limited":
        required_controls = ["ai_disclosure", "content_labeling"]
    elif risk_level == "elevated":
        required_controls = ["bias_testing", "human_oversight", "audit_trail"]

    return {
        "risk_level": risk_level,
        "applicable_regulations": applicable_regulations,
        "required_controls": required_controls,
    }

# Example: Customer support chatbot
result = classify_risk({
    "Does the system make or influence decisions about people?": False,
    "Is the system a chatbot or content generator interacting with users?": True,
})
print(f"Risk level: {result['risk_level']}")
print(f"Required controls: {result['required_controls']}")
```

---

## Step 2: Create a Model Card

A model card documents your AI system's capabilities, limitations, and intended use. Required for high-risk systems under the EU AI Act.

```python
# src/model_card.py
from dataclasses import dataclass, field
from datetime import date

@dataclass
class ModelCard:
    name: str
    version: str
    base_model: str
    training_data_description: str
    intended_use: list[str]
    out_of_scope: list[str]
    limitations: list[str]
    bias_evaluation: str
    monitoring_plan: str
    last_updated: str = field(default_factory=lambda: date.today().isoformat())
    owner: str = ""

    def to_markdown(self) -> str:
        lines = [
            f"# Model Card: {self.name}",
            "",
            f"**Version:** {self.version}",
            f"**Last updated:** {self.last_updated}",
            f"**Owner:** {self.owner}",
            "",
            "## Model Details",
            f"- Base model: {self.base_model}",
            f"- Training data: {self.training_data_description}",
            "",
            "## Intended Use",
        ]
        for use in self.intended_use:
            lines.append(f"- {use}")
        lines.extend(["", "## Out of Scope (NOT intended for)"])
        for item in self.out_of_scope:
            lines.append(f"- {item}")
        lines.extend(["", "## Limitations"])
        for lim in self.limitations:
            lines.append(f"- {lim}")
        lines.extend([
            "", "## Bias and Fairness", self.bias_evaluation,
            "", "## Monitoring", self.monitoring_plan,
        ])
        return "\n".join(lines)

support_bot_card = ModelCard(
    name="Customer Support Assistant",
    version="1.2.0",
    base_model="GPT-4.1-mini (fine-tuned)",
    training_data_description="5,000 curated support conversations from 2024-2025",
    intended_use=[
        "Answer customer questions about products and policies",
        "Escalate complex issues to human agents",
    ],
    out_of_scope=[
        "Medical, legal, or financial advice",
        "Processing payments or modifying accounts",
        "Accessing real-time order status",
    ],
    limitations=[
        "May hallucinate product details not in knowledge base",
        "Performance degrades for non-English queries",
        "Cannot access real-time data (uses RAG over static docs)",
    ],
    bias_evaluation=(
        "Tested across 12 demographic groups: no significant performance "
        "disparities detected. Known gap: lower accuracy on technical jargon "
        "from non-native English speakers."
    ),
    monitoring_plan=(
        "Human review of 5% of conversations weekly. "
        "Automated hallucination detection on all responses. "
        "Monthly bias re-evaluation with updated test sets."
    ),
    owner="AI Engineering Team",
)

print(support_bot_card.to_markdown())
```

---

## Step 3: Build the AI Review Process

Before deploying any AI feature, run it through a structured review:

```python
# src/ai_review.py
from dataclasses import dataclass, field
from enum import Enum

class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CONDITIONAL = "conditional"

@dataclass
class AIReview:
    project_name: str
    risk_level: str
    impact_assessment: dict = field(default_factory=dict)
    technical_review: dict = field(default_factory=dict)
    documentation: dict = field(default_factory=dict)
    status: ReviewStatus = ReviewStatus.PENDING
    conditions: list[str] = field(default_factory=list)

    def run_impact_assessment(self, answers: dict) -> bool:
        self.impact_assessment = answers
        high_impact = answers.get("affects_vulnerable_groups", False)
        auto_decisions = answers.get("automated_decisions", False)
        return not (high_impact and auto_decisions)

    def run_technical_review(self, results: dict) -> bool:
        self.technical_review = results
        passed = True
        if results.get("safety_pass_rate", 0) < 0.8:
            passed = False
            self.conditions.append("Safety pass rate must exceed 80%")
        if results.get("bias_disparity", 0) > 0.1:
            passed = False
            self.conditions.append("Bias disparity must be below 10%")
        return passed

    def approve(self) -> ReviewStatus:
        if self.conditions:
            self.status = ReviewStatus.CONDITIONAL
        else:
            self.status = ReviewStatus.APPROVED
        return self.status

# Example review
review = AIReview(project_name="Support Chatbot v2", risk_level="limited")
review.run_impact_assessment({
    "affects_vulnerable_groups": False,
    "automated_decisions": False,
    "failure_impact": "minor_inconvenience",
})
review.run_technical_review({
    "safety_pass_rate": 0.92,
    "bias_disparity": 0.03,
    "hallucination_rate": 0.05,
})
status = review.approve()
print(f"Review status: {status.value}")
```

### The AI Review Checklist

```
1. Impact Assessment
   - [ ] Who is affected by this AI system?
   - [ ] What happens if it fails or produces wrong output?
   - [ ] Is this a high-risk application under EU AI Act?
   - [ ] Are vulnerable groups disproportionately affected?

2. Technical Review
   - [ ] Safety evaluation pass rate > 80%
   - [ ] Bias evaluation across demographic groups
   - [ ] Security testing (prompt injection, jailbreaks)
   - [ ] Performance benchmarks on diverse inputs
   - [ ] Confidence calibration verified

3. Documentation
   - [ ] Model card completed
   - [ ] System architecture documented
   - [ ] Data flow diagram with human oversight points
   - [ ] Incident response plan written

4. Approval
   - [ ] Ethics lead sign-off (required for high-risk)
   - [ ] Deployment plan with monitoring and rollback
   - [ ] User disclosure implemented (AI labeling)
```

---

## Step 4: Map Compliance to Engineering

Translate regulatory requirements into concrete engineering tasks:

```python
COMPLIANCE_MAP = {
    "ai_disclosure": {
        "regulation": "EU AI Act (Limited Risk)",
        "engineering_task": "Add 'Generated by AI' label to all AI outputs",
        "implementation": "UI badge + API response metadata",
        "test": "Verify label present in all response paths",
    },
    "record_keeping": {
        "regulation": "EU AI Act (High Risk)",
        "engineering_task": "Log all AI decisions with full context",
        "implementation": "AIDecisionLogger from Lesson 6",
        "test": "Verify 100% of decisions have audit trail entries",
    },
    "human_oversight": {
        "regulation": "EU AI Act (High Risk)",
        "engineering_task": "Allow human override of AI decisions",
        "implementation": "Review queue for low-confidence decisions",
        "test": "Verify human can override and override is logged",
    },
    "bias_testing": {
        "regulation": "EU AI Act (High Risk), EEOC",
        "engineering_task": "Test performance across demographic groups",
        "implementation": "Safety eval suite with bias test category",
        "test": "Performance disparity < 10% across groups",
    },
    "accuracy_monitoring": {
        "regulation": "EU AI Act (High Risk)",
        "engineering_task": "Ongoing accuracy measurement in production",
        "implementation": "Weekly eval runs + user feedback tracking",
        "test": "Accuracy does not degrade > 5% month-over-month",
    },
    "cybersecurity": {
        "regulation": "EU AI Act (High Risk)",
        "engineering_task": "Protect against prompt injection and data leakage",
        "implementation": "Input sanitization + output filtering + safety eval CI",
        "test": "Prompt injection test suite passes at 95%+",
    },
}
```

---

## Testing Your Build

### Compliance Verification Checklist

- [ ] Risk level classified for your AI application
- [ ] Model card completed with all required sections
- [ ] AI review process documented and followed
- [ ] Required controls mapped to engineering implementations
- [ ] Safety evaluation suite integrated into CI/CD
- [ ] Audit trail logging active for all AI decisions
- [ ] AI disclosure visible to end users
- [ ] Human override mechanism tested

### Self-Assessment Questions

1. Can you explain your AI system's risk level and why?
2. If a regulator asked "why did the AI make this decision," can you show the audit trail?
3. Have you tested for bias across demographic groups in the last 30 days?
4. Do users know they are interacting with AI?
5. Can a human override any automated AI decision?

---

## Deployment Notes

### Ongoing Compliance (Not One-Time)

Compliance is a continuous process:

| Activity | Frequency | Owner |
|----------|-----------|-------|
| Safety evaluation suite | Every deployment | AI Engineers |
| Bias re-evaluation | Monthly | Data Scientists |
| Model card update | On every model change | AI Engineers |
| Audit trail review | Weekly (sample) | Ethics Lead |
| Full AI review | On every new feature | Review Committee |
| Regulatory scan | Quarterly | Legal/Compliance |

### Incident Response Plan

```
AI Incident Detected
    |
    v
1. CONTAIN: Disable affected feature / route to human
2. ASSESS: Review audit trail, determine scope and severity
3. NOTIFY: Alert ethics lead, legal (if required), affected users
4. FIX: Root cause analysis, implement fix, re-run safety eval
5. DOCUMENT: Post-incident report, update model card
6. PREVENT: Add test case to safety suite, update monitoring
```

---

## Extensions and Challenges

- **Compliance automation**: Build a dashboard that tracks compliance status across all AI systems
- **Regulatory change monitoring**: Subscribe to AI law updates and map new requirements automatically
- **Cross-border deployment**: Document which regulations apply when serving EU vs US users
- **Third-party AI audit**: Prepare documentation packages for external auditors
- **Privacy impact assessment**: Combine AI governance with GDPR DPIA for data-heavy systems

---

## Compliance Checklist for AI Engineers

When building AI applications, use this checklist:

- [ ] **Disclosure**: Users know they are interacting with AI
- [ ] **Data handling**: PII is protected, data retention policies followed
- [ ] **Bias testing**: Tested on diverse demographic groups
- [ ] **Security**: Prompt injection and jailbreak protections in place
- [ ] **Logging**: All AI decisions logged with context
- [ ] **Human oversight**: Humans can review and override AI decisions
- [ ] **Documentation**: Model card and system documentation maintained
- [ ] **Incident response**: Plan exists for AI failures or harmful outputs
- [ ] **Monitoring**: Ongoing monitoring for drift, bias, and errors
- [ ] **User rights**: Users can request explanations and contest decisions

---

## Key Takeaways

- The EU AI Act classifies AI by risk level and imposes requirements accordingly
- NIST AI RMF provides a voluntary framework: Govern, Map, Measure, Manage
- Every AI team needs clear governance roles and a deployment review process
- Model cards document capabilities, limitations, and bias testing results
- Compliance is an ongoing process, not a one-time checklist
- As an AI engineer, you implement the technical controls — document everything

---

## Next Lesson

**Lesson 8: Red Teaming and Adversarial Testing** — Learn systematic approaches to finding vulnerabilities in AI systems before attackers do.
