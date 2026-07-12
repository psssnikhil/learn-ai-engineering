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

## The Regulatory Landscape (2025)

AI regulation is evolving rapidly. As an AI engineer, you need to understand what is required — even if you are not a lawyer.

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

### NIST AI Risk Management Framework (AI RMF)

The US approach is currently voluntary, led by NIST's AI RMF:

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

---

## Building an AI Governance Structure

### Roles and Responsibilities

| Role | Responsibility |
|------|---------------|
| **AI Ethics Lead** | Set policy, review high-risk deployments |
| **AI Engineers** | Implement safety measures, run evaluations |
| **Product Managers** | Assess user impact, define acceptable behavior |
| **Legal/Compliance** | Interpret regulations, manage risk |
| **Data Scientists** | Bias testing, fairness audits |

### The AI Review Process

Before deploying any AI feature, run it through a structured review:

```
1. Impact Assessment
   - Who is affected by this AI system?
   - What happens if it fails or produces wrong output?
   - Is this a high-risk application?

2. Technical Review
   - Bias evaluation results
   - Security testing (prompt injection, jailbreaks)
   - Performance benchmarks on diverse inputs
   - Confidence calibration

3. Documentation
   - Model card (what model, what data, known limitations)
   - System architecture (data flow, human oversight points)
   - Incident response plan

4. Approval
   - Sign-off from ethics lead for high-risk applications
   - Deployment plan with monitoring and rollback strategy
```

---

## Model Cards and Documentation

A model card documents your AI system's capabilities, limitations, and intended use:

```markdown
# Model Card: Customer Support Assistant

## Model Details
- Base model: GPT-4.1-mini (fine-tuned)
- Training data: 5,000 curated support conversations
- Last updated: 2025-03-15

## Intended Use
- Answer customer questions about products and policies
- Escalate complex issues to human agents
- NOT intended for medical, legal, or financial advice

## Limitations
- May hallucinate product details not in knowledge base
- Performance degrades for non-English queries
- Cannot access real-time order status (uses RAG over docs)

## Bias and Fairness
- Tested across 12 demographic groups: no significant
  performance disparities detected
- Known gap: lower accuracy on technical jargon from
  non-native English speakers

## Monitoring
- Human review of 5% of conversations weekly
- Automated hallucination detection on all responses
- Monthly bias re-evaluation
```

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

## Resources

- **EU AI Act Full Text**: Official regulation document
- **NIST AI RMF**: https://www.nist.gov/artificial-intelligence
- **Anthropic's Responsible Scaling Policy**: Example of industry self-governance
- **Google Model Cards**: Examples of well-documented AI systems

---

## Key Takeaways

- The EU AI Act classifies AI by risk level and imposes requirements accordingly
- NIST AI RMF provides a voluntary framework: Govern, Map, Measure, Manage
- Every AI team needs clear governance roles and a deployment review process
- Model cards document capabilities, limitations, and bias testing results
- Compliance is an ongoing process, not a one-time checklist

---

## Next Lesson

**Lesson 8: Red Teaming and Adversarial Testing** — Learn systematic approaches to finding vulnerabilities in AI systems before attackers do.
