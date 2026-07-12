---
title: Responsible Deployment Practices
description: >-
  Deploy AI systems safely — human-in-the-loop patterns, gradual rollouts,
  monitoring dashboards, incident response, and rollback strategies
duration: 35 min
difficulty: advanced
has_code: false
youtube: 'https://www.youtube.com/watch?v=FnM6d7MqpHg'
objectives:
  - Design a human-in-the-loop workflow for high-stakes AI decisions
  - Implement a gradual rollout strategy with safety gates
  - Build monitoring for production AI safety
  - Create an incident response plan for AI failures
---
# Responsible Deployment Practices

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Design human-in-the-loop AI systems | 35 min | Advanced |
| Implement gradual rollout strategies | | |
| Build safety monitoring for production | | |
| Create AI incident response plans | | |

---

## Human-in-the-Loop Patterns

Not every AI output should go directly to the end user. For high-stakes decisions, insert human review points:

### Pattern 1: Human Approval Gate

```
User Request --> AI generates response --> Human reviews --> User sees response

Best for: Medical advice, legal documents, hiring decisions
Tradeoff: Slower, but highest safety
```

### Pattern 2: Confidence-Based Routing

```
User Request --> AI generates response + confidence score
  |
  |-- High confidence (>0.9) --> Send directly to user
  |
  |-- Medium confidence (0.6-0.9) --> AI responds with disclaimer
  |
  +-- Low confidence (<0.6) --> Route to human agent
```

```python
def route_response(ai_response, confidence, threshold_auto=0.9, threshold_human=0.6):
    """Route AI responses based on confidence level."""
    if confidence >= threshold_auto:
        return {
            "action": "send_directly",
            "response": ai_response,
            "disclaimer": None
        }
    elif confidence >= threshold_human:
        return {
            "action": "send_with_disclaimer",
            "response": ai_response,
            "disclaimer": "This is an AI-generated response. "
                         "If you need further assistance, a human agent can help."
        }
    else:
        return {
            "action": "escalate_to_human",
            "response": None,
            "context": ai_response  # Give the human agent the AI's draft
        }
```

### Pattern 3: Post-Hoc Review (Sampling)

```
User Request --> AI responds immediately --> Log response
                                              |
                                              v
                                    Random sample (5-10%)
                                    sent for human review
                                              |
                                              v
                                    Flag issues, retrain, improve
```

---

## Gradual Rollout Strategy

Never deploy an AI feature to 100% of users on day one.

### Phased Rollout

```
Week 1: Internal team only (dogfooding)
  |-- Fix critical issues
  |
Week 2: 1% of users (canary release)
  |-- Monitor error rates, user feedback
  |
Week 3: 10% of users
  |-- Compare metrics against baseline
  |
Week 4: 50% of users
  |-- Final validation
  |
Week 5: 100% of users (GA)
```

### Safety Gates Between Phases

At each phase, check these gates before expanding:

```python
safety_gates = {
    "error_rate": {
        "threshold": 0.02,  # Less than 2% error rate
        "metric": "responses flagged as incorrect / total responses"
    },
    "safety_violations": {
        "threshold": 0,  # Zero tolerance for safety violations
        "metric": "harmful or policy-violating outputs"
    },
    "user_satisfaction": {
        "threshold": 4.0,  # Minimum 4.0/5.0 rating
        "metric": "average user rating of AI responses"
    },
    "escalation_rate": {
        "threshold": 0.15,  # Less than 15% escalated to human
        "metric": "conversations requiring human takeover"
    },
    "latency_p99": {
        "threshold": 5.0,  # Less than 5 seconds for P99
        "metric": "99th percentile response time"
    }
}

def check_safety_gates(current_metrics, gates):
    """Check if all safety gates pass for the next rollout phase."""
    results = {}
    all_pass = True
    for gate_name, gate_config in gates.items():
        current_value = current_metrics.get(gate_name, float('inf'))
        passed = current_value <= gate_config["threshold"]
        results[gate_name] = {
            "passed": passed,
            "current": current_value,
            "threshold": gate_config["threshold"]
        }
        if not passed:
            all_pass = False
    
    return all_pass, results
```

---

## Production Safety Monitoring

### Key Metrics to Track

| Metric | What It Catches | Alert Threshold |
|--------|----------------|-----------------|
| **Refusal rate** | Model refusing valid requests | >10% sudden increase |
| **Hallucination rate** | Factually incorrect outputs | Any increase from baseline |
| **Toxicity score** | Harmful or offensive content | Any non-zero detection |
| **Latency P99** | Performance degradation | >2x baseline |
| **User feedback** | Quality regression | Rating drop >0.5 points |
| **Cost per request** | Unexpected cost spikes | >2x budget |

### Automated Safety Alerts

```python
def check_safety_metrics(metrics, baselines, alert_fn):
    """Monitor safety metrics and fire alerts."""
    alerts = []
    
    # Check toxicity (zero tolerance)
    if metrics["toxicity_detections"] > 0:
        alerts.append({
            "severity": "critical",
            "metric": "toxicity",
            "message": f"Toxic output detected: {metrics['toxicity_detections']} instances",
            "action": "Immediate human review required"
        })
    
    # Check hallucination rate (comparison to baseline)
    hallucination_delta = (
        metrics["hallucination_rate"] - baselines["hallucination_rate"]
    )
    if hallucination_delta > 0.05:  # 5% increase
        alerts.append({
            "severity": "high",
            "metric": "hallucination_rate",
            "message": f"Hallucination rate increased by {hallucination_delta:.1%}",
            "action": "Review recent knowledge base changes"
        })
    
    # Check refusal rate spike
    if metrics["refusal_rate"] > baselines["refusal_rate"] * 2:
        alerts.append({
            "severity": "medium",
            "metric": "refusal_rate",
            "message": "Refusal rate doubled — model may be over-cautious",
            "action": "Check for prompt or safety filter changes"
        })
    
    for alert in alerts:
        alert_fn(alert)
    
    return alerts
```

---

## Incident Response Plan

When an AI system produces harmful output in production:

```
INCIDENT DETECTED (automated alert or user report)
    |
    v
1. ASSESS (within 15 minutes)
   - What happened? What was the harmful output?
   - How many users were affected?
   - Is it ongoing or a one-time event?
    |
    v
2. CONTAIN (within 1 hour)
   - If critical: disable the AI feature, fall back to non-AI flow
   - If high: reduce rollout percentage, increase human review
   - If medium: add the specific input to block list
    |
    v
3. INVESTIGATE (within 24 hours)
   - Root cause analysis: why did the model produce this output?
   - Was it a prompt injection, training data issue, or edge case?
   - How many similar incidents exist in logs?
    |
    v
4. FIX (timeline depends on severity)
   - Implement guardrails to prevent recurrence
   - Update safety filters or system prompt
   - Add to red team test suite
    |
    v
5. POST-MORTEM (within 1 week)
   - Document the incident and resolution
   - Update monitoring to catch similar issues earlier
   - Share learnings with the team
```

---

## Rollback Strategy

Always have a way to instantly revert to a non-AI or previous AI version:

```python
import os

class FeatureFlag:
    """Simple feature flag for AI rollback."""
    
    def __init__(self):
        self.ai_enabled = os.environ.get("AI_FEATURE_ENABLED", "true") == "true"
        self.rollout_percentage = int(os.environ.get("AI_ROLLOUT_PCT", "100"))
    
    def should_use_ai(self, user_id):
        if not self.ai_enabled:
            return False
        # Deterministic assignment based on user ID
        return hash(user_id) % 100 < self.rollout_percentage
    
    def get_response(self, user_id, query, ai_fn, fallback_fn):
        if self.should_use_ai(user_id):
            try:
                return ai_fn(query)
            except Exception:
                return fallback_fn(query)  # Graceful degradation
        return fallback_fn(query)
```

---

## Resources

- **Google SRE Book - Managing Incidents**: Incident response best practices
- **LaunchDarkly Feature Flags**: Production feature flag management
- **PagerDuty Incident Response Guide**: Structured incident management

---

## Key Takeaways

- Use human-in-the-loop patterns for high-stakes AI decisions
- Deploy gradually with safety gates between each rollout phase
- Monitor safety metrics (toxicity, hallucination, refusal rate) continuously
- Have a documented incident response plan before deploying
- Always maintain a rollback strategy — feature flags and graceful degradation

---

## Next Lesson

**Lesson 10: AI Safety Audit — Putting It All Together** — Conduct a comprehensive safety audit of an AI application using everything you have learned.
