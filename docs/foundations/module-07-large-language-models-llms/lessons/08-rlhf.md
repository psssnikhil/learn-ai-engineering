---
title: RLHF (Reinforcement Learning from Human Feedback)
description: >-
  Understand how models are aligned with human preferences using reinforcement
  learning
duration: 35 min
difficulty: advanced
has_code: false
module: module-07
youtube: 'https://www.youtube.com/watch?v=2MBJOuVq380'
---
# RLHF: Aligning LLMs with Human Values

## The Challenge

**Instruction tuning** makes models follow instructions
**BUT**: How do we ensure:
- Helpful responses?
- Harmless content?
- Honest answers (no hallucinations)?

**Answer**: RLHF!

## The RLHF Pipeline

### Step 1: Supervised Fine-Tuning (SFT)
```
Base Model → SFT → Instruction-Following Model
```

### Step 2: Reward Model Training
```
For each prompt, generate multiple responses
Human ranks: Response A > Response B > Response C

Train reward model to predict these rankings
```

### Step 3: RL Optimization (PPO)
```
Use reward model to improve policy via RL

Prompt → Model → Response → Reward → Update Model
```

## Reward Model

```python
# Simplified reward model
class RewardModel:
    def score(self, prompt, response):
        # Returns score: higher = better
        features = extract_features(prompt, response)
        score = self.neural_net(features)
        return score

# Training
for (prompt, response_A, response_B, preference) in dataset:
    score_A = reward_model.score(prompt, response_A)
    score_B = reward_model.score(prompt, response_B)
    
    if preference == "A":
        loss = max(0, score_B - score_A + margin)
    else:
        loss = max(0, score_A - score_B + margin)
    
    loss.backward()
    optimizer.step()
```

## PPO (Proximal Policy Optimization)

**Goal**: Maximize reward while staying close to original policy

```
Objective = E[reward] - β × KL(π_new || π_old)

reward: From reward model
KL divergence: Prevents drastic changes
```

## Challenges

1. **Reward hacking**: Model exploits reward
2. **Human feedback expensive**: Needs many annotations
3. **Alignment tax**: Can reduce capabilities
4. **Diverse preferences**: People disagree!

## Recent Advances

**DPO (Direct Preference Optimization)**:
- Simpler than PPO
- No separate reward model
- Directly optimize preferences

**Constitutional AI (Anthropic)**:
- AI self-critiques responses
- Follows written principles

---

## 📹 Recommended Videos

- [RLHF Explained Simply](https://www.youtube.com/watch?v=2MBJOuVq380) — Hugging Face RLHF walkthrough
- [How ChatGPT is Trained with RLHF](https://www.youtube.com/watch?v=VPRSBzXzavo) — End-to-end training pipeline
- [DPO: Direct Preference Optimization](https://www.youtube.com/watch?v=XZLc09hkMwA) — RLHF without the RL

---

## 📚 Additional Resources

- [RLHF Paper (InstructGPT)](https://arxiv.org/abs/2203.02155) — Training LMs to follow instructions with human feedback
- [DPO Paper](https://arxiv.org/abs/2305.18290) — Direct preference optimization: your LM is secretly a reward model
- [Constitutional AI Paper](https://arxiv.org/abs/2212.08073) — Anthropic's approach to AI alignment
- [Hugging Face TRL Library](https://huggingface.co/docs/trl) — Transformer reinforcement learning
