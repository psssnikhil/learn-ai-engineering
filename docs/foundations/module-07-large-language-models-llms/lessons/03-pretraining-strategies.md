---
title: Pre-training Strategies
description: Master the techniques used to pre-train large language models
duration: 35 min
difficulty: advanced
has_code: false
module: module-07
youtube: 'https://www.youtube.com/watch?v=UU1WVnMk4E8'
---
# Pre-training Strategies

## What is Pre-training?

**Goal**: Learn general language understanding from unlabeled text

**Data**: Books, websites, code, papers (~trillions of tokens!)

**Compute**: Thousands of GPUs, weeks/months of training

## 1. Autoregressive Language Modeling (GPT)

**Objective**: Predict next token

```python
def next_token_prediction_loss(model, tokens):
    # tokens: [batch_size, seq_len]
    
    # Forward pass
    logits = model(tokens[:, :-1])  # Input: all but last
    targets = tokens[:, 1:]          # Target: all but first
    
    # Cross-entropy loss
    loss = cross_entropy(logits, targets)
    return loss
```

**Example**:
```
Text: "The cat sat on the mat"

Training samples:
"The" → "cat"
"The cat" → "sat"
"The cat sat" → "on"
"The cat sat on" → "the"
"The cat sat on the" → "mat"
```

## 2. Masked Language Modeling (BERT)

**Objective**: Predict masked tokens

```python
def masked_lm_loss(model, tokens):
    # Randomly mask 15% of tokens
    masked_tokens, mask_indices = mask_tokens(tokens, mask_prob=0.15)
    
    # Forward pass
    logits = model(masked_tokens)
    
    # Loss only on masked positions
    loss = cross_entropy(logits[mask_indices], tokens[mask_indices])
    return loss
```

**Masking strategy**:
- 80%: Replace with [MASK]
- 10%: Replace with random token
- 10%: Keep original

## 3. Span Corruption (T5)

**Objective**: Predict masked spans

```
Input:  "Thank you for <X> me to your party <Y> week"
Target: "<X> inviting <Y> last <Z>"
```

Learns to handle variable-length predictions!

## 4. Prefix Language Modeling

Hybrid: Some tokens bidirectional, rest causal

## Training Process

```python
# Simplified pre-training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward
        logits = model(batch['input_ids'])
        
        # Calculate loss
        loss = compute_loss(logits, batch['labels'])
        
        # Backward
        loss.backward()
        
        # Update
        optimizer.step()
        optimizer.zero_grad()
        
        # Log progress
        if step % 100 == 0:
            print(f"Step {step}, Loss: {loss:.4f}")
```

## Data Sources

**Common Crawl**: Web scrape (~petabytes)
**Books**: BookCorpus, Project Gutenberg
**Wikipedia**: All languages
**GitHub**: Code repositories
**Papers**: arXiv, PubMed
**Social**: Reddit, Twitter/X

## Data Cleaning

Critical steps:
1. **Deduplication**: Remove duplicate text
2. **Quality filtering**: Remove low-quality content
3. **Toxicity filtering**: Remove harmful content
4. **PII removal**: Remove personal information
5. **Language filtering**: Keep target languages

## Compute Requirements

**GPT-3 (175B)**:
- Hardware: 10,000+ V100 GPUs
- Time: ~34 days
- Cost: ~$4.6 million
- Energy: ~1,287 MWh

**LLaMA-2 70B**:
- Hardware: 2,000 A100 GPUs
- Time: ~21 days
- Cost: ~$1 million

---

## 📹 Recommended Videos

- [Pre-training Large Language Models](https://www.youtube.com/watch?v=UU1WVnMk4E8) — Stanford CS324 lecture on pre-training
- [How GPT Models are Trained](https://www.youtube.com/watch?v=VPRSBzXzavo) — Stanford CS224N on LLM training
- [LLM Training at Scale](https://www.youtube.com/watch?v=Rp3A5q9L_bg) — Practical challenges of large-scale training

---

## 📚 Additional Resources

- [The GPT-3 Paper](https://arxiv.org/abs/2005.14165) — Language models are few-shot learners
- [LLaMA Paper](https://arxiv.org/abs/2302.13971) — Open and efficient foundation LMs
- [Lilian Weng: Large Language Model Training](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) — Comprehensive survey
