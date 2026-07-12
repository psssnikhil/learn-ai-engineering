---
title: Training Transformers
description: Learn techniques for training Transformer models effectively
duration: 35 min
difficulty: advanced
has_code: true
module: module-06
youtube: 'https://www.youtube.com/watch?v=kCc8FmEb1nY'
---
# Training Transformers

## Loss Function

**Cross-Entropy Loss** for next token prediction:

```python
def compute_loss(logits, targets):
    # logits: (batch, seq_len, vocab_size)
    # targets: (batch, seq_len)
    
    log_probs = log_softmax(logits, dim=-1)
    loss = -log_probs.gather(dim=-1, index=targets)
    
    return loss.mean()
```

## Label Smoothing

**Problem**: Model becomes overconfident
**Solution**: Smooth the targets

```python
# Instead of [0, 0, 1, 0]
# Use:      [0.03, 0.03, 0.91, 0.03]

smooth_target = (1 - ε) * true_target + ε / vocab_size
```

## Learning Rate Schedule

**Warmup + Decay**:

```python
def transformer_lr_schedule(step, d_model, warmup_steps=4000):
    arg1 = step ** (-0.5)
    arg2 = step * (warmup_steps ** (-1.5))
    
    lr = (d_model ** (-0.5)) * min(arg1, arg2)
    return lr
```

Pattern: Increase then decrease

## Optimization

**Adam with β₁=0.9, β₂=0.98, ε=10⁻⁹**

```python
optimizer = Adam(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.98),
    eps=1e-9
)
```

## Regularization

1. **Dropout**: 0.1 typical
2. **Attention Dropout**: Drop attention weights
3. **Residual Dropout**: After residual connections
4. **Label Smoothing**: 0.1

## Training Loop

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        src, tgt = batch
        
        # Forward
        output = model(src, tgt[:-1])  # Shift right
        
        # Loss
        loss = criterion(output, tgt[1:])  # Predict next
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

---

## 📹 Recommended Videos

- [Andrej Karpathy: Let's build GPT from scratch](https://www.youtube.com/watch?v=kCc8FmEb1nY) — Complete transformer training walkthrough
- [Training Large Language Models](https://www.youtube.com/watch?v=VPRSBzXzavo) — Stanford CS224N lecture on training
- [Learning Rate Scheduling for Transformers](https://www.youtube.com/watch?v=DE150MslZE0) — Warmup and cosine annealing

---

## 📚 Additional Resources

- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — Harvard NLP line-by-line implementation
- [Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) — Simplest, fastest GPT training repo
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740) — Original paper on FP16 training
