---
title: Training Best Practices & Optimization
description: >-
  Master the full toolkit for training neural networks effectively — weight
  initialization, learning rate schedules, gradient clipping, AdamW, mixed
  precision, and a systematic debugging protocol.
duration: 60 min
difficulty: intermediate
has_code: true
module: module-05
youtube: 'https://www.youtube.com/watch?v=pZEHXsizR7I'
objectives:
  - Derive the correct variance for Xavier and He initialization
  - Implement cosine warmup and cyclical learning rate schedules
  - Apply gradient clipping correctly and understand why it works
  - Use PyTorch AMP (automatic mixed precision) for 2× speedup
  - Systematically debug common training failures
---

# Training Best Practices & Optimization

## Prerequisites

- [Lesson 04: Gradient Descent](./04-gradient-descent.md) — SGD, momentum, Adam
- [Lesson 06: Overfitting & Regularization](./06-overfitting-regularization.md) — weight decay, dropout
- [Lesson 05: Backpropagation](./05-backpropagation.md) — gradient flow analysis

## What You'll Learn

| Technique | What it fixes | When to use |
|-----------|-------------|-------------|
| He/Xavier init | Vanishing/exploding activations at start | Always |
| Warmup LR schedule | Early instability | Transformers, large LR |
| Cosine decay | Final convergence | Default for most models |
| Gradient clipping | Exploding gradients | RNNs, Transformers |
| AdamW | L2 regularization + Adam interaction | Default optimizer |
| Mixed precision | 2× speed, 2× memory | CUDA GPU training |

---

## Intuition: Why Training Fails

Most training failures have identifiable causes:

```
Loss NaN from step 1    → initialization or LR too high
Loss plateaus at random → wrong architecture or bad data  
Train good, val bad     → overfitting (add regularization)
Both losses plateau     → LR too low or schedule too aggressive
Loss spikes then recovers → LR on the edge; lower it
Loss spikes and stays   → LR too high; use warmup next time
```

A systematic debugging protocol is more effective than random hyperparameter search.

---

## 1. Weight Initialization

Proper initialization ensures that the **variance of activations and gradients is consistent** across layers at the start of training.

### The Signal Propagation Problem

For a linear layer with weight `W ∈ ℝ^{n_out × n_in}` and input `x` where `Var(x_i) = 1`:

```
y = W x     (n_out outputs)

Var(y_j) = Var(Σ_i W_{j,i} · x_i)
         = n_in · Var(W_{j,i}) · Var(x_i)    (assuming independence)
         = n_in · Var(W_{j,i})
```

If `Var(W) = 1`, then `Var(y) = n_in` — variance grows with layer width!
If `Var(W) = 1/n_in`, then `Var(y) = 1` — variance preserved.

```python
import numpy as np


def analyze_activation_variance(
    n_in:      int,
    n_layers:  int,
    init_std:  float,
    activation: str = "relu",
) -> None:
    """
    Simulate forward pass variance through n_layers to see signal propagation.
    """
    np.random.seed(42)
    x = np.random.randn(1000, n_in)   # batch of 1000 inputs

    print(f"Init std={init_std:.4f}, activation={activation}")
    print(f"Layer  0: var(x) = {x.var():.4f}")

    for i in range(n_layers):
        W = np.random.randn(n_in, n_in) * init_std
        x = x @ W

        # Apply activation
        if activation == "relu":
            x = np.maximum(0, x)
        elif activation == "tanh":
            x = np.tanh(x)

        print(f"Layer {i+1:2d}: var(x) = {x.var():.6f}")

        if x.var() < 1e-10:
            print("  → Signal VANISHED (all zeros)")
            break
        if x.var() > 1e10:
            print("  → Signal EXPLODED (overflowing)")
            break


# Bad initialization: std=1
analyze_activation_variance(n_in=512, n_layers=5, init_std=1.0, activation="relu")
# Output: 1.0, 256.0, 65536.0, ... (explodes)

print()
# He initialization for ReLU: std = sqrt(2/n_in)
he_std = np.sqrt(2 / 512)
analyze_activation_variance(n_in=512, n_layers=5, init_std=he_std, activation="relu")
# Output: ~1.0, ~1.0, ~1.0, ~1.0 (stable!)


def he_init(n_in: int, n_out: int) -> np.ndarray:
    """
    He initialization for ReLU layers.

    Var(W) = 2/n_in
    The factor of 2 compensates for ReLU zeroing half the inputs.

    Derivation: ReLU halves the effective input variance,
    so we need 2× the variance to compensate.
    """
    std = np.sqrt(2.0 / n_in)
    return np.random.randn(n_in, n_out) * std   # (n_in, n_out)


def xavier_init(n_in: int, n_out: int) -> np.ndarray:
    """
    Xavier/Glorot initialization for tanh/sigmoid layers.

    Var(W) = 2/(n_in + n_out) — compromise between forward and backward
    Uniform variant: W ~ U[-√(6/(n_in+n_out)), √(6/(n_in+n_out))]
    """
    limit = np.sqrt(6.0 / (n_in + n_out))
    return np.random.uniform(-limit, limit, (n_in, n_out))


# PyTorch uses these by default in nn.Linear
import torch.nn as nn

# nn.Linear uses Kaiming (He) initialization for weights by default
linear = nn.Linear(512, 256)
print(f"PyTorch Linear weight std: {linear.weight.data.std().item():.4f}")
# ≈ sqrt(1/512) = 0.044 (Kaiming uniform variant)
```

---

## 2. Learning Rate Schedules

### The Warmup-Decay Pattern

For Transformer models, the standard schedule is:

```
LR = d_model^{-0.5} × min(step^{-0.5}, step × warmup_steps^{-1.5})

- Phase 1 (steps < warmup): LR increases linearly
- Phase 2 (steps > warmup): LR decays as 1/√step
```

```python
import numpy as np
import matplotlib.pyplot as plt


def noam_schedule(
    step:          int,
    d_model:       int   = 512,
    warmup_steps:  int   = 4000,
) -> float:
    """
    Noam schedule from "Attention Is All You Need" (Vaswani 2017).

    Peak LR ≈ 0.002 for d_model=512, warmup=4000
    """
    step = max(step, 1)
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))


def cosine_warmup_schedule(
    step:          int,
    total_steps:   int,
    warmup_steps:  int,
    peak_lr:       float = 1e-3,
    min_lr:        float = 1e-5,
) -> float:
    """
    Cosine decay with linear warmup — standard for modern models.

    Used by: GPT-2, LLaMA, BERT, most recent models.

    Phase 1: linear warmup  (0 → peak_lr)
    Phase 2: cosine decay   (peak_lr → min_lr)
    """
    if step < warmup_steps:
        # Linear warmup
        return peak_lr * step / warmup_steps

    # Cosine decay after warmup
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    cosine   = 0.5 * (1 + np.cos(np.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine


# Visualize schedules
steps = np.arange(1, 10001)
noam_lrs   = [noam_schedule(s)               for s in steps]
cosine_lrs = [cosine_warmup_schedule(s, 10000, 500) for s in steps]

print("Noam schedule (first 10 steps):")
for s in [1, 100, 500, 1000, 4000, 10000]:
    print(f"  Step {s:5d}: LR = {noam_schedule(s):.6f}")

print("\nCosine warmup (first 10 steps):")
for s in [1, 100, 500, 1000, 5000, 10000]:
    print(f"  Step {s:5d}: LR = {cosine_warmup_schedule(s, 10000, 500):.6f}")


# PyTorch implementation
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer,
    warmup_steps:  int,
    total_steps:   int,
    min_lr_ratio:  float = 0.1,
):
    """Hugging Face-style cosine schedule with warmup."""

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)

        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)
```

---

## 3. Gradient Clipping

Gradient clipping prevents exploding gradients by rescaling the gradient vector when its norm exceeds a threshold:

```python
import torch
import torch.nn as nn


def manual_gradient_clip(
    parameters,
    max_norm: float = 1.0,
) -> float:
    """
    Clip gradients by global L2 norm.

    Algorithm:
    1. Compute global norm: ‖g‖ = √(Σ_i ‖g_i‖²)
    2. If ‖g‖ > max_norm: scale all gradients by max_norm / ‖g‖
    3. Otherwise: leave unchanged

    This preserves the direction of the gradient while bounding its magnitude.
    """
    total_norm_sq = sum(
        p.grad.data.norm(2).item() ** 2
        for p in parameters
        if p.grad is not None
    )
    total_norm = total_norm_sq ** 0.5

    clip_coef = max_norm / (total_norm + 1e-6)

    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

    return total_norm   # return pre-clip norm for logging


# PyTorch built-in (use this in practice)
model = nn.Linear(100, 10)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# In training loop:
# loss.backward()
# grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# optimizer.step()


def training_step_with_clip(
    model,
    batch,
    optimizer,
    max_grad_norm: float = 1.0,
    log_freq:      int   = 100,
    step:          int   = 0,
) -> dict:
    """
    Standard training step with gradient clipping.

    Healthy gradient norm range: 0.1 – 5.0
    Always clip before optimizer.step() — never after!
    """
    import torch.nn.functional as F

    optimizer.zero_grad()
    x, y = batch
    loss = F.cross_entropy(model(x), y)
    loss.backward()

    # Clip gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

    # Log if suspicious
    if step % log_freq == 0:
        status = "OK" if grad_norm < max_grad_norm else "CLIPPED"
        print(f"Step {step}: grad_norm={grad_norm:.4f} [{status}]")

    optimizer.step()

    return {"loss": loss.item(), "grad_norm": float(grad_norm)}
```

**Why clipping works**: during backprop, if any layer produces a very large activation, the gradient can cascade into a huge update that moves the weights far from the optimum. Clipping limits the update size while preserving gradient direction.

---

## 4. AdamW: The Correct Way to Apply Weight Decay

Vanilla Adam with L2 regularization has a subtle bug: the L2 penalty is scaled by the adaptive learning rate, making effective weight decay different per parameter. AdamW decouples them:

```python
def adamw_update_step(
    params: dict[str, np.ndarray],
    grads:  dict[str, np.ndarray],
    m:      dict[str, np.ndarray],   # first moment (momentum)
    v:      dict[str, np.ndarray],   # second moment (adaptive learning rate)
    t:      int,                     # time step (for bias correction)
    lr:            float = 1e-3,
    beta1:         float = 0.9,
    beta2:         float = 0.999,
    eps:           float = 1e-8,
    weight_decay:  float = 0.01,
) -> None:
    """
    AdamW update (Loshchilov & Hutter, 2019).

    Key difference from Adam + L2:
    - Adam + L2:  g_t ← g_t + λ·w    (weight decay scaled by v)
    - AdamW:      w  ← w - lr·λ·w   (weight decay applied directly)

    AdamW is what you want. Most frameworks implement it correctly.
    """
    t_float = float(t)
    bias_correction1 = 1 - beta1 ** t_float
    bias_correction2 = 1 - beta2 ** t_float

    for key in params.keys():
        # Standard Adam moment updates
        m[key] = beta1 * m[key] + (1 - beta1) * grads[key]
        v[key] = beta2 * v[key] + (1 - beta2) * grads[key] ** 2

        # Bias-corrected estimates
        m_hat = m[key] / bias_correction1
        v_hat = v[key] / bias_correction2

        # AdamW: apply weight decay BEFORE the adaptive update
        params[key] *= (1 - lr * weight_decay)          # L: decoupled decay
        params[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)  # L: adaptive update
```

---

## 5. Mixed Precision Training

Training with BF16 (Brain Float 16) or FP16 gives ~2× speedup and ~2× memory reduction on modern GPUs (A100, H100, RTX 3090+):

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


def mixed_precision_training_loop(
    model:      nn.Module,
    optimizer,
    dataloader,
    n_epochs:   int   = 10,
    grad_clip:  float = 1.0,
    use_bf16:   bool  = True,     # True for A100/H100, False for older GPUs (use fp16)
) -> list[float]:
    """
    Automatic Mixed Precision (AMP) training loop.

    BF16 (Brain Float 16):
    - Same exponent range as FP32
    - Fewer mantissa bits (7 vs 23)
    - Cannot overflow → no loss scaling needed
    - Best for: A100, H100, RTX 4090+

    FP16:
    - Smaller exponent range → can overflow
    - Requires GradScaler to detect and recover from overflows
    - Best for: V100, RTX 3080, T4

    Rule of thumb: use BF16 if available, FP16 otherwise.
    """
    model = model.cuda()
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    # GradScaler only needed for FP16 (BF16 doesn't overflow)
    scaler = GradScaler() if not use_bf16 else None

    losses = []

    for epoch in range(n_epochs):
        epoch_loss = 0.0

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()

            # Forward pass in reduced precision
            with autocast(dtype=dtype):
                output = model(x)
                loss   = nn.CrossEntropyLoss()(output, y)

            # Backward pass
            if scaler is not None:
                # FP16: scale loss to prevent underflow, unscale before clip
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                # BF16: no scaling needed
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")

    return losses
```

**Memory comparison** for a 7B model (approximate):
| Precision | Memory per param | 7B model |
|---|---|---|
| FP32 | 4 bytes | 28 GB |
| FP16 | 2 bytes | 14 GB |
| BF16 | 2 bytes | 14 GB |
| INT8 | 1 byte  | 7 GB |
| NF4  | 0.5 byte | 3.5 GB |

---

## 6. Systematic Training Debugging Protocol

```python
def debug_training_run(
    model:    nn.Module,
    dataset,
    optimizer,
) -> None:
    """
    Karpathy's 6-step training debugging protocol.

    Always start with simpler cases and work up to full complexity.
    """

    print("Step 1: Overfit a single batch")
    # If the model can't memorize 1 batch, there's a fundamental bug
    single_batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=8)))
    for step in range(200):
        loss = training_step(model, single_batch, optimizer)
        if step % 50 == 0:
            print(f"  Step {step}: loss={loss:.6f}")
    # Expected: loss → near 0 (not exactly 0 if using label smoothing)
    # If loss doesn't decrease: check architecture, loss function, gradients

    print("\nStep 2: Check gradient flow")
    # After one backward pass, all parameters should have non-zero gradients
    loss = compute_loss(model, single_batch)
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"  ⚠ No gradient: {name}")
        elif param.grad.abs().max() < 1e-10:
            print(f"  ⚠ Near-zero gradient: {name}: {param.grad.abs().max():.2e}")

    print("\nStep 3: Verify output distribution")
    # At initialization, output should be approximately uniform
    with torch.no_grad():
        logits = model(single_batch[0])
        probs  = torch.softmax(logits, dim=-1)
        print(f"  Initial output entropy: {-(probs * probs.log()).sum(-1).mean():.3f}")
        print(f"  Expected for uniform:   {np.log(logits.shape[-1]):.3f}")
        # If entropy << expected, model is already biased → bad initialization

    print("\nStep 4: Check loss at initialization")
    # For N-class classification, expected initial loss = log(N)
    N = logits.shape[-1]
    print(f"  Expected initial loss: {np.log(N):.4f}")
    with torch.no_grad():
        initial_loss = compute_loss(model, single_batch)
        print(f"  Actual initial loss:   {initial_loss:.4f}")
    # If actual >> expected: initialization too large
    # If actual << expected: model is somehow already predicting correctly

    print("\nStep 5: Train for a few epochs")
    # Loss should decrease monotonically in the first 10-20 steps
    # then may fluctuate due to stochasticity

    print("\nStep 6: Scale up and monitor")
    # Now run the full training loop, monitoring:
    # - train loss: should decrease
    # - val loss: should decrease then plateau (not blow up)
    # - grad norm: should be < max_norm most of the time
    # - LR: should follow schedule


def compute_loss(model, batch):
    x, y = batch
    return nn.CrossEntropyLoss()(model(x), y)


def training_step(model, batch, optimizer):
    optimizer.zero_grad()
    loss = compute_loss(model, batch)
    loss.backward()
    optimizer.step()
    return loss.item()
```

---

## Complete Training Loop

```python
def production_training_loop(
    model:         nn.Module,
    train_loader,
    val_loader,
    total_steps:   int   = 10000,
    warmup_steps:  int   = 500,
    peak_lr:       float = 3e-4,
    weight_decay:  float = 0.1,
    max_grad_norm: float = 1.0,
    eval_every:    int   = 500,
) -> dict:
    """
    Production training loop with all best practices:
    - AdamW optimizer
    - Cosine schedule with warmup
    - Gradient clipping
    - Mixed precision
    - Evaluation + checkpointing
    """
    import copy, torch

    model = model.cuda()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(0.9, 0.95),         # LLM defaults: β₂=0.95 (vs 0.999)
        weight_decay=weight_decay,
        eps=1e-8,
    )

    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler    = GradScaler()   # for FP16

    best_val_loss = float("inf")
    best_state    = None
    history       = {"train_loss": [], "val_loss": [], "grad_norm": [], "lr": []}

    step = 0
    for epoch in range(1000):  # outer loop; inner is controlled by total_steps
        for x, y in train_loader:
            if step >= total_steps:
                break

            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()

            with autocast(dtype=torch.bfloat16):
                loss = nn.CrossEntropyLoss()(model(x), y)

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()

            history["train_loss"].append(loss.item())
            history["grad_norm"].append(float(grad_norm))
            history["lr"].append(optimizer.param_groups[0]["lr"])

            if step % eval_every == 0:
                # Evaluate on validation
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for x_v, y_v in val_loader:
                        vl = nn.CrossEntropyLoss()(model(x_v.cuda()), y_v.cuda())
                        val_losses.append(vl.item())

                val_loss = np.mean(val_losses)
                history["val_loss"].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state    = copy.deepcopy(model.state_dict())

                print(f"Step {step:5d}: train={loss.item():.4f}, val={val_loss:.4f}, "
                      f"lr={optimizer.param_groups[0]['lr']:.6f}, grad_norm={grad_norm:.2f}")
                model.train()

            step += 1

        if step >= total_steps:
            break

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)

    return history
```

---

## Best Practices Checklist

| Category | Practice | Default value |
|---|---|---|
| Initialization | He for ReLU, Xavier for tanh | — |
| Optimizer | AdamW | lr=1e-3, wd=0.01 |
| LR schedule | Cosine with linear warmup | warmup=5% of steps |
| Gradient clipping | Global norm clip | max_norm=1.0 |
| Mixed precision | BF16 on A100+, FP16 on V100 | — |
| Batch size | Largest that fits in memory | — |
| Data | Standardize (mean=0, std=1) | — |
| Debugging | Overfit 1 batch first | — |

---

## Edge Cases & Misconceptions

!!! warning "Misconception: More epochs = better"
    Training beyond the point of minimum validation loss causes overfitting. Use early stopping or train for a fixed compute budget and select the best checkpoint. The final epoch checkpoint is rarely the best.

!!! note "AdamW betas for LLMs"
    For language model training, `β₂=0.95` (not the PyTorch default 0.999) is commonly used. Lower β₂ makes the adaptive learning rate respond faster to gradient changes — important when gradient magnitudes shift significantly across training.

!!! warning "Misconception: BF16 is always safe"
    BF16 has 7 mantissa bits vs FP32's 23. Operations requiring high precision (e.g., loss computation, final softmax) should be done in FP32. PyTorch's `autocast` handles this automatically by keeping certain operations in FP32.

---

## Production Connection

**LLaMA training setup**: Meta trained LLaMA-3 with AdamW (β₁=0.9, β₂=0.95, wd=0.1), cosine schedule with 2000 warmup steps, gradient clipping at 1.0, and 4M token batch size. The training loss curve had several "spikes" where they rolled back to earlier checkpoints and continued.

**Gradient accumulation**: when the GPU can't fit the desired batch size, use gradient accumulation — run N mini-batches, accumulate gradients, then step. Clip gradients *after* accumulation, not per-mini-batch.

---

## Key Takeaways

1. **He initialization** (`std = √(2/n_in)`) keeps activation variance stable for ReLU networks; **Xavier** (`std = √(2/(n_in+n_out))`) for tanh/sigmoid.
2. **Warmup + cosine decay** is the standard LR schedule for Transformers: ramp up linearly for 1–5% of steps, then decay as cosine to ~10% of peak LR.
3. **Gradient clipping** rescales the full gradient vector when its global norm exceeds `max_norm`, preserving direction while bounding step size.
4. **AdamW** applies weight decay directly to parameters (`w ← w·(1-lr·λ)`) rather than adding it to the gradient — this is mathematically correct for L2 regularization with adaptive optimizers.
5. **Mixed precision** (BF16/FP16) halves memory and doubles throughput on modern GPUs with negligible quality loss.
6. **Debug systematically**: overfit one batch first, check gradient flow, verify initial loss equals log(N).

---

## Further Reading

- [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/) — Andrej Karpathy's practical guide
- [AdamW paper](https://arxiv.org/abs/1711.05101) — Loshchilov & Hutter 2019: Decoupled Weight Decay Regularization
- [Mixed precision training](https://arxiv.org/abs/1710.03740) — Micikevicius et al. 2018
- [PyTorch AMP tutorial](https://pytorch.org/docs/stable/amp.html) — official automatic mixed precision documentation
- [nanoGPT](https://github.com/karpathy/nanoGPT) — complete GPT training in ~300 lines

---

## Module 05 Complete

You have now mastered the neural networks fundamentals:

- Neuron architecture, activation functions, loss functions
- Gradient descent, backpropagation, optimization
- Overfitting, regularization, dropout
- Building networks from scratch in NumPy
- Convolutional, recurrent, and feed-forward architectures
- Professional training practices

**[Module 06: Transformers & Attention →](../../module-06-transformers-attention-mechanisms/index.md)**
