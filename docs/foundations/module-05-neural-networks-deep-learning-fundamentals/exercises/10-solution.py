import numpy as np

# Training Best Practices - Solution

def batch_normalize(X, epsilon=1e-5):
    mean = X.mean(axis=0)
    var = X.var(axis=0)
    X_norm = (X - mean) / np.sqrt(var + epsilon)
    return X_norm, mean, var

def create_mini_batches(X, y, batch_size=32):
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    batches = []
    for i in range(0, len(X), batch_size):
        batches.append((X_shuffled[i:i+batch_size], y_shuffled[i:i+batch_size]))
    return batches

def learning_rate_schedule(initial_lr, epoch, schedule_type="step"):
    if schedule_type == "step":
        return initial_lr * (0.5 ** (epoch // 10))
    elif schedule_type == "exponential":
        return initial_lr * (0.95 ** epoch)
    elif schedule_type == "cosine":
        return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / 100))
    return initial_lr

def early_stopping_check(val_losses, patience=5):
    if len(val_losses) < patience + 1:
        return False
    best = min(val_losses[:-patience])
    recent_best = min(val_losses[-patience:])
    return recent_best >= best

if __name__ == "__main__":
    np.random.seed(42)

    X = np.random.randn(100, 4) * 3 + 5
    X_norm, mean, var = batch_normalize(X)
    print(f"Before: mean={X.mean(axis=0).round(2)}, std={X.std(axis=0).round(2)}")
    print(f"After:  mean={X_norm.mean(axis=0).round(4)}, std={X_norm.std(axis=0).round(4)}")
    print()

    X = np.random.randn(100, 4)
    y = np.random.randint(0, 2, 100)
    batches = create_mini_batches(X, y, batch_size=32)
    print(f"Number of batches: {len(batches)}")
    print(f"Batch sizes: {[len(b[0]) for b in batches]}")
    print()

    for epoch in [0, 5, 10, 20, 50]:
        print(f"Epoch {epoch}: step={learning_rate_schedule(0.01, epoch, 'step'):.6f}, "
              f"exp={learning_rate_schedule(0.01, epoch, 'exponential'):.6f}, "
              f"cos={learning_rate_schedule(0.01, epoch, 'cosine'):.6f}")
    print()

    val_losses = [1.0, 0.9, 0.8, 0.85, 0.86, 0.87, 0.88, 0.89]
    print(f"Should stop: {early_stopping_check(val_losses, patience=5)}")
