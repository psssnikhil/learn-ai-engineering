import numpy as np

# Training Best Practices - Exercise

def batch_normalize(X, epsilon=1e-5):
    """
    TODO: Implement batch normalization.
    For each feature: x_norm = (x - mean) / sqrt(variance + epsilon)
    Return normalized X, mean, and variance.
    """
    pass

def create_mini_batches(X, y, batch_size=32):
    """
    TODO: Split data into mini-batches.
    Shuffle data first, then split into batches of batch_size.
    Last batch may be smaller.
    Return list of (X_batch, y_batch) tuples.
    """
    pass

def learning_rate_schedule(initial_lr, epoch, schedule_type="step"):
    """
    TODO: Implement learning rate schedules.
    - "step": halve LR every 10 epochs
    - "exponential": lr * 0.95^epoch
    - "cosine": lr * 0.5 * (1 + cos(pi * epoch / 100))
    """
    pass

def early_stopping_check(val_losses, patience=5):
    """
    TODO: Check if training should stop.
    Return True if val_loss hasn't improved for `patience` epochs.
    """
    pass

if __name__ == "__main__":
    np.random.seed(42)

    # Batch normalization
    X = np.random.randn(100, 4) * 3 + 5
    X_norm, mean, var = batch_normalize(X)
    print(f"Before: mean={X.mean(axis=0).round(2)}, std={X.std(axis=0).round(2)}")
    print(f"After:  mean={X_norm.mean(axis=0).round(4)}, std={X_norm.std(axis=0).round(4)}")
    print()

    # Mini-batches
    X = np.random.randn(100, 4)
    y = np.random.randint(0, 2, 100)
    batches = create_mini_batches(X, y, batch_size=32)
    print(f"Number of batches: {len(batches)}")
    print(f"Batch sizes: {[len(b[0]) for b in batches]}")
    print()

    # LR schedule
    for epoch in [0, 5, 10, 20, 50]:
        print(f"Epoch {epoch}: step={learning_rate_schedule(0.01, epoch, 'step'):.6f}, "
              f"exp={learning_rate_schedule(0.01, epoch, 'exponential'):.6f}, "
              f"cos={learning_rate_schedule(0.01, epoch, 'cosine'):.6f}")
    print()

    # Early stopping
    val_losses = [1.0, 0.9, 0.8, 0.85, 0.86, 0.87, 0.88, 0.89]
    print(f"Should stop: {early_stopping_check(val_losses, patience=5)}")
