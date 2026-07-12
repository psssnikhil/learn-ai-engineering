import numpy as np

# Training Transformers - Exercise

def create_training_data(vocab_size=10, seq_len=5, num_samples=100):
    """
    TODO: Create synthetic training data for a sequence task.
    Task: reverse the input sequence.
    Return input sequences and target sequences (reversed).
    """
    pass

def one_hot_encode(sequences, vocab_size):
    """TODO: Convert integer sequences to one-hot encoding."""
    pass

def compute_cross_entropy(predictions, targets):
    """
    TODO: Cross-entropy loss for sequence predictions.
    predictions: (batch, seq_len, vocab_size) - logits
    targets: (batch, seq_len) - integer labels
    """
    pass

def learning_rate_warmup(step, d_model, warmup_steps=4000):
    """
    TODO: Transformer learning rate schedule with warmup.
    lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """
    pass

if __name__ == "__main__":
    np.random.seed(42)
    X, Y = create_training_data(vocab_size=10, seq_len=5, num_samples=5)
    print(f"Input: {X[0]}")
    print(f"Target (reversed): {Y[0]}")
    print()

    encoded = one_hot_encode(X[:2], vocab_size=10)
    print(f"One-hot shape: {encoded.shape}")
    print()

    for step in [1, 100, 1000, 4000, 10000]:
        lr = learning_rate_warmup(step, d_model=512)
        print(f"Step {step:>5}: lr = {lr:.8f}")
