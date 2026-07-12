import numpy as np

def create_training_data(vocab_size=10, seq_len=5, num_samples=100):
    X = np.random.randint(0, vocab_size, (num_samples, seq_len))
    Y = X[:, ::-1]
    return X, Y

def one_hot_encode(sequences, vocab_size):
    batch, seq_len = sequences.shape
    encoded = np.zeros((batch, seq_len, vocab_size))
    for i in range(batch):
        for j in range(seq_len):
            encoded[i, j, sequences[i, j]] = 1
    return encoded

def compute_cross_entropy(predictions, targets):
    batch, seq_len, vocab_size = predictions.shape
    exp_pred = np.exp(predictions - predictions.max(axis=-1, keepdims=True))
    probs = exp_pred / exp_pred.sum(axis=-1, keepdims=True)
    loss = 0
    for i in range(batch):
        for j in range(seq_len):
            loss -= np.log(probs[i, j, targets[i, j]] + 1e-10)
    return loss / (batch * seq_len)

def learning_rate_warmup(step, d_model, warmup_steps=4000):
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))

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
