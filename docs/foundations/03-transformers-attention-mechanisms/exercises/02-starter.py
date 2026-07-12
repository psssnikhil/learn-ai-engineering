import numpy as np

# Self-Attention - Exercise

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def self_attention(X, W_Q, W_K, W_V):
    """
    TODO: Implement self-attention.
    Q = X @ W_Q, K = X @ W_K, V = X @ W_V
    scores = Q @ K.T / sqrt(d_k)
    weights = softmax(scores)
    output = weights @ V
    Return output and weights.
    """
    pass

def masked_self_attention(X, W_Q, W_K, W_V):
    """
    TODO: Implement masked (causal) self-attention.
    Same as self_attention but apply a causal mask:
    set scores[i][j] = -inf where j > i (can't attend to future).
    """
    pass

if __name__ == "__main__":
    np.random.seed(42)
    seq_len, d_model, d_k = 4, 8, 4
    X = np.random.randn(seq_len, d_model)
    W_Q = np.random.randn(d_model, d_k) * 0.1
    W_K = np.random.randn(d_model, d_k) * 0.1
    W_V = np.random.randn(d_model, d_k) * 0.1

    out, weights = self_attention(X, W_Q, W_K, W_V)
    print(f"Self-attention output shape: {out.shape}")
    print(f"Weights (rows sum to 1): {weights.sum(axis=1).round(4)}")
    print()

    out_m, weights_m = masked_self_attention(X, W_Q, W_K, W_V)
    print(f"Masked weights:\n{weights_m.round(3)}")
    print("(lower triangular - no future attention)")
