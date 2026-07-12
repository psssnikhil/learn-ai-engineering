import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def self_attention(X, W_Q, W_K, W_V):
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores)
    return weights @ V, weights

def masked_self_attention(X, W_Q, W_K, W_V):
    Q, K, V = X @ W_Q, X @ W_K, X @ W_V
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    mask = np.triu(np.ones_like(scores) * -1e9, k=1)
    scores += mask
    weights = softmax(scores)
    return weights @ V, weights

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
