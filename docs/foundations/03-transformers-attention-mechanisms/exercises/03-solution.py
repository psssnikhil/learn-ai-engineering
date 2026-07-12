import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def single_head_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores)
    return weights @ V, weights

def multi_head_attention(X, num_heads, d_model):
    d_k = d_model // num_heads
    head_outputs = []
    all_weights = []

    for _ in range(num_heads):
        W_Q = np.random.randn(d_model, d_k) * 0.1
        W_K = np.random.randn(d_model, d_k) * 0.1
        W_V = np.random.randn(d_model, d_k) * 0.1
        Q, K, V = X @ W_Q, X @ W_K, X @ W_V
        out, weights = single_head_attention(Q, K, V)
        head_outputs.append(out)
        all_weights.append(weights)

    concat = np.concatenate(head_outputs, axis=-1)
    W_O = np.random.randn(d_model, d_model) * 0.1
    return concat @ W_O, all_weights

if __name__ == "__main__":
    np.random.seed(42)
    seq_len, d_model, num_heads = 4, 16, 4
    X = np.random.randn(seq_len, d_model)

    output, all_weights = multi_head_attention(X, num_heads, d_model)
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of heads: {len(all_weights)}")
    for i, w in enumerate(all_weights):
        print(f"Head {i} attention pattern (row 0): {w[0].round(3)}")
