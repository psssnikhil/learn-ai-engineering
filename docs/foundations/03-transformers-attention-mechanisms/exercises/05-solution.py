import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def layer_norm(x, epsilon=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + epsilon)

def feed_forward(x, d_model, d_ff):
    W1 = np.random.randn(d_model, d_ff) * 0.1
    b1 = np.zeros(d_ff)
    W2 = np.random.randn(d_ff, d_model) * 0.1
    b2 = np.zeros(d_model)
    return np.maximum(0, x @ W1 + b1) @ W2 + b2

def multi_head_attn(X, num_heads, d_model):
    d_k = d_model // num_heads
    heads = []
    for _ in range(num_heads):
        Q = X @ (np.random.randn(d_model, d_k) * 0.1)
        K = X @ (np.random.randn(d_model, d_k) * 0.1)
        V = X @ (np.random.randn(d_model, d_k) * 0.1)
        scores = Q @ K.T / np.sqrt(d_k)
        weights = softmax(scores)
        heads.append(weights @ V)
    concat = np.concatenate(heads, axis=-1)
    return concat @ (np.random.randn(d_model, d_model) * 0.1)

def transformer_block(X, num_heads, d_model, d_ff):
    attn_out = multi_head_attn(X, num_heads, d_model)
    X = layer_norm(X + attn_out)
    ff_out = feed_forward(X, d_model, d_ff)
    X = layer_norm(X + ff_out)
    return X

if __name__ == "__main__":
    np.random.seed(42)
    seq_len, d_model = 4, 16
    X = np.random.randn(seq_len, d_model)

    normed = layer_norm(X)
    print(f"Layer norm mean: {normed.mean(axis=-1).round(6)}")
    print(f"Layer norm std: {normed.std(axis=-1).round(4)}")
    print()

    output = transformer_block(X, num_heads=4, d_model=16, d_ff=64)
    print(f"Transformer block output shape: {output.shape}")
    print(f"Output differs from input: {not np.allclose(X, output)}")
