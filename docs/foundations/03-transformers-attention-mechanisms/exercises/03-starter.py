import numpy as np

# Multi-Head Attention - Exercise

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def single_head_attention(Q, K, V):
    """Scaled dot-product attention for one head."""
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores)
    return weights @ V, weights

def multi_head_attention(X, num_heads, d_model):
    """
    TODO: Implement multi-head attention.
    1. d_k = d_model // num_heads
    2. For each head: project X -> Q, K, V using random weight matrices (d_model, d_k)
    3. Compute attention for each head
    4. Concatenate all head outputs
    5. Apply final projection (d_model, d_model)
    Return output (seq_len, d_model) and list of attention weights per head.
    """
    pass

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
