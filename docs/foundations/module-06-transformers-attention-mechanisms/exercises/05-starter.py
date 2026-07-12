import numpy as np

# Complete Transformer Architecture - Exercise

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def layer_norm(x, epsilon=1e-5):
    """TODO: Layer normalization. Normalize across last dimension."""
    pass

def feed_forward(x, d_model, d_ff):
    """TODO: FFN(x) = ReLU(x @ W1 + b1) @ W2 + b2"""
    pass

def transformer_block(X, num_heads, d_model, d_ff):
    """
    TODO: One transformer encoder block.
    1. Multi-head self-attention (simplified: use random projections)
    2. Add & LayerNorm (residual connection)
    3. Feed-forward network
    4. Add & LayerNorm (residual connection)
    Return output (same shape as X).
    """
    pass

if __name__ == "__main__":
    np.random.seed(42)
    seq_len, d_model = 4, 16
    X = np.random.randn(seq_len, d_model)

    # Layer norm test
    normed = layer_norm(X)
    print(f"Layer norm mean: {normed.mean(axis=-1).round(6)}")  # Should be ~0
    print(f"Layer norm std: {normed.std(axis=-1).round(4)}")    # Should be ~1
    print()

    # Transformer block
    output = transformer_block(X, num_heads=4, d_model=16, d_ff=64)
    print(f"Transformer block output shape: {output.shape}")
    print(f"Output differs from input: {not np.allclose(X, output)}")
