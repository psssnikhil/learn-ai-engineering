import numpy as np

# Implementing Attention from Scratch - Exercise

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class AttentionLayer:
    """Complete attention layer implementation."""

    def __init__(self, d_model, d_k, d_v):
        """TODO: Initialize W_Q, W_K, W_V, W_O weight matrices."""
        pass

    def forward(self, X, mask=None):
        """
        TODO: Full attention forward pass.
        1. Project to Q, K, V
        2. Compute scaled dot-product attention (with optional mask)
        3. Project output
        Return output and attention weights.
        """
        pass


class MultiHeadAttentionLayer:
    """Multi-head attention with proper implementation."""

    def __init__(self, d_model, num_heads):
        """TODO: Create num_heads AttentionLayer instances."""
        pass

    def forward(self, X, mask=None):
        """TODO: Run all heads, concatenate, project."""
        pass


def visualize_attention(words, weights):
    """
    TODO: Print attention weights as a formatted table.
    Show which words attend to which other words.
    """
    pass


if __name__ == "__main__":
    np.random.seed(42)

    # Single head
    attn = AttentionLayer(d_model=8, d_k=4, d_v=4)
    X = np.random.randn(5, 8)
    out, weights = attn.forward(X)
    print(f"Single head output: {out.shape}")
    print()

    # Multi-head
    mha = MultiHeadAttentionLayer(d_model=16, num_heads=4)
    X = np.random.randn(4, 16)
    out, weights = mha.forward(X)
    print(f"Multi-head output: {out.shape}")
    print()

    # Visualize
    words = ["The", "cat", "sat", "on"]
    visualize_attention(words, weights[0])  # First head
