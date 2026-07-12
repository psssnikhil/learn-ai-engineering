import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

class AttentionLayer:
    def __init__(self, d_model, d_k, d_v):
        self.W_Q = np.random.randn(d_model, d_k) * np.sqrt(2.0 / (d_model + d_k))
        self.W_K = np.random.randn(d_model, d_k) * np.sqrt(2.0 / (d_model + d_k))
        self.W_V = np.random.randn(d_model, d_v) * np.sqrt(2.0 / (d_model + d_v))
        self.W_O = np.random.randn(d_v, d_model) * np.sqrt(2.0 / (d_v + d_model))

    def forward(self, X, mask=None):
        Q, K, V = X @ self.W_Q, X @ self.W_K, X @ self.W_V
        d_k = Q.shape[-1]
        scores = Q @ K.T / np.sqrt(d_k)
        if mask is not None:
            scores += mask
        weights = softmax(scores)
        attn_out = weights @ V
        return attn_out @ self.W_O, weights

class MultiHeadAttentionLayer:
    def __init__(self, d_model, num_heads):
        d_k = d_model // num_heads
        self.heads = [AttentionLayer(d_model, d_k, d_k) for _ in range(num_heads)]
        self.W_O = np.random.randn(d_model, d_model) * 0.1

    def forward(self, X, mask=None):
        outputs = []
        all_weights = []
        for head in self.heads:
            out, w = head.forward(X, mask)
            outputs.append(out)
            all_weights.append(w)
        # Note: each head already projects to d_model, so we average instead
        combined = np.mean(outputs, axis=0)
        return combined, all_weights

def visualize_attention(words, weights):
    n = len(words)
    header = "      " + "".join(f"{w:>8}" for w in words)
    print(header)
    for i in range(n):
        row = f"{words[i]:>6}"
        for j in range(n):
            row += f"{weights[i][j]:8.3f}"
        print(row)

if __name__ == "__main__":
    np.random.seed(42)

    attn = AttentionLayer(d_model=8, d_k=4, d_v=4)
    X = np.random.randn(5, 8)
    out, weights = attn.forward(X)
    print(f"Single head output: {out.shape}")
    print()

    mha = MultiHeadAttentionLayer(d_model=16, num_heads=4)
    X = np.random.randn(4, 16)
    out, weights = mha.forward(X)
    print(f"Multi-head output: {out.shape}")
    print()

    words = ["The", "cat", "sat", "on"]
    visualize_attention(words, weights[0])
