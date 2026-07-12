import numpy as np

# Introduction to Attention - Exercise

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def basic_attention(query, keys, values):
    """
    TODO: Implement basic dot-product attention.
    1. scores = query @ keys.T
    2. weights = softmax(scores)
    3. output = weights @ values
    Return output and weights.
    """
    pass

def additive_attention(query, keys, values, W1, W2, v):
    """
    TODO: Implement Bahdanau (additive) attention.
    score_i = v @ tanh(W1 @ query + W2 @ key_i)
    """
    pass

if __name__ == "__main__":
    np.random.seed(42)
    d = 4
    query = np.random.randn(d)
    keys = np.random.randn(5, d)
    values = np.random.randn(5, d)

    output, weights = basic_attention(query, keys, values)
    print(f"Attention weights: {weights.round(4)}")
    print(f"Output: {output.round(4)}")
