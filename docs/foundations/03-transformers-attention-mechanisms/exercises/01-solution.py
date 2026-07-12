import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def basic_attention(query, keys, values):
    scores = keys @ query
    weights = softmax(scores)
    output = weights @ values
    return output, weights

def additive_attention(query, keys, values, W1, W2, v):
    scores = []
    for key in keys:
        score = v @ np.tanh(W1 @ query + W2 @ key)
        scores.append(score)
    weights = softmax(np.array(scores))
    output = weights @ values
    return output, weights

if __name__ == "__main__":
    np.random.seed(42)
    d = 4
    query = np.random.randn(d)
    keys = np.random.randn(5, d)
    values = np.random.randn(5, d)

    output, weights = basic_attention(query, keys, values)
    print(f"Attention weights: {weights.round(4)}")
    print(f"Output: {output.round(4)}")
