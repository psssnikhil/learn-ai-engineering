import numpy as np

# Backpropagation - Exercise

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

class TwoLayerNetwork:
    """A 2-layer network to demonstrate backpropagation."""

    def __init__(self, input_size, hidden_size, output_size):
        """Initialize with random weights."""
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros(output_size)

    def forward(self, X):
        """
        TODO: Forward pass. Store intermediate values for backprop.
        self.z1 = X @ W1 + b1
        self.a1 = sigmoid(z1)
        self.z2 = a1 @ W2 + b2
        self.a2 = sigmoid(z2)
        Return self.a2
        """
        pass

    def backward(self, X, y, learning_rate=0.1):
        """
        TODO: Backpropagation.
        1. Compute output error: delta2 = (self.a2 - y) * sigmoid_derivative(self.z2)
        2. Compute hidden error: delta1 = (delta2 @ W2.T) * sigmoid_derivative(self.z1)
        3. Compute gradients:
           dW2 = a1.T @ delta2 / n
           db2 = mean(delta2, axis=0)
           dW1 = X.T @ delta1 / n
           db1 = mean(delta1, axis=0)
        4. Update weights: W -= learning_rate * dW
        Return the loss (MSE).
        """
        pass

    def train(self, X, y, epochs=1000, learning_rate=0.1):
        """TODO: Train for given epochs. Return list of losses."""
        pass

if __name__ == "__main__":
    # XOR problem
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])

    np.random.seed(42)
    net = TwoLayerNetwork(2, 4, 1)
    losses = net.train(X, y, epochs=5000, learning_rate=1.0)

    print("=== XOR Problem ===")
    predictions = net.forward(X)
    for i in range(len(X)):
        print(f"Input: {X[i]} -> Predicted: {predictions[i][0]:.4f}, Expected: {y[i][0]}")
    print(f"\nFinal loss: {losses[-1]:.6f}")
