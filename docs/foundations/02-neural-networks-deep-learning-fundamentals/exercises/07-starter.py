import numpy as np

# Building a Neural Network from Scratch - Exercise

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class NeuralNetwork:
    """A configurable neural network built from scratch."""

    def __init__(self, layer_sizes):
        """
        TODO: Initialize network with given layer sizes.
        E.g., [2, 4, 3, 1] means 2 inputs, hidden layers of 4 and 3, 1 output.
        Initialize weights with Xavier initialization: randn * sqrt(2 / (fan_in + fan_out))
        Store as self.weights (list of matrices) and self.biases (list of vectors).
        """
        pass

    def forward(self, X):
        """
        TODO: Forward pass through all layers.
        Store activations for backprop in self.activations.
        Use sigmoid activation for all layers.
        Return final output.
        """
        pass

    def backward(self, X, y, learning_rate=0.1):
        """
        TODO: Backpropagation through all layers.
        1. Compute output layer delta
        2. Propagate deltas backward through each layer
        3. Update all weights and biases
        Return loss (MSE).
        """
        pass

    def train(self, X, y, epochs=1000, learning_rate=0.1, verbose=True):
        """TODO: Training loop. Print loss every 200 epochs if verbose."""
        pass

    def predict(self, X):
        """TODO: Forward pass and round to 0/1."""
        pass

if __name__ == "__main__":
    # Test on XOR
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])

    np.random.seed(42)
    nn = NeuralNetwork([2, 8, 4, 1])
    nn.train(X, y, epochs=5000, learning_rate=2.0)

    print("\nPredictions:")
    for i in range(len(X)):
        pred = nn.predict(X[i:i+1])
        print(f"  {X[i]} -> {pred[0][0]} (expected {y[i][0]})")
