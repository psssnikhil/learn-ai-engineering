import numpy as np

# Backpropagation - Solution

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

class TwoLayerNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros(output_size)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, learning_rate=0.1):
        n = len(X)
        delta2 = (self.a2 - y) * sigmoid_derivative(self.z2)
        delta1 = (delta2 @ self.W2.T) * sigmoid_derivative(self.z1)

        dW2 = self.a1.T @ delta2 / n
        db2 = np.mean(delta2, axis=0)
        dW1 = X.T @ delta1 / n
        db1 = np.mean(delta1, axis=0)

        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

        return np.mean((self.a2 - y) ** 2)

    def train(self, X, y, epochs=1000, learning_rate=0.1):
        losses = []
        for _ in range(epochs):
            self.forward(X)
            loss = self.backward(X, y, learning_rate)
            losses.append(loss)
        return losses

if __name__ == "__main__":
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
