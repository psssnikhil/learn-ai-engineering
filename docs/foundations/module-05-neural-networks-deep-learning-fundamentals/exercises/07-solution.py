import numpy as np

# Building a Neural Network from Scratch - Solution

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i+1]
            w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / (fan_in + fan_out))
            b = np.zeros(fan_out)
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        current = X
        for w, b in zip(self.weights, self.biases):
            z = current @ w + b
            self.z_values.append(z)
            current = sigmoid(z)
            self.activations.append(current)
        return current

    def backward(self, X, y, learning_rate=0.1):
        n = len(X)
        num_layers = len(self.weights)

        # Output layer delta
        delta = (self.activations[-1] - y) * self.activations[-1] * (1 - self.activations[-1])

        for i in range(num_layers - 1, -1, -1):
            dw = self.activations[i].T @ delta / n
            db = np.mean(delta, axis=0)
            if i > 0:
                delta = (delta @ self.weights[i].T) * self.activations[i] * (1 - self.activations[i])
            self.weights[i] -= learning_rate * dw
            self.biases[i] -= learning_rate * db

        return np.mean((self.activations[-1] - y) ** 2)

    def train(self, X, y, epochs=1000, learning_rate=0.1, verbose=True):
        losses = []
        for epoch in range(epochs):
            self.forward(X)
            loss = self.backward(X, y, learning_rate)
            losses.append(loss)
            if verbose and (epoch + 1) % 1000 == 0:
                print(f"Epoch {epoch+1}: loss = {loss:.6f}")
        return losses

    def predict(self, X):
        return np.round(self.forward(X))

if __name__ == "__main__":
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])

    np.random.seed(42)
    nn = NeuralNetwork([2, 8, 4, 1])
    nn.train(X, y, epochs=5000, learning_rate=2.0)

    print("\nPredictions:")
    for i in range(len(X)):
        pred = nn.predict(X[i:i+1])
        print(f"  {X[i]} -> {int(pred[0][0])} (expected {y[i][0]})")
