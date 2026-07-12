import numpy as np

# Neurons & Activation Functions - Solution

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def neuron_forward(inputs, weights, bias, activation_fn):
    z = np.dot(inputs, weights) + bias
    output = activation_fn(z)
    return z, output

if __name__ == "__main__":
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    print("=== Activation Functions ===")
    print(f"Sigmoid: {sigmoid(x)}")
    print(f"ReLU: {relu(x)}")
    print(f"Tanh: {tanh(x)}")
    print()

    inputs = np.array([0.5, 0.3, 0.2])
    weights = np.array([0.4, 0.6, 0.8])
    bias = 0.1
    z, output = neuron_forward(inputs, weights, bias, sigmoid)
    print(f"Neuron: z={z:.4f}, output={output:.4f}")
