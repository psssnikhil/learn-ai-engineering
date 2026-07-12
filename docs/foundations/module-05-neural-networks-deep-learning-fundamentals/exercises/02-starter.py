import numpy as np

# Neurons & Activation Functions - Exercise

def sigmoid(x):
    """TODO: Implement sigmoid. Formula: 1 / (1 + exp(-x))"""
    pass

def sigmoid_derivative(x):
    """TODO: Implement sigmoid derivative. Formula: sigmoid(x) * (1 - sigmoid(x))"""
    pass

def relu(x):
    """TODO: Implement ReLU. Formula: max(0, x)"""
    pass

def relu_derivative(x):
    """TODO: Implement ReLU derivative. Returns 1 where x > 0, else 0."""
    pass

def tanh(x):
    """TODO: Implement tanh. Formula: (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""
    pass

def neuron_forward(inputs, weights, bias, activation_fn):
    """
    TODO: Implement forward pass of a single neuron.
    1. Compute weighted sum: z = dot(inputs, weights) + bias
    2. Apply activation function: output = activation_fn(z)
    Return both z and output.
    """
    pass

# Test
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
