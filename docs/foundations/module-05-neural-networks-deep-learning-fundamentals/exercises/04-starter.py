import numpy as np

# Gradient Descent - The Learning Algorithm - Exercise

def numerical_gradient(f, x, epsilon=1e-5):
    """
    TODO: Compute numerical gradient using finite differences.
    Formula: (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)
    """
    pass

def gradient_descent_1d(f, df, x_init, lr=0.1, steps=50):
    """
    TODO: 1D gradient descent.
    1. Start at x_init
    2. For each step: x = x - lr * df(x)
    3. Track x and f(x) history
    Return final x and history of (x, f(x)) pairs.
    """
    pass

def gradient_descent_linear_regression(X, y, lr=0.01, steps=100):
    """
    TODO: Gradient descent for linear regression y = Xw + b.
    1. Initialize w = zeros, b = 0
    2. For each step compute gradients and update
    Return w, b, loss_history.
    """
    pass

if __name__ == "__main__":
    # Test on f(x) = x^2, minimum at x=0
    f = lambda x: x**2
    df = lambda x: 2*x
    x_final, history = gradient_descent_1d(f, df, x_init=5.0, lr=0.1, steps=20)
    print(f"Minimum found at x={x_final:.4f} (expected 0.0)")
    print(f"f(x)={f(x_final):.6f}")
    print()

    # Test numerical gradient
    print(f"Numerical gradient at x=3: {numerical_gradient(f, 3.0):.4f} (expected 6.0)")
    print()

    # Linear regression
    np.random.seed(42)
    X = np.random.randn(100, 2)
    true_w = np.array([3.0, -2.0])
    y = X @ true_w + 1.0 + np.random.randn(100) * 0.1
    w, b, losses = gradient_descent_linear_regression(X, y, lr=0.01, steps=200)
    print(f"Learned w: {w} (expected [3, -2])")
    print(f"Learned b: {b:.4f} (expected 1.0)")
