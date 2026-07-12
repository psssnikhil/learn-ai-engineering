import numpy as np

# Gradient Descent - Solution

def numerical_gradient(f, x, epsilon=1e-5):
    return (f(x + epsilon) - f(x - epsilon)) / (2 * epsilon)

def gradient_descent_1d(f, df, x_init, lr=0.1, steps=50):
    x = x_init
    history = [(x, f(x))]
    for _ in range(steps):
        x = x - lr * df(x)
        history.append((x, f(x)))
    return x, history

def gradient_descent_linear_regression(X, y, lr=0.01, steps=100):
    n = len(y)
    w = np.zeros(X.shape[1])
    b = 0.0
    loss_history = []
    for _ in range(steps):
        pred = X @ w + b
        error = pred - y
        dw = (2/n) * X.T @ error
        db = (2/n) * np.sum(error)
        w -= lr * dw
        b -= lr * db
        loss_history.append(np.mean(error**2))
    return w, b, loss_history

if __name__ == "__main__":
    f = lambda x: x**2
    df = lambda x: 2*x
    x_final, history = gradient_descent_1d(f, df, x_init=5.0, lr=0.1, steps=20)
    print(f"Minimum found at x={x_final:.4f} (expected 0.0)")
    print(f"f(x)={f(x_final):.6f}")
    print()

    print(f"Numerical gradient at x=3: {numerical_gradient(f, 3.0):.4f} (expected 6.0)")
    print()

    np.random.seed(42)
    X = np.random.randn(100, 2)
    true_w = np.array([3.0, -2.0])
    y = X @ true_w + 1.0 + np.random.randn(100) * 0.1
    w, b, losses = gradient_descent_linear_regression(X, y, lr=0.01, steps=200)
    print(f"Learned w: {w} (expected [3, -2])")
    print(f"Learned b: {b:.4f} (expected 1.0)")
