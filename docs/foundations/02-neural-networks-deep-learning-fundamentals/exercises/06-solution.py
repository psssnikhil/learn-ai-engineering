import numpy as np

# Overfitting, Regularization & Dropout - Solution

def l2_regularization(weights, lambda_reg=0.01):
    return lambda_reg * np.sum(weights ** 2)

def l1_regularization(weights, lambda_reg=0.01):
    return lambda_reg * np.sum(np.abs(weights))

def dropout_forward(X, dropout_rate=0.5, training=True):
    if not training:
        return X, np.ones_like(X)
    mask = (np.random.rand(*X.shape) > dropout_rate).astype(float)
    output = X * mask / (1 - dropout_rate)
    return output, mask

def demonstrate_overfitting():
    np.random.seed(42)
    x_train = np.linspace(0, 1, 20)
    y_train = np.sin(2 * np.pi * x_train) + np.random.randn(20) * 0.3
    x_test = np.linspace(0, 1, 100)
    y_test = np.sin(2 * np.pi * x_test)

    degree = 15
    X_train = np.vander(x_train, degree + 1, increasing=True)
    X_test = np.vander(x_test, degree + 1, increasing=True)

    # No regularization
    w_no_reg, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)
    train_mse = np.mean((X_train @ w_no_reg - y_train) ** 2)
    test_mse = np.mean((X_test @ w_no_reg - y_test) ** 2)
    print(f"No regularization - Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")

    # L2 regularization (Ridge)
    lambda_reg = 0.001
    I = np.eye(degree + 1)
    w_ridge = np.linalg.solve(X_train.T @ X_train + lambda_reg * I, X_train.T @ y_train)
    train_mse_reg = np.mean((X_train @ w_ridge - y_train) ** 2)
    test_mse_reg = np.mean((X_test @ w_ridge - y_test) ** 2)
    print(f"L2 regularization - Train MSE: {train_mse_reg:.4f}, Test MSE: {test_mse_reg:.4f}")
    print(f"Regularization reduced test error by {(test_mse - test_mse_reg)/test_mse*100:.1f}%")

if __name__ == "__main__":
    w = np.array([0.5, -1.2, 0.8, 0.3])
    print(f"L2 reg: {l2_regularization(w, 0.01):.6f}")
    print(f"L1 reg: {l1_regularization(w, 0.01):.6f}")
    print()

    np.random.seed(42)
    X = np.random.randn(5, 4)
    out, mask = dropout_forward(X, dropout_rate=0.5)
    print(f"Dropout mask:\n{mask}")
    print(f"Active neurons: {int(mask.sum())}/{mask.size}")
    print()

    demonstrate_overfitting()
