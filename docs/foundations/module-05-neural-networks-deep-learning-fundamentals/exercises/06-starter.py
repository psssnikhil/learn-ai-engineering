import numpy as np

# Overfitting, Regularization & Dropout - Exercise

def l2_regularization(weights, lambda_reg=0.01):
    """TODO: Compute L2 regularization term. Formula: lambda * sum(w^2)"""
    pass

def l1_regularization(weights, lambda_reg=0.01):
    """TODO: Compute L1 regularization term. Formula: lambda * sum(|w|)"""
    pass

def dropout_forward(X, dropout_rate=0.5, training=True):
    """
    TODO: Implement dropout.
    During training: randomly zero out neurons with probability dropout_rate,
    scale remaining by 1/(1-dropout_rate).
    During inference: return X unchanged.
    Return output and mask.
    """
    pass

def demonstrate_overfitting():
    """
    TODO: Show overfitting vs regularization.
    1. Generate noisy polynomial data
    2. Fit with high-degree polynomial (overfitting)
    3. Fit with regularization
    Compare training vs test error.
    """
    np.random.seed(42)
    x_train = np.linspace(0, 1, 20)
    y_train = np.sin(2 * np.pi * x_train) + np.random.randn(20) * 0.3
    x_test = np.linspace(0, 1, 100)
    y_test = np.sin(2 * np.pi * x_test)

    # TODO: Create polynomial features (degree 15)
    # TODO: Fit without regularization (use np.linalg.lstsq)
    # TODO: Fit with L2 regularization (Ridge: (X^T X + lambda I)^-1 X^T y)
    # TODO: Compare train/test MSE
    pass

if __name__ == "__main__":
    w = np.array([0.5, -1.2, 0.8, 0.3])
    print(f"L2 reg: {l2_regularization(w, 0.01):.6f}")
    print(f"L1 reg: {l1_regularization(w, 0.01):.6f}")
    print()

    np.random.seed(42)
    X = np.random.randn(5, 4)
    out, mask = dropout_forward(X, dropout_rate=0.5)
    print(f"Dropout mask:\n{mask}")
    print(f"Active neurons: {mask.sum()}/{mask.size}")
    print()

    demonstrate_overfitting()
