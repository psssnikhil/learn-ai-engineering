import numpy as np

# Loss Functions - Exercise

def mse_loss(predictions, targets):
    """TODO: Mean Squared Error. Formula: mean((pred - target)^2)"""
    pass

def mae_loss(predictions, targets):
    """TODO: Mean Absolute Error. Formula: mean(|pred - target|)"""
    pass

def binary_cross_entropy(predictions, targets, epsilon=1e-15):
    """
    TODO: Binary Cross-Entropy Loss.
    Formula: -mean(target * log(pred) + (1-target) * log(1-pred))
    Clip predictions to [epsilon, 1-epsilon] to avoid log(0).
    """
    pass

def categorical_cross_entropy(predictions, targets, epsilon=1e-15):
    """
    TODO: Categorical Cross-Entropy Loss.
    predictions: shape (n, num_classes) - predicted probabilities
    targets: shape (n, num_classes) - one-hot encoded targets
    Formula: -mean(sum(target * log(pred)))
    """
    pass

if __name__ == "__main__":
    pred = np.array([2.5, 0.5, 2.1, 7.0])
    target = np.array([3.0, 0.5, 2.0, 7.5])
    print(f"MSE Loss: {mse_loss(pred, target)}")
    print(f"MAE Loss: {mae_loss(pred, target)}")
    print()

    pred_prob = np.array([0.9, 0.2, 0.8, 0.1])
    target_binary = np.array([1, 0, 1, 0])
    print(f"BCE Loss: {binary_cross_entropy(pred_prob, target_binary)}")
    print()

    pred_cat = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    target_cat = np.array([[1, 0, 0], [0, 1, 0]])
    print(f"CCE Loss: {categorical_cross_entropy(pred_cat, target_cat)}")
