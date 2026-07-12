import numpy as np

# Loss Functions - Solution

def mse_loss(predictions, targets):
    return np.mean((predictions - targets) ** 2)

def mae_loss(predictions, targets):
    return np.mean(np.abs(predictions - targets))

def binary_cross_entropy(predictions, targets, epsilon=1e-15):
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    return -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))

def categorical_cross_entropy(predictions, targets, epsilon=1e-15):
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    return -np.mean(np.sum(targets * np.log(predictions), axis=1))

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
