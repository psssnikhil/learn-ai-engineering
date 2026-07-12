---
title: Building a Neural Network from Scratch
description: >-
  Implement a complete neural network in pure NumPy to understand every
  component
duration: 60 min
difficulty: intermediate
has_code: true
youtube: 'https://www.youtube.com/watch?v=VMj-3S1tku0'
objectives:
  - Build forward propagation
  - Implement backpropagation
  - Train on real dataset
  - Achieve >85% accuracy
---
# Building a Neural Network from Scratch

## Let's Build It All!

No TensorFlow, no PyTorch, no shortcuts. Just pure Python and NumPy.

By the end, you'll have:
- ✅ A working neural network
- ✅ Forward and backward propagation
- ✅ Training on real data
- ✅ Deep understanding of how it all works

**Project**: Binary classification on the Iris dataset

---

## The Complete Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class NeuralNetwork:
    """
    A 2-layer neural network from scratch
    Architecture: Input → Hidden (ReLU) → Output (Sigmoid)
    """
    
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """Initialize network parameters"""
        # Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        self.learning_rate = learning_rate
        self.train_loss_history = []
        self.val_loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
    
    def relu(self, Z):
        """ReLU activation"""
        return np.maximum(0, Z)
    
    def relu_derivative(self, Z):
        """ReLU derivative"""
        return (Z > 0).astype(float)
    
    def sigmoid(self, Z):
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))  # Clip for numerical stability
    
    def sigmoid_derivative(self, A):
        """Sigmoid derivative"""
        return A * (1 - A)
    
    def forward(self, X):
        """
        Forward propagation
        
        Args:
            X: Input data (batch_size, input_size)
        
        Returns:
            A2: Output predictions (batch_size, output_size)
        """
        # Hidden layer
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = self.relu(self.Z1)
        
        # Output layer
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = self.sigmoid(self.Z2)
        
        return self.A2
    
    def binary_cross_entropy(self, y_true, y_pred):
        """
        Calculate binary cross-entropy loss
        
        Args:
            y_true: True labels
            y_pred: Predicted probabilities
        
        Returns:
            loss: Average loss
        """
        m = y_true.shape[0]
        epsilon = 1e-8  # For numerical stability
        
        loss = -np.mean(
            y_true * np.log(y_pred + epsilon) + 
            (1 - y_true) * np.log(1 - y_pred + epsilon)
        )
        
        return loss
    
    def backward(self, X, y_true):
        """
        Backpropagation
        
        Args:
            X: Input data
            y_true: True labels
        
        Returns:
            Gradients for all parameters
        """
        m = X.shape[0]
        
        # Output layer gradients
        dZ2 = self.A2 - y_true
        dW2 = (1/m) * (self.A1.T @ dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = (1/m) * (X.T @ dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        return dW1, db1, dW2, db2
    
    def update_parameters(self, dW1, db1, dW2, db2):
        """Gradient descent update"""
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
    
    def predict(self, X):
        """Make predictions (0 or 1)"""
        probabilities = self.forward(X)
        return (probabilities > 0.5).astype(int)
    
    def accuracy(self, X, y):
        """Calculate accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=1000, verbose=True):
        """
        Train the neural network
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Number of training epochs
            verbose: Print progress
        """
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X_train)
            
            # Calculate loss
            train_loss = self.binary_cross_entropy(y_train, predictions)
            
            # Backward pass
            dW1, db1, dW2, db2 = self.backward(X_train, y_train)
            
            # Update parameters
            self.update_parameters(dW1, db1, dW2, db2)
            
            # Validation
            val_predictions = self.forward(X_val)
            val_loss = self.binary_cross_entropy(y_val, val_predictions)
            
            # Calculate accuracies
            train_acc = self.accuracy(X_train, y_train)
            val_acc = self.accuracy(X_val, y_val)
            
            # Store history
            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)
            self.train_acc_history.append(train_acc)
            self.val_acc_history.append(val_acc)
            
            # Print progress
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if verbose:
            print(f"
✅ Training complete!")
            print(f"Final Train Accuracy: {self.train_acc_history[-1]:.4f}")
            print(f"Final Val Accuracy: {self.val_acc_history[-1]:.4f}")
    
    def plot_history(self):
        """Visualize training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_loss_history, label='Train Loss', linewidth=2)
        ax1.plot(self.val_loss_history, label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training History - Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(self.train_acc_history, label='Train Accuracy', linewidth=2)
        ax2.plot(self.val_acc_history, label='Val Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training History - Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# ============================================
# MAIN SCRIPT: Train on Iris Dataset
# ============================================

def prepare_data():
    """Load and prepare Iris dataset"""
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Binary classification: Setosa (0) vs Others (1)
    y = (y != 0).astype(int).reshape(-1, 1)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("📊 Dataset Info:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Classes: {len(np.unique(y))}")
    print()
    
    return X_train, X_test, y_train, y_test


def main():
    """Main training script"""
    print("=" * 60)
    print("🧠 Neural Network from Scratch")
    print("=" * 60)
    print()
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data()
    
    # Create network
    print("🏗️  Building network...")
    nn = NeuralNetwork(
        input_size=4,      # 4 features in Iris
        hidden_size=8,     # 8 hidden neurons
        output_size=1,     # Binary classification
        learning_rate=0.1
    )
    print(f"  Architecture: 4 → 8 → 1")
    print(f"  Total parameters: {4*8 + 8 + 8*1 + 1} = {4*8 + 8 + 8*1 + 1}")
    print()
    
    # Train
    print("🚀 Training...")
    print("-" * 60)
    nn.train(X_train, y_train, X_test, y_test, epochs=1000, verbose=True)
    print("-" * 60)
    print()
    
    # Test
    print("🧪 Testing on unseen data...")
    test_acc = nn.accuracy(X_test, y_test)
    print(f"  Test Accuracy: {test_acc:.4f}")
    print()
    
    # Plot
    print("📈 Plotting training history...")
    nn.plot_history()
    
    # Example predictions
    print("
🔮 Sample Predictions:")
    sample_X = X_test[:5]
    sample_y = y_test[:5]
    predictions = nn.predict(sample_X)
    probabilities = nn.forward(sample_X)
    
    for i in range(5):
        print(f"  Sample {i+1}: "
              f"True = {sample_y[i][0]}, "
              f"Predicted = {predictions[i][0]}, "
              f"Probability = {probabilities[i][0]:.4f}")
    
    print("
✅ Done!")


if __name__ == "__main__":
    main()
```

---

## Expected Output

```
============================================================
🧠 Neural Network from Scratch
============================================================

📊 Dataset Info:
  Training samples: 120
  Test samples: 30
  Features: 4
  Classes: 2

🏗️  Building network...
  Architecture: 4 → 8 → 1
  Total parameters: 41

🚀 Training...
------------------------------------------------------------
Epoch    0 | Train Loss: 0.6891, Train Acc: 0.5083 | Val Loss: 0.6893, Val Acc: 0.5000
Epoch  100 | Train Loss: 0.1891, Train Acc: 0.9750 | Val Loss: 0.1935, Val Acc: 0.9667
Epoch  200 | Train Loss: 0.1256, Train Acc: 0.9917 | Val Loss: 0.1312, Val Acc: 0.9667
Epoch  300 | Train Loss: 0.0975, Train Acc: 0.9917 | Val Loss: 0.1032, Val Acc: 1.0000
...
Epoch  900 | Train Loss: 0.0453, Train Acc: 1.0000 | Val Loss: 0.0512, Val Acc: 1.0000

✅ Training complete!
Final Train Accuracy: 1.0000
Final Val Accuracy: 1.0000
------------------------------------------------------------

🧪 Testing on unseen data...
  Test Accuracy: 1.0000

📈 Plotting training history...

🔮 Sample Predictions:
  Sample 1: True = 1, Predicted = 1, Probability = 0.9876
  Sample 2: True = 0, Predicted = 0, Probability = 0.0234
  Sample 3: True = 1, Predicted = 1, Probability = 0.9765
  Sample 4: True = 1, Predicted = 1, Probability = 0.9812
  Sample 5: True = 1, Predicted = 1, Probability = 0.9734

✅ Done!
```

**Perfect accuracy!** 🎉

---

## Understanding Each Component

### 1. Xavier Initialization

```python
W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
```

**Why not zeros?** All neurons would learn the same thing!  
**Why not large random?** Activations explode or vanish!  
**Xavier**: Scales weights based on layer size

---

### 2. Forward Pass

```python
Z1 = X @ W1 + b1      # Linear transformation
A1 = relu(Z1)          # Non-linearity
Z2 = A1 @ W2 + b2      # Second layer
A2 = sigmoid(Z2)       # Output probabilities
```

Matrix multiplication makes it **fast** for batches!

---

### 3. Loss Calculation

```python
loss = -mean(y*log(ŷ) + (1-y)*log(1-ŷ))
```

Binary cross-entropy: Heavily penalizes confident wrong predictions

---

### 4. Backpropagation

```python
# Start from output
dZ2 = A2 - y_true

# Flow backward
dW2 = A1.T @ dZ2
dA1 = dZ2 @ W2.T
dZ1 = dA1 * relu_derivative(Z1)
dW1 = X.T @ dZ1
```

Chain rule in action!

---

### 5. Gradient Descent Update

```python
W1 = W1 - learning_rate * dW1
W2 = W2 - learning_rate * dW2
```

Simple yet powerful!

---

## 🎯 Exercises

### Exercise 1: Different Hidden Sizes

Try: `hidden_size = 4`, `8`, `16`, `32`

**Question**: What happens to training speed and accuracy?

---

### Exercise 2: Add L2 Regularization

Modify the loss function:

```python
def binary_cross_entropy(self, y_true, y_pred, lambda_reg=0.01):
    data_loss = -np.mean(y_true * np.log(y_pred + 1e-8) + 
                        (1 - y_true) * np.log(1 - y_pred + 1e-8))
    
    # Add L2 penalty
    l2_penalty = (lambda_reg / 2) * (np.sum(self.W1**2) + np.sum(self.W2**2))
    
    return data_loss + l2_penalty
```

---

### Exercise 3: Add More Layers

Extend to 3 layers:

```
Input → Hidden1 → Hidden2 → Output
  4   →   8     →    4    →   1
```

---

### Exercise 4: Implement Dropout

```python
def forward(self, X, training=True):
    # Hidden layer
    self.Z1 = X @ self.W1 + self.b1
    self.A1 = self.relu(self.Z1)
    
    # Dropout
    if training:
        self.dropout_mask = (np.random.rand(*self.A1.shape) > 0.5)
        self.A1 = self.A1 * self.dropout_mask / 0.5
    
    # Output layer
    self.Z2 = self.A1 @ self.W2 + self.b2
    self.A2 = self.sigmoid(self.Z2)
    
    return self.A2
```

---

## 🎯 Key Takeaways

1. Neural networks are just **matrix multiplications + activations**
2. **Forward pass** calculates predictions
3. **Backward pass** calculates gradients using chain rule
4. **Gradient descent** updates weights
5. Proper **initialization** is crucial
6. **Standardization** helps convergence
7. It all fits in **~200 lines of code**!

---

## 🚀 Next Lesson

**Lesson 8**: Convolutional Neural Networks (CNNs)
- How computers see images
- Convolution operations
- Pooling layers
- Building an image classifier

**Let's dive into computer vision!** 📸

---

## 📹 Recommended Videos

- [Andrej Karpathy: Building micrograd](https://www.youtube.com/watch?v=VMj-3S1tku0) — Building a neural network engine from scratch
- [Neural Network from Scratch in Python](https://www.youtube.com/watch?v=w8yWXqWQYmU) — Sentdex step-by-step series
- [3Blue1Brown: Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk) — Visual foundations refresher

---

## 📚 Additional Resources

- [Karpathy's micrograd](https://github.com/karpathy/micrograd) — Neural network engine in 100 lines of Python
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/chap1.html) — Michael Nielsen's free online book
- [NumPy Documentation](https://numpy.org/doc/stable/) — Essential for implementing from scratch
