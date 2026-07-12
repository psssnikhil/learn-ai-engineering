---
title: Convolutional Neural Networks (CNNs)
description: >-
  Learn how neural networks process images through convolution and pooling
  operations
duration: 45 min
difficulty: intermediate
has_code: true
youtube: 'https://www.youtube.com/watch?v=KuXjwB4LzSA'
objectives:
  - Understand convolution operations
  - Explain pooling layers
  - Build a simple CNN
  - Classify images
---
# Convolutional Neural Networks (CNNs)

## How Computers See Images

Traditional neural networks treat images as flat arrays:

```
Image (28×28 pixels)
     ↓
Flatten to 784 numbers
     ↓
Feed to neural network
```

**Problem**: Loses spatial information! Pixels near each other should be treated together.

**Solution**: Convolutional Neural Networks (CNNs)!

---

## The Big Idea: Local Patterns

Images have **local patterns**:
- Edges
- Textures
- Shapes
- Objects

CNNs use **filters** (also called kernels) to detect these patterns.

---

## Convolution Operation

### What is Convolution?

Slide a small filter over the image and compute dot products:

```
Image (5×5):          Filter (3×3):       Output (3×3):
[1 2 3 4 5]           [1 0 -1]
[0 1 2 3 4]           [1 0 -1]            Convolve
[0 0 1 2 3]     *     [1 0 -1]       →    Result
[0 0 0 1 2]
[0 0 0 0 1]
```

### Step-by-Step Example

**Filter (Edge Detector)**:
```
[1  0 -1]
[1  0 -1]
[1  0 -1]
```

**Image Patch**:
```
[50 50 100]
[50 50 100]
[50 50 100]
```

**Computation**:
```
Result = 1×50 + 0×50 + (-1)×100 +
         1×50 + 0×50 + (-1)×100 +
         1×50 + 0×50 + (-1)×100
       = -150
```

Negative value = **vertical edge detected!**

---

## CNN Architecture

```
Input Image
     ↓
Convolution Layer (learn features)
     ↓
Activation (ReLU)
     ↓
Pooling Layer (reduce size)
     ↓
Convolution Layer
     ↓
Activation (ReLU)
     ↓
Pooling Layer
     ↓
Flatten
     ↓
Fully Connected Layers
     ↓
Output (class probabilities)
```

---

## Convolution Layer

### Parameters:
- **Filters**: Number of filters to learn (e.g., 32, 64)
- **Kernel Size**: Size of filter (e.g., 3×3, 5×5)
- **Stride**: How many pixels to move (usually 1)
- **Padding**: Add zeros around borders to control output size

### Example:

```python
import numpy as np

def convolve2d(image, kernel):
    """
    Perform 2D convolution
    
    Args:
        image: (H, W) array
        kernel: (k, k) array
    
    Returns:
        output: Convolved result
    """
    image_height, image_width = image.shape
    kernel_size = kernel.shape[0]
    
    # Output size (no padding, stride=1)
    output_height = image_height - kernel_size + 1
    output_width = image_width - kernel_size + 1
    
    output = np.zeros((output_height, output_width))
    
    # Slide kernel over image
    for i in range(output_height):
        for j in range(output_width):
            # Extract patch
            patch = image[i:i+kernel_size, j:j+kernel_size]
            
            # Element-wise multiply and sum
            output[i, j] = np.sum(patch * kernel)
    
    return output


# Example: Edge detection
image = np.array([
    [0, 0, 0, 200, 200],
    [0, 0, 0, 200, 200],
    [0, 0, 0, 200, 200],
    [0, 0, 0, 200, 200],
    [0, 0, 0, 200, 200]
])

# Vertical edge detector
kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
])

result = convolve2d(image, kernel)
print("Convolution result:")
print(result)
# High values where there's a vertical edge!
```

---

## Common Filters

### 1. Vertical Edge Detector
```
[ 1  0 -1]
[ 1  0 -1]
[ 1  0 -1]
```

### 2. Horizontal Edge Detector
```
[ 1  1  1]
[ 0  0  0]
[-1 -1 -1]
```

### 3. Blur
```
[1/9 1/9 1/9]
[1/9 1/9 1/9]
[1/9 1/9 1/9]
```

### 4. Sharpen
```
[ 0 -1  0]
[-1  5 -1]
[ 0 -1  0]
```

**Key insight**: CNNs **learn** these filters automatically!

---

## Pooling Layers

**Purpose**: Reduce spatial dimensions while keeping important features.

### Max Pooling (Most Common)

Take the maximum value in each region:

```
Input (4×4):          Max Pooling (2×2, stride=2):
[1  3  2  4]
[5  6  7  8]     →    [6  8]
[3  2  1  2]          [3  4]
[1  1  3  4]
```

**Why it works**:
- Preserves strongest activations (features)
- Reduces computation
- Provides translation invariance (shift in input → same output)

### Average Pooling

Take the average instead of max:

```
[1  3  2  4]
[5  6  7  8]     →    [3.75  5.25]
[3  2  1  2]          [1.75  2.50]
[1  1  3  4]
```

---

## Complete CNN Implementation

```python
import numpy as np

class SimpleCNN:
    """Simple CNN for MNIST-like images"""
    
    def __init__(self):
        # Conv layer: 1 input channel, 8 filters, 3×3 kernels
        self.conv1_filters = np.random.randn(8, 3, 3) * 0.1
        self.conv1_bias = np.zeros(8)
        
        # Fully connected layer (after flattening)
        # For 28×28 input: after conv (26×26) and pool (13×13) → 13×13×8 = 1352
        self.fc_weights = np.random.randn(1352, 10) * 0.1
        self.fc_bias = np.zeros(10)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def convolve(self, image, kernel):
        """2D convolution"""
        h, w = image.shape
        k = kernel.shape[0]
        output_h, output_w = h - k + 1, w - k + 1
        output = np.zeros((output_h, output_w))
        
        for i in range(output_h):
            for j in range(output_w):
                output[i, j] = np.sum(image[i:i+k, j:j+k] * kernel)
        
        return output
    
    def max_pool(self, image, pool_size=2):
        """Max pooling"""
        h, w = image.shape
        output_h, output_w = h // pool_size, w // pool_size
        output = np.zeros((output_h, output_w))
        
        for i in range(output_h):
            for j in range(output_w):
                patch = image[i*pool_size:(i+1)*pool_size, 
                            j*pool_size:(j+1)*pool_size]
                output[i, j] = np.max(patch)
        
        return output
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input image (28, 28)
        
        Returns:
            probabilities: Class probabilities (10,)
        """
        # Convolution layer
        conv_outputs = []
        for i in range(8):  # 8 filters
            conv = self.convolve(x, self.conv1_filters[i])
            conv = self.relu(conv + self.conv1_bias[i])
            pooled = self.max_pool(conv)
            conv_outputs.append(pooled)
        
        # Stack and flatten
        conv_stack = np.stack(conv_outputs, axis=0)  # (8, 13, 13)
        flattened = conv_stack.flatten()  # (1352,)
        
        # Fully connected layer
        logits = flattened @ self.fc_weights + self.fc_bias
        probabilities = self.softmax(logits)
        
        return probabilities


# Example usage
cnn = SimpleCNN()

# Create dummy image (28×28)
image = np.random.randn(28, 28)

# Forward pass
probs = cnn.forward(image)

print("Output probabilities:")
print(probs)
print(f"Predicted class: {np.argmax(probs)}")
```

---

## Why CNNs Work for Images

### 1. Parameter Sharing

**Traditional NN**: Each weight connects to one pixel  
**CNN**: Same filter applied to entire image

```
Traditional: 28×28×100 = 78,400 parameters (first layer!)
CNN: 3×3×32 = 288 parameters (first layer)

🎉 99% fewer parameters!
```

---

### 2. Translation Invariance

If object moves in image, CNN still detects it:

```
Cat in top-left → Detected ✅
Cat in bottom-right → Detected ✅
```

Same filter slides everywhere!

---

### 3. Hierarchical Learning

```
Layer 1: Edges, colors
Layer 2: Textures, simple shapes
Layer 3: Parts (eyes, wheels, etc.)
Layer 4: Objects (faces, cars, etc.)
```

Each layer builds on previous!

---

## Famous CNN Architectures

### LeNet-5 (1998) - The Pioneer
```
Input (32×32)
 → Conv (6 filters)
 → Pool
 → Conv (16 filters)
 → Pool
 → FC (120)
 → FC (84)
 → Output (10)
```
**Use**: Handwritten digit recognition

---

### AlexNet (2012) - ImageNet Winner
```
Input (224×224×3)
 → Conv (96 filters, 11×11)
 → Pool
 → Conv (256 filters, 5×5)
 → Pool
 → Conv (384 filters, 3×3)
 → Conv (384 filters, 3×3)
 → Conv (256 filters, 3×3)
 → Pool
 → FC (4096)
 → FC (4096)
 → Output (1000 classes)
```
**Breakthrough**: First deep CNN to win ImageNet

---

### VGG-16 (2014) - Deeper is Better
```
16 weight layers
All 3×3 convolutions
Very deep, very accurate
```

---

### ResNet (2015) - Skip Connections
```
Identity shortcuts:
x → Conv → Conv → (+) → Output
└──────────────────┘

Enables 100+ layer networks!
```

---

## 📹 Recommended Videos

- [3Blue1Brown: CNNs](https://www.youtube.com/watch?v=KuXjwB4LzSA) - Beautiful visualization
- [Stanford CS231n: CNNs for Visual Recognition](https://www.youtube.com/watch?v=bNb2fEVKeEo)
- [Computerphile: How CNNs Work](https://www.youtube.com/watch?v=py5byOOHZM8)

---

## 🎯 Key Takeaways

1. **CNNs** use convolution to detect local patterns
2. **Filters** learn features automatically
3. **Pooling** reduces spatial dimensions
4. **Parameter sharing** makes CNNs efficient
5. **Hierarchical learning**: edges → textures → objects
6. CNNs revolutionized computer vision

---

## 📹 Recommended Videos

- [But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA) — 3Blue1Brown's visual explanation
- [CNNs Explained](https://www.youtube.com/watch?v=YRhxdVk_sIs) — Deeplizard full course intro
- [CS231n CNNs for Visual Recognition](https://www.youtube.com/watch?v=bNb2fEVKeEo) — Stanford lecture by Andrej Karpathy

---

## 📚 Additional Resources

- [A Beginner's Guide to CNNs](https://cs231n.github.io/convolutional-networks/) — Stanford CS231n notes
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/) — Interactive visual CNN tool (Georgia Tech)
- [Understanding Convolutions](https://colah.github.io/posts/2014-07-Understanding-Convolutions/) — Chris Olah's blog

---

## 🚀 Next Lesson

**Lesson 9**: Recurrent Neural Networks (RNNs) & LSTMs
- Processing sequences
- Memory in neural networks
- Text generation
- Time series prediction

**Let's learn about memory!** 🧠
