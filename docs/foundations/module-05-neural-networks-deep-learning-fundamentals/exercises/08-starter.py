import numpy as np

# CNNs Intro - Exercise

def convolve2d(image, kernel):
    """
    TODO: Implement 2D convolution (no padding, stride=1).
    image: shape (H, W)
    kernel: shape (kH, kW)
    Return output of shape (H-kH+1, W-kW+1)
    """
    pass

def max_pool2d(image, pool_size=2):
    """
    TODO: Implement 2D max pooling.
    Divide image into pool_size x pool_size blocks.
    Return the max value from each block.
    """
    pass

def apply_filters(image):
    """
    TODO: Apply edge detection filters to an image.
    1. Horizontal edge: [[-1,-1,-1],[0,0,0],[1,1,1]]
    2. Vertical edge: [[-1,0,1],[-1,0,1],[-1,0,1]]
    Return both filtered images.
    """
    pass

if __name__ == "__main__":
    # Create a simple 6x6 image
    image = np.array([
        [0,0,0,0,0,0],
        [0,1,1,1,1,0],
        [0,1,0,0,1,0],
        [0,1,0,0,1,0],
        [0,1,1,1,1,0],
        [0,0,0,0,0,0]
    ], dtype=float)

    # Test convolution
    kernel = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    result = convolve2d(image, kernel)
    print(f"Convolution result shape: {result.shape}")
    print(f"Result:\n{result}")
    print()

    # Test max pooling
    pooled = max_pool2d(image, pool_size=2)
    print(f"Max pool result:\n{pooled}")
    print()

    # Test edge detection
    h_edges, v_edges = apply_filters(image)
    print(f"Horizontal edges:\n{h_edges}")
    print(f"Vertical edges:\n{v_edges}")
