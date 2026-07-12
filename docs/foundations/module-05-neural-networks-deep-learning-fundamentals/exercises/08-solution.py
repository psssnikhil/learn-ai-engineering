import numpy as np

# CNNs Intro - Solution

def convolve2d(image, kernel):
    H, W = image.shape
    kH, kW = kernel.shape
    out_H, out_W = H - kH + 1, W - kW + 1
    output = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            output[i, j] = np.sum(image[i:i+kH, j:j+kW] * kernel)
    return output

def max_pool2d(image, pool_size=2):
    H, W = image.shape
    out_H, out_W = H // pool_size, W // pool_size
    output = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            block = image[i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size]
            output[i, j] = np.max(block)
    return output

def apply_filters(image):
    h_kernel = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    v_kernel = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    return convolve2d(image, h_kernel), convolve2d(image, v_kernel)

if __name__ == "__main__":
    image = np.array([
        [0,0,0,0,0,0],
        [0,1,1,1,1,0],
        [0,1,0,0,1,0],
        [0,1,0,0,1,0],
        [0,1,1,1,1,0],
        [0,0,0,0,0,0]
    ], dtype=float)

    kernel = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    result = convolve2d(image, kernel)
    print(f"Convolution result shape: {result.shape}")
    print(f"Result:\n{result}")
    print()

    pooled = max_pool2d(image, pool_size=2)
    print(f"Max pool result:\n{pooled}")
    print()

    h_edges, v_edges = apply_filters(image)
    print(f"Horizontal edges:\n{h_edges}")
    print(f"Vertical edges:\n{v_edges}")
