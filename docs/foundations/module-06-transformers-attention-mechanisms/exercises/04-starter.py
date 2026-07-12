import numpy as np

# Positional Encoding - Exercise

def sinusoidal_positional_encoding(seq_len, d_model):
    """
    TODO: Implement sinusoidal positional encoding.
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    Return array of shape (seq_len, d_model).
    """
    pass

def relative_position_bias(seq_len, num_heads):
    """
    TODO: Compute relative position bias matrix.
    bias[i][j] = learnable parameter based on (i - j).
    For simplicity, use bias = clip(i - j, -max_dist, max_dist) as float.
    Return shape (seq_len, seq_len).
    """
    pass

def show_encoding_properties(pe):
    """
    TODO: Show that nearby positions have more similar encodings.
    Compute cosine similarity between position 0 and all others.
    Print the similarities.
    """
    pass

if __name__ == "__main__":
    pe = sinusoidal_positional_encoding(10, 16)
    print(f"PE shape: {pe.shape}")
    print(f"Position 0: {pe[0, :8].round(4)}")
    print(f"Position 1: {pe[1, :8].round(4)}")
    print()

    show_encoding_properties(pe)
    print()

    bias = relative_position_bias(6, 2)
    print(f"Relative position bias:\n{bias}")
