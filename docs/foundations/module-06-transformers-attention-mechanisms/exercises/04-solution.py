import numpy as np

def sinusoidal_positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe

def relative_position_bias(seq_len, num_heads):
    positions = np.arange(seq_len)
    bias = positions[:, None] - positions[None, :]
    return bias.astype(float)

def show_encoding_properties(pe):
    pos0 = pe[0]
    norm0 = np.linalg.norm(pos0)
    print("Cosine similarity of position 0 with other positions:")
    for i in range(len(pe)):
        sim = np.dot(pos0, pe[i]) / (norm0 * np.linalg.norm(pe[i]))
        bar = "#" * int(sim * 30)
        print(f"  pos {i:2d}: {sim:.4f} {bar}")

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
