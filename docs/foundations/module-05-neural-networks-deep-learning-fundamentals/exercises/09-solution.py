import numpy as np

# RNNs & LSTMs - Solution

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def rnn_cell(x_t, h_prev, W_xh, W_hh, b_h):
    return np.tanh(W_xh @ x_t + W_hh @ h_prev + b_h)

def rnn_forward(X, hidden_size):
    seq_len, input_size = X.shape
    W_xh = np.random.randn(hidden_size, input_size) * 0.1
    W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
    b_h = np.zeros(hidden_size)

    h = np.zeros(hidden_size)
    hidden_states = []
    for t in range(seq_len):
        h = rnn_cell(X[t], h, W_xh, W_hh, b_h)
        hidden_states.append(h)
    return np.array(hidden_states)

def lstm_cell(x_t, h_prev, c_prev, Wf, Wi, Wc, Wo, bf, bi, bc, bo):
    combined = np.concatenate([h_prev, x_t])
    f_t = sigmoid(Wf @ combined + bf)
    i_t = sigmoid(Wi @ combined + bi)
    c_hat = np.tanh(Wc @ combined + bc)
    c_t = f_t * c_prev + i_t * c_hat
    o_t = sigmoid(Wo @ combined + bo)
    h_t = o_t * np.tanh(c_t)
    return h_t, c_t

if __name__ == "__main__":
    np.random.seed(42)

    seq_len, input_size, hidden_size = 5, 3, 4
    X = np.random.randn(seq_len, input_size)
    hidden_states = rnn_forward(X, hidden_size)
    print(f"RNN hidden states shape: {hidden_states.shape}")
    print(f"Last hidden state: {hidden_states[-1]}")
    print()

    x_t = np.random.randn(input_size)
    h_prev = np.zeros(hidden_size)
    W_xh = np.random.randn(hidden_size, input_size) * 0.1
    W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
    b_h = np.zeros(hidden_size)
    h_t = rnn_cell(x_t, h_prev, W_xh, W_hh, b_h)
    print(f"RNN cell output: {h_t}")
