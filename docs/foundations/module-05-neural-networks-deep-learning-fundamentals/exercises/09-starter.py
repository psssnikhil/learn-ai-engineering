import numpy as np

# RNNs & LSTMs - Exercise

def rnn_cell(x_t, h_prev, W_xh, W_hh, b_h):
    """
    TODO: Implement a single RNN cell.
    h_t = tanh(W_xh @ x_t + W_hh @ h_prev + b_h)
    Return h_t.
    """
    pass

def rnn_forward(X, hidden_size):
    """
    TODO: Forward pass of an RNN over a sequence.
    X: shape (seq_len, input_size)
    1. Initialize random weights and zero hidden state
    2. Process each timestep with rnn_cell
    3. Return all hidden states, shape (seq_len, hidden_size)
    """
    pass

def lstm_cell(x_t, h_prev, c_prev, Wf, Wi, Wc, Wo, bf, bi, bc, bo):
    """
    TODO: Implement a single LSTM cell.
    Forget gate: f_t = sigmoid(Wf @ [h_prev, x_t] + bf)
    Input gate: i_t = sigmoid(Wi @ [h_prev, x_t] + bi)
    Candidate: c_hat = tanh(Wc @ [h_prev, x_t] + bc)
    Cell state: c_t = f_t * c_prev + i_t * c_hat
    Output gate: o_t = sigmoid(Wo @ [h_prev, x_t] + bo)
    Hidden state: h_t = o_t * tanh(c_t)
    Return h_t, c_t.
    """
    pass

if __name__ == "__main__":
    np.random.seed(42)

    # Test RNN
    seq_len, input_size, hidden_size = 5, 3, 4
    X = np.random.randn(seq_len, input_size)
    hidden_states = rnn_forward(X, hidden_size)
    print(f"RNN hidden states shape: {hidden_states.shape}")
    print(f"Last hidden state: {hidden_states[-1]}")
    print()

    # Test RNN cell
    x_t = np.random.randn(input_size)
    h_prev = np.zeros(hidden_size)
    W_xh = np.random.randn(hidden_size, input_size) * 0.1
    W_hh = np.random.randn(hidden_size, hidden_size) * 0.1
    b_h = np.zeros(hidden_size)
    h_t = rnn_cell(x_t, h_prev, W_xh, W_hh, b_h)
    print(f"RNN cell output: {h_t}")
