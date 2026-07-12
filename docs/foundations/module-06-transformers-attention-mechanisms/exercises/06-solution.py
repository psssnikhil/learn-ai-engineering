import numpy as np

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def layer_norm(x, eps=1e-5):
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps)

def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores += mask
    return softmax(scores) @ V, softmax(scores)

def encoder(X, d_model, d_k, d_ff):
    W_Q = np.random.randn(d_model, d_k) * 0.1
    W_K = np.random.randn(d_model, d_k) * 0.1
    W_V = np.random.randn(d_model, d_k) * 0.1
    W_O = np.random.randn(d_k, d_model) * 0.1
    attn, _ = attention(X @ W_Q, X @ W_K, X @ W_V)
    X = layer_norm(X + attn @ W_O)
    W1 = np.random.randn(d_model, d_ff) * 0.1
    W2 = np.random.randn(d_ff, d_model) * 0.1
    ff = np.maximum(0, X @ W1) @ W2
    return layer_norm(X + ff)

def cross_attention(decoder_state, encoder_output, d_k):
    d_model = decoder_state.shape[-1]
    Q = decoder_state @ (np.random.randn(d_model, d_k) * 0.1)
    K = encoder_output @ (np.random.randn(d_model, d_k) * 0.1)
    V = encoder_output @ (np.random.randn(d_model, d_k) * 0.1)
    return attention(Q, K, V)

def decoder_step(decoder_input, encoder_output, d_model, d_k, d_ff):
    W_Q = np.random.randn(d_model, d_k) * 0.1
    W_K = np.random.randn(d_model, d_k) * 0.1
    W_V = np.random.randn(d_model, d_k) * 0.1
    W_O = np.random.randn(d_k, d_model) * 0.1
    mask = np.triu(np.ones((len(decoder_input), len(decoder_input))) * -1e9, k=1)
    sa, _ = attention(decoder_input @ W_Q, decoder_input @ W_K, decoder_input @ W_V, mask)
    X = layer_norm(decoder_input + sa @ W_O)
    ca, _ = cross_attention(X, encoder_output, d_k)
    W_CO = np.random.randn(d_k, d_model) * 0.1
    X = layer_norm(X + ca @ W_CO)
    W1 = np.random.randn(d_model, d_ff) * 0.1
    W2 = np.random.randn(d_ff, d_model) * 0.1
    return layer_norm(X + np.maximum(0, X @ W1) @ W2)

if __name__ == "__main__":
    np.random.seed(42)
    src_len, tgt_len, d_model, d_k, d_ff = 5, 3, 16, 8, 32
    src = np.random.randn(src_len, d_model)
    tgt = np.random.randn(tgt_len, d_model)
    enc_out = encoder(src, d_model, d_k, d_ff)
    print(f"Encoder output shape: {enc_out.shape}")
    dec_out = decoder_step(tgt, enc_out, d_model, d_k, d_ff)
    print(f"Decoder output shape: {dec_out.shape}")
