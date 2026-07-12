import numpy as np

# Encoder-Decoder Architecture - Exercise

def softmax(x, axis=-1):
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def encoder(X, d_model, d_k, d_ff):
    """
    TODO: Simple encoder: self-attention + FFN with residual + layer norm.
    Return encoder output.
    """
    pass

def cross_attention(decoder_state, encoder_output, d_k):
    """
    TODO: Cross-attention: decoder queries attend to encoder keys/values.
    Q from decoder_state, K and V from encoder_output.
    Return output and weights.
    """
    pass

def decoder_step(decoder_input, encoder_output, d_model, d_k, d_ff):
    """
    TODO: One decoder step:
    1. Masked self-attention on decoder_input
    2. Cross-attention to encoder_output
    3. Feed-forward
    Return decoder output.
    """
    pass

if __name__ == "__main__":
    np.random.seed(42)
    src_len, tgt_len, d_model, d_k, d_ff = 5, 3, 16, 8, 32

    src = np.random.randn(src_len, d_model)
    tgt = np.random.randn(tgt_len, d_model)

    enc_out = encoder(src, d_model, d_k, d_ff)
    print(f"Encoder output shape: {enc_out.shape}")

    dec_out = decoder_step(tgt, enc_out, d_model, d_k, d_ff)
    print(f"Decoder output shape: {dec_out.shape}")
