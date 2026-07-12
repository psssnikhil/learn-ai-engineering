---
title: Tokenization Deep Dive
description: Master how text is converted to tokens that LLMs can process
duration: 30 min
difficulty: intermediate
has_code: false
youtube: 'https://www.youtube.com/watch?v=zduSFxRajkE'
---
# Tokenization

## What are Tokens?

Tokens are pieces of text (words, subwords, characters)

```
"Hello world!" → ["Hello", " world", "!"]
"GPT-3" → ["G", "PT", "-", "3"] or ["GPT", "-", "3"]
```

## Why Not Just Words?

**Problem with word-level**:
- Huge vocabulary (millions of words)
- Can't handle new words
- Language-specific

**Solution**: Subword tokenization!

## BPE (Byte Pair Encoding)

Used by GPT models

```python
# Example BPE
vocabulary = ["h", "e", "l", "o", "ll", "hello"]

"hello" → ["hello"] (if in vocab)
"hellooo" → ["hello", "o", "o"] (subwords)
```

## Implementation

```python
from transformers import AutoTokenizer

# Load GPT-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Encode
text = "Hello, how are you?"
tokens = tokenizer.encode(text)
print(tokens)  # [15496, 11, 703, 389, 345, 30]

# Decode
decoded = tokenizer.decode(tokens)
print(decoded)  # "Hello, how are you?"

# See individual tokens
for token in tokens:
    print(f"{token}: {tokenizer.decode([token])}")
```

## Token Count = Cost!

```
GPT-4 pricing:
- Input: $0.03 per 1K tokens
- Output: $0.06 per 1K tokens

"Hello world" ≈ 2 tokens
1000 words ≈ 750-1333 tokens
```

---

## 📚 Additional Resources

- [Andrej Karpathy: Let's Build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) — BPE tokenization from scratch
- [tiktoken](https://github.com/openai/tiktoken) — OpenAI's fast tokenization library
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/) — Fast tokenizer library docs
- [SentencePiece](https://github.com/google/sentencepiece) — Google's unsupervised text tokenizer
