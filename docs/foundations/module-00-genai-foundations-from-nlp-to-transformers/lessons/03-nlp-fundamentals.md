---
title: NLP Fundamentals — How Computers Process Language
description: >-
  Understand the foundations of Natural Language Processing — tokenization, word
  representations, and how text becomes numbers that neural networks can process
duration: 45 min
difficulty: beginner
has_code: true
module: module-00
---
# NLP Fundamentals — How Computers Process Language

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand why computers need to convert text to numbers | 45 min | Beginner |
| Learn how tokenization breaks text into processable units | | |
| Understand one-hot encoding, word2vec, and modern embeddings | | |
| See the evolution from bag-of-words to contextual representations | | |

---

## The Fundamental Problem

Computers work with numbers. Language is made of words. The entire history of NLP is about solving one question: **How do we represent language as numbers in a way that preserves meaning?**

```
"The cat sat on the mat"
         ↓
    ??? How ???
         ↓
[0.23, -0.45, 0.87, 0.12, ...]  (numbers a neural network can process)
```

---

## Step 1: Tokenization

Before any processing, text must be split into **tokens** — the atomic units the model works with.

### Word-Level Tokenization

```python
# Simplest approach: split on spaces
text = "The cat sat on the mat"
tokens = text.split()
print(tokens)  # ['The', 'cat', 'sat', 'on', 'the', 'mat']

# Problem: what about "don't", "New York", "state-of-the-art"?
# Problem: vocabulary gets huge (every unique word = one token)
```

### Subword Tokenization (What Modern LLMs Use)

Modern models use **Byte-Pair Encoding (BPE)** or **SentencePiece** to split words into subword tokens:

```python
# How GPT tokenizes:
"unhappiness" → ["un", "happiness"]
"tokenization" → ["token", "ization"]
"ChatGPT" → ["Chat", "G", "PT"]

# Benefits:
# - Smaller vocabulary (50,000 tokens covers all languages)
# - Can handle unseen words by decomposing them
# - Balances between character-level and word-level
```

```python
# Using the tiktoken library (OpenAI's tokenizer)
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

text = "Tokenization is fundamental to LLMs"
tokens = enc.encode(text)
print(f"Text: {text}")
print(f"Token IDs: {tokens}")
print(f"Token count: {len(tokens)}")

# Decode back to see individual tokens
for token_id in tokens:
    print(f"  {token_id} → '{enc.decode([token_id])}'")
```

---

## Step 2: From Tokens to Numbers

### One-Hot Encoding (The Naive Approach)

Represent each word as a vector with a 1 in its position and 0s everywhere else:

```python
vocabulary = ["cat", "dog", "mat", "sat"]

# One-hot vectors:
cat = [1, 0, 0, 0]
dog = [0, 1, 0, 0]
mat = [0, 0, 1, 0]
sat = [0, 0, 0, 1]

# Problems:
# 1. Vectors are HUGE (vocabulary of 50,000 words → 50,000-dimensional vectors)
# 2. No notion of similarity (cat and dog are equally different as cat and mat)
# 3. No semantic meaning captured
```

### Word2Vec — Words as Meaning Vectors (2013)

The breakthrough idea: train a neural network to predict words from their context. Words that appear in similar contexts get similar vectors.

```
"The cat sat on the ___"     → "mat" (likely)
"The dog sat on the ___"     → "mat" (likely)
→ Therefore: vector("cat") ≈ vector("dog")  (similar contexts)
```

```python
# Famous Word2Vec relationships:
# king - man + woman ≈ queen
# Paris - France + Italy ≈ Rome

# This showed that vector arithmetic captures semantic meaning!

import numpy as np

# Simplified example (real vectors are 100-300 dimensions):
king = np.array([0.5, 0.3, -0.1, 0.8])
man = np.array([0.4, 0.2, -0.3, 0.1])
woman = np.array([0.45, 0.25, 0.3, 0.15])

# king - man + woman should ≈ queen
result = king - man + woman
print(f"king - man + woman = {result}")
# The result vector would be close to the "queen" vector in a trained model
```

---

## Step 3: The Evolution of Word Representations

| Era | Method | Key Idea | Limitation |
|-----|--------|----------|-----------|
| **Pre-2013** | One-hot / Bag-of-Words | Count word occurrences | No semantic meaning |
| **2013** | Word2Vec, GloVe | Dense vectors from context | One vector per word (no context) |
| **2018** | ELMo | Context-dependent embeddings | Slow (bidirectional LSTM) |
| **2018+** | BERT, GPT (Transformers) | Deep contextual embeddings | Expensive to train |
| **2022+** | GPT-4, Claude (Modern LLMs) | Massive scale contextual | Requires large compute |

### The Context Problem

Word2Vec gives each word ONE vector, regardless of context:

```
"I went to the bank to deposit money"   → bank = financial institution
"I sat on the river bank"               → bank = edge of a river

Word2Vec: same vector for "bank" in both sentences!
Transformers: different vector for "bank" depending on context!
```

This limitation is exactly what Transformers solve — and why they were revolutionary.

---

## Embeddings in Practice

Modern embeddings convert text into dense vectors that capture meaning:

```python
from openai import OpenAI

client = OpenAI()

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Similar sentences → similar vectors
emb1 = get_embedding("The cat sat on the mat")
emb2 = get_embedding("A kitten rested on the rug")
emb3 = get_embedding("Stock prices rose sharply today")

# Cosine similarity
def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(cosine_sim(emb1, emb2))  # ~0.85 (very similar — both about animals on surfaces)
print(cosine_sim(emb1, emb3))  # ~0.15 (very different — cat vs stocks)
```

---

## Key Takeaways

- Tokenization converts text to processable units — modern LLMs use subword tokenization (BPE)
- Word representations evolved from one-hot (no meaning) to Word2Vec (fixed meaning) to Transformers (contextual meaning)
- Word2Vec showed that vector arithmetic can capture semantic relationships
- The key limitation of pre-Transformer approaches: one vector per word, regardless of context
- Modern embeddings (from Transformers) produce context-dependent vectors — the same word gets different vectors in different sentences

## Resources

- [YouTube: Word2Vec Explained](https://www.youtube.com/watch?v=viZrOnJclY0) -- Clear visual explanation of word embeddings
- [YouTube: Stanford CS224N — Word Vectors](https://www.youtube.com/watch?v=8rXD5-xhemo) -- Stanford NLP course lecture on word representations
- [Jay Alammar: The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) -- Visual guide to word embeddings
- [YouTube: Tokenization Explained](https://www.youtube.com/watch?v=zduSFxRajkE) -- How BPE tokenization works (by Andrej Karpathy)

---

Next: From Word2Vec to Contextual Embeddings
