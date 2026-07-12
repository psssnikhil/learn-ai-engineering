---
title: "Tokenization Internals: BPE from Scratch"
description: >-
  How Byte-Pair Encoding builds a vocabulary from a raw corpus — merge rules,
  a complete worked example, and why token boundaries matter for engineering
---

# Tokenization Internals: BPE

**Prerequisite**: [Tokenization Deep Dive](../foundations/module-07-large-language-models-llms/lessons/04-tokenization.md)

**What you'll get**: After this page you can implement BPE tokenization from
scratch, explain every merge decision, and reason about engineering edge cases
like token boundary mismatches and vocabulary size trade-offs.

---

## Intuition First

Language models don't read characters or words — they read **tokens**. Tokens
are the atomic units the model processes, and they come from a vocabulary that's
fixed at training time.

The challenge: English has ~170,000 words. A word-level vocabulary misses typos,
rare words, foreign words, code identifiers, and any word the model wasn't trained
on (the **out-of-vocabulary** problem). Character-level vocabularies (26 letters
+ punctuation) handle everything but create enormously long sequences where
learning long-range dependencies becomes difficult.

**Byte-Pair Encoding** (BPE) finds a middle ground: start with individual
characters (or bytes), and *iteratively merge the most frequent adjacent pairs*
until you have a vocabulary of the desired size. Frequent subwords become single
tokens; rare words decompose into familiar pieces.

The name comes from data compression: BPE was originally an algorithm for
replacing the most common byte pair in data with a new symbol to compress files.
Sennrich et al. (2016) adapted it for NLP.

---

## Algorithm

### Training-time BPE (building the vocabulary)

```
Input:  Raw corpus
        Target vocabulary size V (e.g., 50,000)

Step 1: Pre-tokenize
        Split corpus into words using whitespace/punctuation rules.
        Append a special end-of-word marker (e.g., </w>) to each word.
        Represent each word as a sequence of its characters.

Step 2: Count word frequencies
        Count how many times each character-level word appears.

Step 3: Build initial vocabulary
        Start with all unique characters (+ </w>) as the initial vocab.

Step 4: Repeat until vocab size = V:
        a. Count frequency of all adjacent symbol pairs across the corpus
           (weighted by word frequency).
        b. Find the most frequent pair.
        c. Merge that pair into a new symbol.
        d. Add the new symbol to the vocab.
        e. Update all word representations by applying the merge.

Output: List of merge rules (in order) + final vocabulary
```

### Encoding at inference time

```
Input:  New text, list of merge rules in training order

Step 1: Pre-tokenize (same rules as training)
Step 2: Represent each word as characters + </w>
Step 3: Apply merge rules in order (left to right, top priority first)
        Stop when no more merges apply.
Step 4: The resulting symbols are the tokens.
```

The merge rules are **deterministic and ordered** — the 1st learned merge always
takes priority over the 2nd, and so on. This is why the same text always tokenizes
identically (unlike some probabilistic tokenizers).

---

## Worked Example: BPE by Hand

We'll train BPE on a tiny "corpus" of 5 words.

### Setup

```python
from collections import Counter, defaultdict

# Corpus: word → frequency
# </w> marks end-of-word (allows the model to distinguish "low" in "lower" vs standalone "low")
corpus = {
    "l o w </w>": 5,
    "l o w e r </w>": 2,
    "n e w e s t </w>": 6,
    "w i d e s t </w>": 3,
    "n e w </w>": 2,
}

print("Initial corpus (character-level):")
for word, freq in corpus.items():
    print(f"  {freq}× '{word}'")
```

**Output:**
```
Initial corpus (character-level):
  5× 'l o w </w>'
  2× 'l o w e r </w>'
  6× 'n e w e s t </w>'
  3× 'w i d e s t </w>'
  2× 'n e w </w>'
```

### Count adjacent pair frequencies

```python
def get_pair_counts(corpus: dict[str, int]) -> Counter:
    counts = Counter()
    for word, freq in corpus.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pair = (symbols[i], symbols[i+1])
            counts[pair] += freq
    return counts

pair_counts = get_pair_counts(corpus)
print("\nTop 5 most frequent pairs:")
for pair, count in pair_counts.most_common(5):
    print(f"  {pair[0]} + {pair[1]}: {count}")
```

**Output:**
```
Top 5 most frequent pairs:
  e + s: 9       ← 6 (newest) + 3 (widest)
  s + t: 9       ← 6 (newest) + 3 (widest)
  e + w: 8       ← 6 (newest) + 2 (newer)
  l + o: 7       ← 5 (low) + 2 (lower)
  o + w: 7       ← 5 (low) + 2 (lower)
```

### Apply the best merge: `e + s` → `es`

```python
def apply_merge(corpus: dict[str, int], pair: tuple[str, str]) -> dict[str, int]:
    new_corpus = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for word, freq in corpus.items():
        new_word = word.replace(bigram, replacement)
        new_corpus[new_word] = freq
    return new_corpus

best_pair = pair_counts.most_common(1)[0][0]
print(f"\nMerge 1: {best_pair[0]} + {best_pair[1]} → {''.join(best_pair)}")
corpus = apply_merge(corpus, best_pair)

print("\nCorpus after merge 1:")
for word, freq in corpus.items():
    print(f"  {freq}× '{word}'")
```

**Output:**
```
Merge 1: e + s → es

Corpus after merge 1:
  5× 'l o w </w>'
  2× 'l o w e r </w>'
  6× 'n e w es t </w>'
  3× 'w i d es t </w>'
  2× 'n e w </w>'
```

### Full BPE training loop

```python
def train_bpe(corpus: dict[str, int], num_merges: int) -> list[tuple[str, str]]:
    merge_rules = []
    
    for i in range(num_merges):
        pair_counts = get_pair_counts(corpus)
        if not pair_counts:
            break
        
        best_pair = pair_counts.most_common(1)[0][0]
        merge_rules.append(best_pair)
        corpus = apply_merge(corpus, best_pair)
        
        merged = "".join(best_pair)
        pair_count = pair_counts[best_pair]
        print(f"Merge {i+1:2d}: '{best_pair[0]}' + '{best_pair[1]}' "
              f"→ '{merged}' (freq={pair_count})")
    
    print("\nFinal vocabulary:")
    vocab = set()
    for word in corpus:
        vocab.update(word.split())
    for tok in sorted(vocab):
        print(f"  '{tok}'")
    
    return merge_rules

corpus_reset = {
    "l o w </w>": 5,
    "l o w e r </w>": 2,
    "n e w e s t </w>": 6,
    "w i d e s t </w>": 3,
    "n e w </w>": 2,
}

merge_rules = train_bpe(corpus_reset, num_merges=10)
```

**Expected output:**
```
Merge  1: 'e' + 's' → 'es' (freq=9)
Merge  2: 'es' + 't' → 'est' (freq=9)
Merge  3: 'l' + 'o' → 'lo' (freq=7)
Merge  4: 'lo' + 'w' → 'low' (freq=7)
Merge  5: 'n' + 'e' → 'ne' (freq=8)
Merge  6: 'ne' + 'w' → 'new' (freq=8)
Merge  7: 'new' + 'est' → 'newest' (freq=6)
Merge  8: 'w' + 'i' → 'wi'  (freq=3)
Merge  9: 'wi' + 'd' → 'wid' (freq=3)
Merge 10: 'wid' + 'est' → 'widest' (freq=3)

Final vocabulary:
  '</w>' 'e' 'er' 'i' 'l' 'lo' 'low' 'n' 'ne' 'new' 'newest' 'o' 'r' ...
```

Notice what happened:
- "newest" and "widest" became single tokens after just 7 merges — they were frequent enough.
- "lower" decomposed as `low` + `e` + `r` + `</w>` — "lower" was too rare to get its own token.
- Common subwords (`low`, `new`, `est`) emerged as reusable tokens.

### Inference: encoding new text

```python
def encode(word: str, merge_rules: list[tuple[str, str]]) -> list[str]:
    """Encode a single word using the trained merge rules."""
    # Start as individual characters + end-of-word marker
    symbols = list(word) + ["</w>"]
    
    for pair in merge_rules:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == pair[0] and symbols[i+1] == pair[1]:
                symbols = symbols[:i] + ["".join(pair)] + symbols[i+2:]
            else:
                i += 1
    
    return symbols

# Test on known and unknown words
test_words = ["low", "lowest", "newer", "unknown"]
print("Encoding test words:")
for word in test_words:
    tokens = encode(word, merge_rules)
    print(f"  '{word}' → {tokens}")
```

**Expected:**
```
Encoding test words:
  'low'     → ['low', '</w>']
  'lowest'  → ['low', 'est', '</w>']         ← reuses 'low' and 'est'!
  'newer'   → ['new', 'e', 'r', '</w>']      ← 'newer' not in vocab, decomposes
  'unknown' → ['u', 'n', 'k', 'n', 'o', 'w', 'n', '</w>']  ← character-level fallback
```

This is BPE's key property: **graceful degradation**. Words in the training corpus
get efficient token representations; rare or unseen words fall back to character
pieces. Nothing is truly out-of-vocabulary (for byte-level BPE, even the
character-level fallback is guaranteed via the 256 raw bytes).

---

## Byte-Level BPE (GPT-2 and beyond)

Standard BPE operates on Unicode characters. GPT-2 introduced **byte-level BPE**:

1. Encode the input string as UTF-8 bytes (256 possible byte values).
2. Apply BPE on bytes, not characters.

Benefits:
- **Zero unknown tokens**: any input can be represented as bytes.
- **Cross-lingual**: handles any language/script without language-specific pre-tokenization.
- GPT-2's vocabulary: 256 base byte tokens + 50,000 merge rules = 50,257 tokens total.

```python
# GPT-2 base vocabulary: bytes as strings (not raw bytes)
base_vocab = {bytes([i]).decode("latin-1"): i for i in range(256)}

# OpenAI maps bytes to visible characters to avoid control chars in the vocab
# This is why GPT tokenizer shows "Ġ" (U+0120) for space-prefixed tokens
BYTE_ENCODER = {
    b: chr(i + 256) if (b < 33 or b == 127 or 128 <= b < 161) else chr(b)
    for i, b in enumerate(
        [b for b in range(256) if not (33 <= b <= 126 or 161 <= b <= 172 or 174 <= b <= 255)]
        + [b for b in range(256) if (33 <= b <= 126 or 161 <= b <= 172 or 174 <= b <= 255)]
    )
}
```

The `Ġ` character you see in GPT tokenizer outputs represents a space byte (0x20),
remapped to a visible Unicode character so the vocabulary file is human-readable.

---

## The Pre-Tokenization Step

Before BPE runs, the text is split by a **pre-tokenizer** that handles whitespace
and punctuation. GPT-2 uses a regex-based pre-tokenizer:

```python
import regex as re

# GPT-2's pre-tokenization regex
PRETOKENIZE_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

def pretokenize(text: str) -> list[str]:
    return re.findall(PRETOKENIZE_PATTERN, text)

examples = [
    "Hello, world!",
    "I've been here.",
    "tokenization isn't trivial",
    "GPT-4 costs $0.03/1K tokens",
]

for ex in examples:
    pieces = pretokenize(ex)
    print(f"'{ex}'")
    print(f"  → {pieces}\n")
```

**Output:**
```
'Hello, world!'
  → ['Hello', ',', ' world', '!']

"I've been here."
  → ['I', "'ve", ' been', ' here', '.']

'tokenization isn't trivial'
  → ['tokenization', ' isn', "'t", ' trivial']

'GPT-4 costs $0.03/1K tokens'
  → ['GPT', '-', '4', ' costs', ' $', '0', '.', '03', '/', '1', 'K', ' tokens']
```

The pre-tokenizer ensures:
- Contractions ("I've") split correctly (BPE won't merge across them)
- Numbers stay separate from words
- Leading spaces are attached to the *following* word (hence " world" not "world")
  — this is why tokens like ` the` (space + "the") are different from `the`

---

## Why Token Boundaries Matter for Engineering

### 1. Context window is in tokens, not characters

GPT-4's 128K context window means 128,000 tokens, not characters. A single
Unicode character from a language with multi-byte representations (e.g., Chinese)
may use 1–3 tokens. A single Python identifier `__init__` tokenizes as:
`['__', 'init', '__']` — 3 tokens for one identifier.

```python
# Estimate token count without API call
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

texts = [
    "Hello, world!",                       # English: ~3 tokens
    "你好，世界！",                         # Chinese: ~6-8 tokens  
    "def __init__(self, param: int):",      # Python: ~12-14 tokens
    "1 2 3 4 5 6 7 8 9 10",               # Numbers: ~20 tokens (each a token!)
]

for text in texts:
    tokens = enc.encode(text)
    print(f"{len(tokens):3d} tokens | char/token={len(text)/len(tokens):.1f} | '{text[:40]}'")
```

### 2. Prompt injection via token splitting

An attacker can craft inputs where a harmful word is split across a token boundary
that a keyword filter might miss:

```
"Ignore previous instructions" as characters: I g n o r e ...
Tokenized:                                   ['Ignore', ' previous', ' instructions']
```

But with subtle Unicode:
```
"Ign\u200bore" (zero-width space inserted)
Tokenized: ['Ign', '\u200b', 'ore']
```

The string looks like "Ignore" to humans but tokenizes differently, potentially
bypassing naive string matching filters.

### 3. Off-by-one in structured output

When using structured generation (constrained decoding), you constrain generation
to a specific token sequence. Token boundary mismatches cause silent errors:

```python
# What you expect:
target = '{"status": "ok"}'
tokens = enc.encode(target)
# [123, 29280, 2065, 794, 330, 564, 1, 15]  ← depends on BPE

# If you constrain to token 330 ('ok'), but the model wants to produce "okay",
# the constraint fails silently — it can't produce 'ok' as a standalone token
# in a context where BPE would merge it differently.
```

### 4. Few-shot example alignment

LLMs often perform better when few-shot examples are token-aligned — meaning
example answers start and end at token boundaries. Misalignment causes the model
to "start mid-token" which can degrade output quality for structured tasks.

---

## Vocabulary Size Trade-offs

| Vocab size | Tokens per English word (avg) | Sequence length | Memory | Cross-lingual |
|------------|-------------------------------|-----------------|--------|---------------|
| 1,000 | ~5–8 | Very long | Low | Poor |
| 10,000 | ~2–3 | Long | Medium | Moderate |
| **50,000** | **~1.3** | **Medium** | **Medium** | **Good** |
| 100,000 | ~1.1 | Short | High | Better |
| 500,000+ | ~1.0 | Shortest | Very high | Best |

GPT-2/3/4 use ~50,000–100,000. LLaMA 2 uses 32,000 (smaller vocab → longer
sequences → harder for the model but cheaper in embedding memory). LLaMA 3
upgraded to 128,000 tokens to improve multilingual and code coverage.

!!! note "Why not bigger vocabularies?"
    The embedding table is `(vocab_size × d_model)`. At vocab=128K and d_model=4096,
    that's 128K × 4096 × 2 bytes = 1GB just for token embeddings. Larger
    vocabularies also require more training data for each token to appear enough
    times to learn a good embedding.

---

## Key Takeaways

- **BPE** iteratively merges the most frequent adjacent symbol pair until a target
  vocabulary size is reached. The result is a hierarchy of subword tokens.
- **Merge rules are ordered and deterministic** — applying them in training order
  reproduces the exact tokenization seen during training.
- **Byte-level BPE** (GPT-2+) starts from 256 raw bytes, guaranteeing zero
  unknown tokens for any input.
- **Pre-tokenization** (regex splitting on whitespace/punctuation) runs before BPE,
  ensuring contractions and words don't merge across natural boundaries.
- **Token count ≠ character count** — Chinese text, code, and numbers use
  significantly more tokens per surface character than English prose.
- **Token boundary mismatches** cause subtle bugs in structured generation,
  keyword filtering, and few-shot alignment.
- **Vocabulary size** trades off sequence length, embedding memory, and rare-word
  coverage — 50K–100K is the current engineering sweet spot for English-centric models.

---

## Further Reading

- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — Sennrich et al. (2016), original BPE for NLP
- [tiktokenizer](https://tiktokenizer.vercel.app/) — interactive tokenizer visualization
- [tiktoken](https://github.com/openai/tiktoken) — OpenAI's production BPE library (Rust + Python)
- [SentencePiece](https://github.com/google/sentencepiece) — Google's tokenizer (used by LLaMA, T5, Gemini); implements BPE and unigram LM
- [Tokenization is NLP's original sin](https://medium.com/@l.chartier/tokenization-is-nlps-original-sin-1c7153fa28d4) — engineering perspective on token artifacts
- [Let's Build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) — Andrej Karpathy builds BPE from scratch (2h video)

← [Deep Dives Hub](index.md)
