---
title: Chunking Strategies
description: Learn how to split documents effectively for optimal RAG performance
duration: 30 min
difficulty: intermediate
has_code: false
module: module-09
youtube: 'https://www.youtube.com/watch?v=8OJC21T2SL4'
---
# Chunking Strategies

## Why Chunk?

**Problem**: Documents are too long for embeddings

```
Max context for embedding models: 512-8K tokens
Your document: 50K tokens

Solution: Split into chunks!
```

## Fixed-Size Chunking

**Simplest approach**: Split every N characters/tokens

```python
def chunk_fixed_size(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # Overlap prevents cutting mid-sentence
    return chunks

text = "Long document..." * 1000
chunks = chunk_fixed_size(text, chunk_size=500, overlap=50)
```

**Pros**: Simple, consistent size
**Cons**: May split mid-sentence/paragraph

## Sentence-Based Chunking

**Better**: Respect sentence boundaries

```python
import re

def chunk_by_sentences(text, max_sentences=5):
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    current = []
    
    for sentence in sentences:
        current.append(sentence)
        if len(current) >= max_sentences:
            chunks.append(' '.join(current))
            current = []
    
    if current:
        chunks.append(' '.join(current))
    
    return chunks
```

## Semantic Chunking

**Advanced**: Split where topic changes

```python
from langchain.text_splitter import SemanticChunker

chunker = SemanticChunker(
    embeddings=OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile"  # Split at semantic boundaries
)

chunks = chunker.split_text(long_document)
```

## Recursive Chunking

**LangChain approach**: Try multiple separators

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["

", "
", ". ", " ", ""]  # Try in order
)

chunks = splitter.split_text(document)
```

## Chunking Best Practices

1. **Overlap**: 10-20% overlap between chunks
2. **Size**: 200-1000 tokens per chunk
3. **Boundaries**: Respect paragraphs/sections
4. **Metadata**: Keep source info with each chunk
5. **Test**: Try different strategies, measure retrieval quality

## Metadata Enrichment

```python
def chunk_with_metadata(document, doc_id):
    chunks = []
    sections = document.split('

')
    
    for i, section in enumerate(sections):
        chunk = {
            'text': section,
            'doc_id': doc_id,
            'chunk_id': i,
            'doc_title': document.split('
')[0],  # First line
            'length': len(section)
        }
        chunks.append(chunk)
    
    return chunks
```

---

## 📹 Recommended Videos

- [RAG Chunking Strategies](https://www.youtube.com/watch?v=8OJC21T2SL4) — How different chunking approaches affect RAG quality
- [Document Chunking for RAG](https://www.youtube.com/watch?v=eqOfr4AGLk8) — Practical chunking tutorial

---

## 📚 Additional Resources

- [Chunking Strategies Guide](https://www.pinecone.io/learn/chunking-strategies/) — Pinecone's comprehensive guide
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/) — Text splitting implementations
- [Unstructured.io](https://unstructured.io/) — Document parsing and chunking library
