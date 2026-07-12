---
title: Graph RAG and Knowledge Graphs
description: >-
  Combine graph-structured knowledge with RAG — entity extraction, community
  summaries, and the Microsoft GraphRAG architecture
duration: 60 min
difficulty: advanced
has_code: true
module: module-09
---

# Graph RAG and Knowledge Graphs

## Prerequisites

- [Advanced RAG Techniques](06-Advanced-RAG-Techniques.md) — parent-child, HyDE, query routing
- [Agentic RAG](09-Agentic-RAG.md) — multi-step retrieval loops
- Basic graph concepts (nodes, edges, adjacency) — no special graph theory required

## What You'll Learn

| Objective | Time | Difficulty |
|-----------|------|------------|
| Understand why flat vector retrieval fails on relationship-heavy questions | 10 min | Intermediate |
| Build an entity-extraction and graph-construction pipeline | 15 min | Advanced |
| Implement community detection and hierarchical summaries | 10 min | Advanced |
| Query a knowledge graph with vector-enhanced traversal | 15 min | Advanced |
| Know when Graph RAG is worth the extra complexity | 10 min | Intermediate |

---

## Intuition First

Imagine your RAG system answers questions about a 500-page biomedical report.
A user asks: *"What are all the interactions between Protein X, the MAPK pathway,
and the drugs mentioned in Section 4?"*

Standard vector RAG retrieves the top-K chunks most similar to the query. But
this question requires **connecting dots across many chunks** — Protein X is
mentioned in Chapter 2, MAPK in Chapter 5, Drug A in Section 4, and the
relationship between them is assembled from fragments scattered across dozens
of pages. No single chunk is a good match; the answer only exists in the *structure*
of connections.

This is the canonical failure mode of flat retrieval: **multi-hop, cross-entity
questions where the answer lives in relationships, not passages**.

**Graph RAG** (Edge & Wang et al., Microsoft Research, 2024) solves this by:
1. Extracting entities (Protein X, MAPK pathway, Drug A) and their relationships
   from all documents.
2. Building a knowledge graph where nodes = entities, edges = relationships.
3. Running community detection to find clusters of closely related entities.
4. Generating **community summaries** — prose descriptions of each cluster's
   key themes.
5. At query time, retrieving both graph-level summaries *and* vector-similar
   chunks, then fusing them.

The result: questions that require synthesizing information across an entire
corpus become answerable in a single pass.

---

## Core Theory

### 1. Knowledge Graphs: Structure and Terminology

A knowledge graph is a directed, labeled multigraph:

```
G = (V, E, L)

V = {entity_1, entity_2, ...}           # nodes
E ⊆ V × V                               # directed edges
L: E → label_set                        # edge labels (relation types)
```

Example triples (RDF-style: subject, predicate, object):

```
(Imatinib, inhibits, BCR-ABL1)
(BCR-ABL1, activates, MAPK pathway)
(Imatinib, treats, Chronic myeloid leukemia)
(MAPK pathway, regulates, cell proliferation)
```

**Entity types** in biomedical KGs: Drug, Gene, Protein, Disease, Pathway, Cell type.
**Relation types**: inhibits, activates, treats, binds, co-occurs with, is-a, part-of.

In GraphRAG, the graph is *extracted from unstructured text* rather than curated
by hand. This means entity types and relation labels emerge from LLM extraction,
making the graph noisy but comprehensive.

### 2. Entity and Relationship Extraction

The extraction step turns text chunks into graph triples using an LLM:

```
Input:  "Imatinib (Gleevec) is a BCR-ABL1 tyrosine kinase inhibitor used as
         first-line treatment for chronic myeloid leukemia (CML)."

Output:
  Entities:
    - Imatinib (type: Drug, also: Gleevec)
    - BCR-ABL1 (type: Protein/Kinase)
    - Chronic myeloid leukemia / CML (type: Disease)
  Relationships:
    - (Imatinib, INHIBITS, BCR-ABL1)
    - (Imatinib, TREATS, Chronic myeloid leukemia)
```

The extraction prompt is crucial. Microsoft's GraphRAG uses a multi-turn approach:
- **Pass 1**: Extract all entities, including type, description, and aliases.
- **Pass 2**: For each entity pair that co-occurs, extract the relationship type,
  description, and strength score.
- **Pass 3** (optional): Gleaning — ask "are there any entities you missed?"
  to improve recall.

Entity **disambiguation** (Imatinib = Gleevec = STI571) is handled by clustering
extracted entities with high embedding similarity into a single canonical node.

### 3. Community Detection: Leiden Algorithm

Once you have a graph with ~10,000 nodes and ~50,000 edges, it's too large to
summarize in one prompt. Community detection partitions the graph into clusters
of densely connected nodes.

**Leiden algorithm** (Traag et al., 2019) is used by Microsoft GraphRAG. It
optimizes **modularity** — a measure of how many edges fall within communities
vs. across them:

```
Q = (1/2m) × Σ_{i,j} [A_ij - k_i*k_j/(2m)] × δ(c_i, c_j)

where:
  m      = total edge weight
  A_ij   = edge weight between i and j
  k_i    = degree of node i
  c_i    = community of node i
  δ(a,b) = 1 if a==b, else 0
```

Leiden is a refinement of the Louvain algorithm — it guarantees that communities
are internally connected (Louvain can produce disconnected communities).

**Hierarchical communities**: Leiden can be run at multiple resolutions, producing
a hierarchy:
- Level 0: ~50 macro-communities (broad themes: "Cancer Biology", "Drug Discovery")
- Level 1: ~500 meso-communities (specific pathways, drug classes)
- Level 2: ~5,000 micro-communities (individual protein interactions)

This hierarchy is key for query routing — broad queries go to level-0 summaries,
specific queries go to level-2 summaries.

### 4. Community Summaries

For each community (at each level), an LLM generates a prose summary:

```
Community C47 (Level 1):
  Entities: Imatinib, Dasatinib, Nilotinib, BCR-ABL1, CML, resistance mutations
  Relationships: inhibition, treatment, resistance
  
LLM prompt: "Write a comprehensive summary of what these entities and 
  relationships represent in the context of this research corpus..."

Output:
  "This community centers on BCR-ABL1 tyrosine kinase inhibitors (TKIs) used
   in chronic myeloid leukemia (CML) treatment. Imatinib (first-generation),
   dasatinib and nilotinib (second-generation) all target BCR-ABL1. A key theme
   is resistance: T315I and other mutations in BCR-ABL1 reduce drug binding,
   motivating development of third-generation inhibitors like ponatinib..."
```

These summaries are **pre-computed** at index time — they're not generated
per-query. This is important: the summarization cost is paid once during
indexing, not repeated at query time.

### 5. Query-Time Retrieval: Map-Reduce Fusion

At query time, GraphRAG uses a **map-reduce** pattern:

```
Query: "How do resistance mutations affect treatment choices in CML?"

MAP phase:
  For each relevant community summary (selected by embedding similarity to query):
    Generate intermediate answer: "Based on this community: ..."
    Score relevance: 0.0–1.0

REDUCE phase:
  Combine intermediate answers, weighted by relevance score:
    "Resistance mutations in BCR-ABL1, particularly T315I (the 'gatekeeper'
     mutation), reduce binding affinity for first- and second-generation TKIs.
     This drives the clinical progression: start with imatinib, switch to
     dasatinib or nilotinib on failure, and use ponatinib for T315I carriers..."
```

The reduce step can incorporate **local retrieval** (exact chunk matches from
vector search) alongside global community summaries, blending specific evidence
with broad context.

### 6. GraphRAG vs. Alternatives

| Approach | Best for | Limitation |
|----------|----------|------------|
| Flat vector RAG | Specific factual lookup | Fails on multi-hop, cross-entity |
| Parent-child RAG | Maintaining context within one document | Doesn't span documents |
| HyDE | Sparse relevant passages | Still limited to chunk-level retrieval |
| **GraphRAG** | Global synthesis, relationship queries | High indexing cost, complex setup |
| NL2Cypher | Structured KGs with clean schema | Requires pre-existing ontology |
| KGRAG (hybrid) | Both factual and relational queries | Implementation complexity |

!!! note "GraphRAG is expensive to index"
    Indexing 1M tokens with GraphRAG requires ~10–40× more LLM calls than building
    a standard vector index. For small corpora or simple queries, it's not worth it.

---

## Worked Example: Building a Mini Knowledge Graph

We'll build a simplified GraphRAG pipeline for a small corpus.

### Step 1: Corpus

```python
corpus = [
    "BERT (Bidirectional Encoder Representations from Transformers) was introduced "
    "by Devlin et al. at Google in 2018. It uses masked language modeling (MLM) "
    "and next sentence prediction (NSP) for pre-training.",

    "GPT-3, developed by OpenAI, is an autoregressive language model with 175B "
    "parameters. It demonstrated few-shot learning capabilities without fine-tuning.",

    "The Transformer architecture, introduced by Vaswani et al. in 'Attention Is "
    "All You Need' (2017), uses self-attention mechanisms to process sequences "
    "in parallel. Both BERT and GPT-3 are based on Transformers.",

    "InstructGPT extended GPT-3 using RLHF (Reinforcement Learning from Human "
    "Feedback). It used supervised fine-tuning followed by PPO-based reward "
    "optimization. GPT-4 and later GPT models build on InstructGPT's approach.",
]
```

### Step 2: Entity Extraction

```python
import json
import openai
from typing import NamedTuple

client = openai.OpenAI()

class Entity(NamedTuple):
    name: str
    entity_type: str
    description: str

class Relationship(NamedTuple):
    source: str
    relation: str
    target: str
    description: str

EXTRACTION_PROMPT = """
Extract all named entities and relationships from the following text.

Return JSON with this exact structure:
{
  "entities": [
    {"name": "...", "type": "...", "description": "..."}
  ],
  "relationships": [
    {"source": "...", "relation": "...", "target": "...", "description": "..."}
  ]
}

Entity types: Model, Organization, Person, Technique, Paper, Concept
Relation types: DEVELOPED_BY, BASED_ON, USES, INTRODUCED_IN, EXTENDS, TRAINED_WITH

Text:
{text}
"""

def extract_graph_elements(text: str) -> tuple[list[Entity], list[Relationship]]:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": EXTRACTION_PROMPT.format(text=text)
        }],
        response_format={"type": "json_object"},
        temperature=0,
    )
    data = json.loads(response.choices[0].message.content)
    
    entities = [
        Entity(e["name"], e["type"], e["description"])
        for e in data.get("entities", [])
    ]
    relationships = [
        Relationship(r["source"], r["relation"], r["target"], r["description"])
        for r in data.get("relationships", [])
    ]
    return entities, relationships


# Run extraction on all chunks
all_entities: list[Entity] = []
all_relationships: list[Relationship] = []

for chunk in corpus:
    ents, rels = extract_graph_elements(chunk)
    all_entities.extend(ents)
    all_relationships.extend(rels)

print(f"Extracted {len(all_entities)} entities, {len(all_relationships)} relationships")
```

### Step 3: Build Graph with NetworkX

```python
import networkx as nx
from collections import defaultdict

def build_knowledge_graph(
    entities: list[Entity],
    relationships: list[Relationship],
) -> nx.Graph:
    G = nx.Graph()

    # Normalize entity names (simple: lowercase + strip)
    def normalize(name: str) -> str:
        return name.strip().lower()

    # Add nodes
    entity_map: dict[str, Entity] = {}
    for e in entities:
        key = normalize(e.name)
        if key not in entity_map:  # take first occurrence
            entity_map[key] = e
            G.add_node(key, **{
                "original_name": e.name,
                "type": e.entity_type,
                "description": e.description,
                "weight": 1,
            })
        else:
            # Increment weight for co-mentioned entities
            G.nodes[key]["weight"] += 1

    # Add edges
    for r in relationships:
        src = normalize(r.source)
        tgt = normalize(r.target)
        if src in G and tgt in G:
            if G.has_edge(src, tgt):
                G[src][tgt]["weight"] += 1
                G[src][tgt]["relations"].append(r.relation)
            else:
                G.add_edge(src, tgt, weight=1, relations=[r.relation],
                           description=r.description)

    return G

G = build_knowledge_graph(all_entities, all_relationships)
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Inspect neighbors of a node
node = "bert"
if node in G:
    print(f"\nNeighbors of '{node}':")
    for neighbor in G.neighbors(node):
        edge_data = G[node][neighbor]
        print(f"  → {neighbor} ({', '.join(edge_data['relations'])})")
```

### Step 4: Community Detection

```python
import community as community_louvain  # pip install python-louvain

# Detect communities (Louvain; use graspologic for Leiden)
partition = community_louvain.best_partition(G, weight="weight")

# Group nodes by community
communities: dict[int, list[str]] = defaultdict(list)
for node, comm_id in partition.items():
    communities[comm_id].append(node)

print(f"\n{len(communities)} communities detected:")
for comm_id, members in sorted(communities.items()):
    print(f"  Community {comm_id}: {', '.join(members)}")
```

### Step 5: Generate Community Summaries

```python
COMMUNITY_SUMMARY_PROMPT = """
You are analyzing a knowledge graph community from an AI/ML research corpus.

Community members (entities): {entities}
Key relationships: {relationships}

Write a comprehensive summary (3-5 sentences) explaining:
1. What this cluster of concepts represents
2. The key relationships between them
3. Why they form a coherent group
"""

def summarize_community(
    comm_id: int,
    members: list[str],
    G: nx.Graph,
) -> str:
    # Collect entity descriptions
    entity_descs = []
    for node in members:
        data = G.nodes[node]
        entity_descs.append(
            f"- {data['original_name']} ({data['type']}): {data['description']}"
        )

    # Collect internal relationships
    rel_descs = []
    for i, u in enumerate(members):
        for v in members[i+1:]:
            if G.has_edge(u, v):
                edge = G[u][v]
                rel_descs.append(
                    f"- {G.nodes[u]['original_name']} {', '.join(edge['relations'])} "
                    f"{G.nodes[v]['original_name']}"
                )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": COMMUNITY_SUMMARY_PROMPT.format(
                entities="\n".join(entity_descs),
                relationships="\n".join(rel_descs) or "No direct edges between members.",
            )
        }],
        max_tokens=300,
        temperature=0,
    )
    return response.choices[0].message.content


community_summaries = {}
for comm_id, members in communities.items():
    if len(members) >= 2:  # skip singleton communities
        summary = summarize_community(comm_id, members, G)
        community_summaries[comm_id] = {
            "members": members,
            "summary": summary,
        }

print(f"\nGenerated {len(community_summaries)} community summaries")
```

### Step 6: Query-Time Map-Reduce

```python
import numpy as np
from openai import OpenAI

def embed(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


MAP_PROMPT = """
Given this community summary from an AI research knowledge graph,
answer the following question as specifically as possible based ONLY
on the information in the summary. If the summary is not relevant,
respond with "NOT RELEVANT".

Question: {query}

Community Summary:
{summary}

Answer:"""

REDUCE_PROMPT = """
You are synthesizing multiple partial answers to form a comprehensive response.

Question: {query}

Partial answers from different knowledge graph communities:
{partial_answers}

Write a comprehensive, coherent answer that integrates all relevant information.
Remove redundancy and organize by theme."""


def graph_rag_query(query: str, top_k_communities: int = 3) -> str:
    query_embedding = embed(query)

    # Score communities by summary embedding similarity
    scored = []
    for comm_id, data in community_summaries.items():
        summary_embedding = embed(data["summary"])
        score = cosine_similarity(query_embedding, summary_embedding)
        scored.append((score, comm_id, data["summary"]))

    scored.sort(reverse=True)
    top_communities = scored[:top_k_communities]

    # MAP: generate partial answers from each community
    partial_answers = []
    for score, comm_id, summary in top_communities:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": MAP_PROMPT.format(query=query, summary=summary)
            }],
            max_tokens=200,
            temperature=0,
        )
        answer = response.choices[0].message.content
        if "NOT RELEVANT" not in answer:
            partial_answers.append(f"[Community {comm_id}, score={score:.2f}]\n{answer}")

    if not partial_answers:
        return "No relevant information found in the knowledge graph."

    # REDUCE: synthesize partial answers
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": REDUCE_PROMPT.format(
                query=query,
                partial_answers="\n\n---\n\n".join(partial_answers),
            )
        }],
        max_tokens=600,
        temperature=0,
    )
    return response.choices[0].message.content


# Test the pipeline
answer = graph_rag_query(
    "How do BERT and GPT-3 relate to the original Transformer architecture?"
)
print(answer)
```

---

## Microsoft GraphRAG: Production Architecture

The open-source [graphrag](https://github.com/microsoft/graphrag) library
(Apache 2.0) implements the full pipeline described in the paper. Key config:

```yaml
# graphrag/settings.yml (simplified)
encoding_model: cl100k_base
skip_workflows: []

llm:
  api_key: ${GRAPHRAG_API_KEY}
  type: openai_chat
  model: gpt-4o-mini
  model_supports_json: true
  max_tokens: 4000

embeddings:
  async_mode: threaded
  llm:
    api_key: ${GRAPHRAG_API_KEY}
    type: openai_embedding
    model: text-embedding-3-small

chunks:
  size: 300
  overlap: 100
  group_by_columns: [id]

entity_extraction:
  prompt: prompts/entity_extraction.txt
  entity_types: [organization, person, geo, event]
  max_gleanings: 1

community_reports:
  prompt: prompts/community_report.txt
  max_length: 2000
  max_input_length: 8000
```

```bash
# Index pipeline (run once per corpus)
python -m graphrag.index --root ./my-project

# Query modes
python -m graphrag.query \
  --root ./my-project \
  --method global \          # uses community summaries
  "What are the main themes across all documents?"

python -m graphrag.query \
  --root ./my-project \
  --method local \           # uses entity-specific context
  "What does BERT use for pre-training?"
```

!!! warning "Indexing cost"
    Indexing a 1M token corpus with default settings costs ~$20–50 in OpenAI API
    calls. Use `gpt-4o-mini` for entity extraction to reduce costs. The `--model`
    flag and settings.yml both accept cheaper models.

---

## Edge Cases & Misconceptions

### "GraphRAG is just fancy vector RAG"

No — the fundamental difference is that GraphRAG retrieves **community summaries**
(pre-computed global context) rather than individual chunks. The graph structure
enables answering questions whose answers span the entire corpus, not just a
few nearby passages.

### "Extracted graphs are as reliable as curated ontologies"

LLM extraction is noisy. Expect 10–20% of extracted relationships to be hallucinated
or miscategorized. For high-stakes applications (medical, legal), add a validation
step:

```python
def validate_relationship(triple: Relationship, source_text: str) -> bool:
    """Verify the triple is grounded in the source text."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                f"Does this text explicitly support the claim that "
                f"'{triple.source} {triple.relation} {triple.target}'?\n\n"
                f"Text: {source_text}\n\nAnswer yes or no."
            )
        }],
        max_tokens=5,
    )
    return response.choices[0].message.content.lower().startswith("yes")
```

### "More graph hops = better answers"

Multi-hop traversal (A → B → C → D) accumulates errors at each hop. Limit
traversal depth to 2–3 hops; beyond that, precision degrades and hallucination
increases. Use community summaries for global questions instead of long hop chains.

### "GraphRAG replaces vector RAG"

Use them together. Vector RAG handles precise factual lookups ("What year was
BERT published?"). GraphRAG handles synthesis questions ("How did Transformer-based
models evolve over time?"). The `local` query mode in Microsoft's implementation
already blends both.

!!! note "KGQA vs. GraphRAG"
    GraphRAG is different from traditional Knowledge Graph Question Answering
    (KGQA / NL2Cypher). KGQA assumes a pre-existing, curated KG with a formal
    schema; it translates NL questions to graph queries (SPARQL, Cypher).
    GraphRAG extracts the graph from unstructured text and uses LLM-based
    reasoning over community summaries, not formal graph queries.

---

## Production Connection

### When to choose GraphRAG

| Signal | Recommendation |
|--------|---------------|
| Users ask cross-document synthesis questions | GraphRAG — designed for this |
| Corpus has dense entity relationships (biomedical, legal, financial) | GraphRAG |
| Corpus < 50,000 tokens | Standard vector RAG is sufficient |
| Real-time indexing required (new docs every minute) | GraphRAG indexing is slow — use vector RAG with triggers |
| Budget < $50/month for indexing | Consider cost carefully |
| Queries are always specific and factual | Local mode or standard RAG |

### Incremental indexing

Microsoft GraphRAG 0.4+ supports incremental indexing — add new documents without
rebuilding the full graph:

```python
from graphrag.index import run_pipeline_with_config
from graphrag.config import create_graphrag_config

# Only index new documents since last run
config = create_graphrag_config(root_dir="./my-project")
result = await run_pipeline_with_config(
    config,
    run_id="incremental-2025-07",
    is_update_run=True,  # incremental mode
)
```

### Monitoring extraction quality

Track entity and relationship extraction quality with a small eval set:

```python
def extraction_precision_recall(
    predicted: list[Relationship],
    gold: list[Relationship],
) -> dict:
    pred_set = {(r.source.lower(), r.relation, r.target.lower()) for r in predicted}
    gold_set = {(r.source.lower(), r.relation, r.target.lower()) for r in gold}
    
    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0
    recall = tp / len(gold_set) if gold_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    
    return {"precision": precision, "recall": recall, "f1": f1}
```

---

## Key Takeaways

- **Flat vector RAG fails on multi-hop, cross-entity questions** — the answer
  lives in relationships between passages, not within any single passage.
- **Knowledge graph extraction** turns unstructured text into structured triples
  (entity, relation, entity) via LLM prompting — expect noise and plan for it.
- **Community detection** (Leiden/Louvain) partitions the graph into thematic
  clusters; hierarchical communities support both broad and narrow queries.
- **Community summaries** are the key innovation of GraphRAG: pre-computed prose
  descriptions of each cluster that enable global query answering without scanning
  the entire corpus.
- **Map-reduce at query time**: each relevant community generates a partial answer
  (MAP), then a final LLM call synthesizes them (REDUCE).
- **Microsoft GraphRAG** (open-source) provides a production-ready implementation;
  use `global` mode for synthesis, `local` mode for entity-specific retrieval.
- **Cost is real**: indexing is 10–40× more expensive than building a vector index;
  evaluate whether your query profile justifies it.
- **GraphRAG and vector RAG are complementary** — the best systems blend both,
  using graph for global context and vectors for specific evidence.

---

## Further Reading

- [From Local to Global: A Graph RAG Approach](https://arxiv.org/abs/2404.16130) — Edge, Wang et al. (2024), the original GraphRAG paper
- [microsoft/graphrag](https://github.com/microsoft/graphrag) — official open-source implementation
- [HippoRAG](https://arxiv.org/abs/2405.14831) — graph-based RAG inspired by human memory indexing
- [LightRAG](https://arxiv.org/abs/2410.05779) — simpler graph-enhanced RAG with dual retrieval
- [REBEL](https://arxiv.org/abs/2104.07650) — end-to-end relation extraction model for graph building
- [Leiden Algorithm paper](https://www.nature.com/articles/s41598-019-41695-z) — Traag, Waltman, van Eck (2019)
- [Knowledge Graphs + LLMs survey](https://arxiv.org/abs/2306.08302) — Pan et al., comprehensive review

---

## Next Lesson

You've completed the RAG module. Continue to:

→ [Module 11: AI Agents Fundamentals](../../module-11-ai-agents-fundamentals/index.md)

Or explore the related deep dive:

→ [Attention Math: Full QKV Derivation](../../../deep-dives/attention-math.md)
