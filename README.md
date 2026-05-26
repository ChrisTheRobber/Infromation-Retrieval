```
Information Retrieval System
============================

A three-phase information retrieval system built for the IR2025 course.
Implements and compares three retrieval approaches — BM25 baseline, dense
semantic search, and a hybrid re-ranking model — evaluated in TREC format.

Phases
------

Phase 1 — BM25 Baseline (Elasticsearch)
    Indexes documents into Elasticsearch with an English analyzer (stemming
    and stopword removal) and BM25 similarity. Retrieves the top-k documents
    per query and outputs results in TREC format.

Phase 2 — Semantic Retrieval (Transformers + FAISS)
    Encodes all documents using the all-MiniLM-L6-v2 sentence transformer
    (384 dimensions). Builds a FAISS flat index with cosine similarity via
    L2 normalization and inner product. Embeddings are cached to disk after
    the first run to avoid recomputation.

Phase 3 — Hybrid Model (BM25 + Transformer Re-ranking)
    Uses Elasticsearch (Phase 1 index) to retrieve the top 200 candidate
    documents per query, then re-ranks them using transformer embeddings
    and FAISS cosine similarity. Combines lexical recall with semantic
    precision.

Requirements
------------
    pip install pandas numpy faiss-cpu sentence-transformers elasticsearch

Elasticsearch must be running locally on port 9200.
Phase 3 requires the index created in Phase 1 to already exist.

Input Files
-----------
    documents.csv    Columns: ID, Text
    queries.csv      Columns: ID, Text

Usage
-----
    python phase1_solution.py
    python phase2_solution.py
    python phase3_solution.py

Each phase generates three output files:
    results_k20.txt
    results_k30.txt
    results_k50.txt

Output Format
-------------
Results are written in TREC format:

    query_id Q0 doc_id rank score run_tag

Run tags:
    ES_BM25_Baseline          Phase 1
    transformer_faiss         Phase 2
    Hybrid_BM25_Transformer   Phase 3

Evaluation
----------
Results can be evaluated using trec_eval against a qrels file:

    trec_eval qrels.txt results_k20.txt

Notes
-----
- Phase 2 saves document embeddings to doc_embeddings.pkl after the first
  run and loads from disk on subsequent runs
- Query IDs are normalized to the format Q01, Q02, etc. to match qrels
- Phase 3 uses N=200 Elasticsearch candidates before re-ranking
```
