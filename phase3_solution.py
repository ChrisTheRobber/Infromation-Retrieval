
# Imports
import pandas as pd
import numpy as np
import faiss
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# STEP 1: Setup & Connections
client = Elasticsearch("http://localhost:9200")
INDEX_NAME = "ir2025_phase_1" # Index apo phase 1
model = SentenceTransformer('all-MiniLM-L6-v2')

queries_df = pd.read_csv('queries.csv')
# Dictionary for mapping doc_id to its text
df_docs = pd.read_csv('documents.csv')
doc_text_map = dict(zip(df_docs['ID'].astype(str), df_docs['Text']))

N = 200  # Number of candidates from Elasticsearch
k_values = [20, 30, 50]

# STEP 2: Hybrid Retrieval Loop
for k in k_values:
    output_filename = f"results_k{k}.txt"
    print(f"Generating {output_filename}...")
    
    with open(output_filename, 'w') as f:
        for _, q_row in queries_df.iterrows():
            # Standardize query ID
            raw_id = str(q_row['ID']).strip()
            q_id = raw_id if raw_id.startswith('Q') else f"Q{raw_id.zfill(2)}"
            q_text = q_row['Text']

            # --- PART A: Elasticsearch ---
            res = client.search(
                index=INDEX_NAME,
                size=N,
                query={"match": {"text": q_text}}
            )
            
            # Extract doc IDs and their actual text for the candidates
            candidate_hits = res['hits']['hits']
            candidate_ids = [hit['_id'] for hit in candidate_hits]
            candidate_texts = [doc_text_map[cid] for cid in candidate_ids]

            if not candidate_texts:
                continue

            # --- PART B: Semantic Embedding (Transformer) ---
            q_vec = model.encode([q_text], convert_to_numpy=True)
            doc_vecs = model.encode(candidate_texts, convert_to_numpy=True)
            
            # Normalize for Cosine Similarity
            faiss.normalize_L2(q_vec)
            faiss.normalize_L2(doc_vecs)

            # --- PART C: Re-ranking (FAISS) ---
            dim = doc_vecs.shape[1]
            temp_index = faiss.IndexFlatIP(dim)
            temp_index.add(doc_vecs)

            # Search the small index
            D, I = temp_index.search(q_vec, min(k, len(candidate_ids)))

            # --- PART D: Output in TREC Format ---
            for rank, (candidate_idx, score) in enumerate(zip(I[0], D[0])):
                doc_id = candidate_ids[candidate_idx]
                f.write(f"{q_id} Q0 {doc_id} {rank + 1} {score:.6f} Hybrid_BM25_Transformer\n")

print("Hybrid model complete!")
