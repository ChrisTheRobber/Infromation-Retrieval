
import pandas as pd
import numpy as np
import faiss
import os
import pickle
from sentence_transformers import SentenceTransformer

# ==========================================
# STEP 1: Load and Preprocess Data
# ==========================================
print("Step 1: Loading documents and queries...")
df = pd.read_csv('documents.csv')
queries_df = pd.read_csv('queries.csv')

# Preprocessing: Keep it natural for the Transformer
df['prepared_text'] = df['Text'].astype(str).str.strip()
doc_ids = df['ID'].tolist()

# Prepare queries
queries_df['prepared_text'] = queries_df['Text'].astype(str).str.strip()
query_dict = dict(zip(queries_df['ID'], queries_df['prepared_text']))

print(f"Loaded {len(df)} documents and {len(query_dict)} queries.")

# ==========================================
# STEP 2: Generate or Load Embeddings
# ==========================================
# Model requirement: all-MiniLM-L6-v2 (384 dimensions)
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_file = 'doc_embeddings.pkl'

if not os.path.exists(embedding_file):
    print("Step 2: Generating document embeddings (this may take a few minutes)...")
    doc_embeddings = model.encode(df['prepared_text'].tolist(), show_progress_bar=True, convert_to_numpy=True)
    
    with open(embedding_file, 'wb') as f:
        pickle.dump({'ids': doc_ids, 'embeddings': doc_embeddings}, f)
    print("Embeddings generated and saved.")
else:
    print("Step 2: Loading existing embeddings from disk...")
    with open(embedding_file, 'rb') as f:
        data = pickle.load(f)
        doc_embeddings = data['embeddings']
    print("Embeddings loaded successfully.")

# ==========================================
# STEP 3: Build the FAISS Index
# ==========================================
print("Step 3: Building FAISS index...")
dimension = doc_embeddings.shape[1]

# IndexFlatIP + L2 Normalization = Cosine Similarity
index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(doc_embeddings)
index.add(doc_embeddings)

print(f"Index built with {index.ntotal} vectors.")

# ==========================================
# STEP 4: Semantic Search & Export
# ==========================================
print("Step 4: Running searches and generating results...")
k_values = [20, 30, 50]
tag = "transformer_faiss"

for k in k_values:
    output_filename = f"results_k{k}.txt"
    with open(output_filename, 'w') as f:
        for q_id, q_text in query_dict.items():
            # a) Encode and Normalize Query
            query_vec = model.encode([q_text], convert_to_numpy=True)
            faiss.normalize_L2(query_vec)
            
            # b) Search FAISS
            D, I = index.search(query_vec, k)
            
            # c) Write in TREC format
            for rank, (doc_idx, score) in enumerate(zip(I[0], D[0])):
                doc_id = doc_ids[doc_idx]
                line = f"{q_id} Q0 {doc_id} {rank + 1} {score:.6f} {tag}\n"
                f.write(line)
                
    print(f"Created {output_filename}")

print("\n--- All Steps Complete! ---")
