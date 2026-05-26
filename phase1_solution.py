
# Imports
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# 1. Σύνδεση στην μηχανή αναζήτησης
client = Elasticsearch("http://localhost:9200")
INDEX_NAME = "ir2025_phase_1"

# 2. DATA LOADING & PREPROCESSING - Load the documents and queries
df_docs = pd.read_csv("documents.csv") 
df_queries = pd.read_csv("queries.csv")

# Prepare documents for bulk indexing - Map 'ID' from CSV to '_id' in ElasticSearch and 'Text' to 'text'
docs_to_index = []
for _, row in df_docs.iterrows():
    docs_to_index.append({
        "_index": INDEX_NAME,
        "_id": str(row['ID']), 
        "_source": {
            "text": row['Text']
        }
    })

# 3. INDEX CONFIGURATION (BM25 & Analyzer)
if client.indices.exists(index=INDEX_NAME):
    client.indices.delete(index=INDEX_NAME)

mapping = {
    "settings": {
        "analysis": {
            "analyzer": {
                "default": {"type": "english"} # Stemming/stopwords
            }
        },
        "similarity": {
            "default": {
                "type": "BM25" # BM25
            }
        }
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"}
        }
    }
}

# Create index and upload data
client.indices.create(index=INDEX_NAME, body=mapping)
bulk(client, docs_to_index)
print(f"Successfully indexed {len(docs_to_index)} documents.")

# 4. EXECUTION & RESULTS GENERATION in trec_eval format
k_values = [20, 30, 50]

for k in k_values:
    output_file = f"results_k{k}.txt"
    with open(output_file, "w") as f:
        for _, q_row in df_queries.iterrows():
            # Standardize query ID to match qrels
            raw_id = str(q_row['ID']).strip()
            # if ID == "1", -> "Q01"
            q_id = raw_id if raw_id.startswith('Q') else f"Q{raw_id.zfill(2)}"
            
            q_text = q_row['Text']
            
            # Search using the 'text' field
            res = client.search(
                index=INDEX_NAME,
                size=k,
                query={"match": {"text": q_text}}
            )
            
            # Format: query_id Q0 doc_id rank score run_tag
            for rank, hit in enumerate(res['hits']['hits'], start=1):
                doc_id = hit['_id']
                score = hit['_score']
                f.write(f"{q_id} Q0 {doc_id} {rank} {score} ES_BM25_Baseline\n")
    
    print(f"Generated {output_file} for trec_eval.")

print("All tasks for Phase 1 are complete.")
