import faiss
import os
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

input_data_path = "data/transcripts/"
index_file = "vector_ts.index"
doc_mapping_file = "docs_mapping_ts.pkl"
output_index_path = "data/index/"

model = SentenceTransformer("all-MiniLM-L6-v2")

# Load the mapping once (instead of inside the function every time)
with open(os.path.join(output_index_path, doc_mapping_file), "rb") as f:
    id_to_metadata = pickle.load(f)  

def retrieve_relevant_chunks(query: str,index, k: int = 3):

    # Encode query
    query_vec = model.encode([query], convert_to_numpy=True)

    # Search
    distances, ids = index.search(query_vec, k)

    # Collect results (id + file + chunk_id + text + score)
    results = []
    for i, doc_id in enumerate(ids[0]):
        metadata = id_to_metadata.get(int(doc_id), None)

        if metadata:
            results.append({
                "id": int(doc_id),
                "file": metadata["file"],
                "chunk_id": metadata["chunk_id"],
                "text": metadata["chunk"],
                "score": float(distances[0][i])
            })

    return results