import faiss
import os
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

input_data_path = "data/transcripts/"
index_file = "vector.index"
doc_mapping_file = "docs_mapping.pkl"
output_index_path = "data/index/"

model = SentenceTransformer("all-MiniLM-L6-v2")

# Load the mapping once (instead of inside the function every time)
with open(os.path.join(output_index_path, doc_mapping_file), "rb") as f:
    id_to_text = pickle.load(f)  

def retrieve_relevant_chunks(query: str, k: int = 2):
    # Load index
    index = faiss.read_index(os.path.join(output_index_path,index_file))

    # Encode query
    query_vec = model.encode([query], convert_to_numpy=True)

    # Search
    distances, ids = index.search(query_vec, k)

    # Collect results (text + id + score)
    results = []
    for i, doc_id in enumerate(ids[0]):
        text = id_to_text.get(doc_id, "[Text not found]")
        results.append({
            "id": int(doc_id),
            "text": text,
            "score": float(distances[0][i])
        })

    return results