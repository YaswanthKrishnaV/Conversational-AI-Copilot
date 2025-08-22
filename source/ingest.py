import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2") ##TBM

input_data_path = "data/transcripts/"
index_file = "vector.index"
doc_mapping_file = "docs_mapping.pkl"
output_index_path = "data/index/"

def ingest():
    texts = []
    for file in os.listdir(input_data_path):
        if file.endswith(".txt"):
            with open(os.path.join(input_data_path, file), "r", encoding="utf-8") as f:
                texts.append(f.read())

    # Encode
    embeddings = model.encode(texts, convert_to_numpy=True)

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, os.path.join(output_index_path,index_file))

    # Create mapping {faiss_id: text}
    id_to_text = {i: text for i, text in enumerate(texts)}

    # Save mapping
    with open(os.path.join(output_index_path, doc_mapping_file), "wb") as f:
        pickle.dump(id_to_text, f)

    print("Ingestion complete: index + mapping saved!")
    return index

if __name__ == "__main__":
    index = ingest()