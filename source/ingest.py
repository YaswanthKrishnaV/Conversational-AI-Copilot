import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter


model = SentenceTransformer("all-MiniLM-L6-v2") ##TBM

input_data_path = "data/transcripts/"
index_file = "vector_ts.index"
doc_mapping_file = "docs_mapping_ts.pkl"
output_index_path = "data/index/"

def ingest():
    texts = []          # store chunks
    meta_mapping = []   # store metadata (chunk, file, chunk_id)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )


    # Process each file
    for file in os.listdir(input_data_path):
        if file.endswith(".txt"):
            with open(os.path.join(input_data_path, file), "r", encoding="utf-8") as f:
                text = f.read()
            
            # Split into chunks
            chunks = splitter.split_text(text)

            for i, chunk in enumerate(chunks):
                texts.append(chunk)
                meta_mapping.append({
                    "file": file,
                    "chunk_id": i,
                    "chunk": chunk
                })
    
    if not texts:
        raise ValueError("No text files found for ingestion!")
    # print(f"✅ Ingested {len(texts)} chunks from {len(os.listdir(input_data_path))} files.")

    # Encode all chunks
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Ensure output directory exists
    os.makedirs(output_index_path, exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, os.path.join(output_index_path, index_file))

    # Save mapping: {faiss_id: metadata}
    id_to_metadata = {i: meta_mapping[i] for i in range(len(meta_mapping))}
    with open(os.path.join(output_index_path, doc_mapping_file), "wb") as f:
        pickle.dump(id_to_metadata, f)

    print("Ingestion complete: index + mapping saved!")
    return index

if __name__ == "__main__":
    index = ingest()