# app/retriever.py
from __future__ import annotations

import os
import pickle
from typing import List, Dict, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = "data/index/vector.index"
MAPPING_PATH = "data/index/docs_mapping.pkl"
EMB_MODEL = "all-MiniLM-L6-v2"


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def load_index_and_mapping() -> Tuple[faiss.Index, Dict[int, Dict], SentenceTransformer]:
    if not (os.path.exists(INDEX_PATH) and os.path.exists(MAPPING_PATH)):
        raise FileNotFoundError("Index or mapping not found; run ingestion first.")
    index = faiss.read_index(INDEX_PATH)
    with open(MAPPING_PATH, "rb") as f:
        mapping: Dict[int, Dict] = pickle.load(f)
    model = SentenceTransformer(EMB_MODEL)
    return index, mapping, model


def retrieve_relevant_chunks(query: str,
                             index: faiss.Index,
                             mapping: Dict[int, Dict],
                             model: SentenceTransformer,
                             k: int = 4) -> List[Dict]:
    q_vec = model.encode([query], convert_to_numpy=True)
    q_vec = _normalize_rows(q_vec).astype(np.float32)  # cosine via IP

    sims, ids = index.search(q_vec, k)
    sims = sims[0].tolist()
    ids = ids[0].tolist()

    results: List[Dict] = []
    rank = 1
    for vid, score in zip(ids, sims):
        if vid == -1:
            continue
        meta = mapping.get(int(vid))
        if not meta:
            continue
        results.append({
            "rank": rank,
            "vector_id": int(vid),
            "score": float(score),  # cosine similarity (-1..1); higher is better
            "file": meta["file"],
            "chunk_id": meta["chunk_id"],
            "text": meta["text"],
        })
        rank += 1

    return results
