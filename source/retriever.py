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

def get_all_call_ids(mapping):
    """Return unique call_ids sorted by call_date if available, else by name."""
    items = []
    for meta in mapping.values():
        items.append((meta.get("call_id"), meta.get("call_date")))
    # de-dupe while preserving last seen date
    seen = {}
    for cid, cdate in items:
        if cid:
            seen[cid] = cdate
    # sort by date asc; None last; tie-break by id
    def _key(x):
        cid, cdate = x
        return ((cdate is None), cdate or "", cid or "")
    return [cid for cid, _ in sorted(seen.items(), key=_key)]

def get_latest_call_id(mapping):
    """Pick call with the most recent call_date (synthetic); fallback to max by name."""
    call_dates = {}
    for meta in mapping.values():
        cid = meta.get("call_id")
        cdate = meta.get("call_date")
        if cid:
            # keep max date if duplicates
            if cdate and (cid not in call_dates or cdate > call_dates[cid]):
                call_dates[cid] = cdate
            elif cid not in call_dates:
                call_dates[cid] = None
    # prefer those with dates; pick max date; fallback to max id
    dated = [(cid, d) for cid, d in call_dates.items() if d]
    if dated:
        return max(dated, key=lambda x: x[1])[0]
    return max(call_dates.keys()) if call_dates else None

def get_chunks_for_call(mapping, call_id):
    """Return ALL chunks (dicts) for a given call_id, ordered by chunk_id."""
    rows = [
        {"rank": i, "vector_id": vid, "score": 1.0, **meta}
        for i, (vid, meta) in enumerate(sorted(
            [(vid, m) for vid, m in mapping.items() if m.get("call_id") == call_id],
            key=lambda x: x[1].get("chunk_id", 0)
        ), start=1)
    ]
    return rows

def load_index_and_mapping() -> Tuple[faiss.Index, Dict[int, Dict], SentenceTransformer]:
    if not (os.path.exists(INDEX_PATH) and os.path.exists(MAPPING_PATH)):
        raise FileNotFoundError("Index or mapping not found; run ingestion first.")
    index = faiss.read_index(INDEX_PATH)
    with open(MAPPING_PATH, "rb") as f:
        mapping: Dict[int, Dict] = pickle.load(f)
    model = SentenceTransformer(EMB_MODEL)
    return index, mapping, model

def retrieve_topic_chunks(
    topic_query: str,
    index: faiss.Index,
    mapping: dict[int, dict],
    model,
    k_max: int = 1000,
    score_threshold: float = 0.25,
) -> list[dict]:
    """
    Retrieve MANY chunks related to `topic_query` across ALL calls.
    - Returns all hits above `score_threshold`, up to k_max (capped at index size).
    - Sorted by (call_date asc, chunk_id asc) for coherent reading.
    """
    if index is None or index.ntotal == 0:
        return []

    # Encode + normalize (cosine)
    q = model.encode([topic_query], convert_to_numpy=True)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)

    k = min(k_max, index.ntotal)
    sims, ids = index.search(q.astype(np.float32), k)  # IP on normalized → cosine
    sims, ids = sims[0], ids[0]

    rows = []
    for rank, (vec_id, score) in enumerate(zip(ids, sims), start=1):
        if vec_id < 0:
            continue
        if score < score_threshold:
            continue
        meta = mapping.get(int(vec_id))
        if not meta:
            continue
        rows.append({
            "rank": rank,
            "vector_id": int(vec_id),
            "score": float(score),
            **meta,
        })

    # Stable order for reading: by call_date then chunk_id
    rows.sort(key=lambda r: (r.get("call_date",""), r.get("chunk_id", 0)))
    # re-rank after sort (optional)
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return rows

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

    # ids, sims should be 1-D here (flatten if needed)
    for vid, score in zip(ids, sims):
        if vid == -1:
            continue
        meta = mapping.get(int(vid))
        if not meta:
            continue

        results.append({
            "rank": rank,
            "vector_id": int(vid),
            "score": float(score),               # cosine similarity (-1..1); higher is better
            "file": meta.get("file"),
            "chunk_id": meta.get("chunk_id"),
            "text": meta.get("text", ""),
            "call_id": meta.get("call_id"),
            "call_date": meta.get("call_date"),  # synthetic YYYY-MM-DD if you used that logic
            "chunk_start": meta.get("chunk_start"),  # "mm:ss" or None
            "chunk_end": meta.get("chunk_end"),
            "speakers": meta.get("speakers", []),
        })
        rank += 1

    return results

