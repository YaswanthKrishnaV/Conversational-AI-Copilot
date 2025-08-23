# app/ingest.py
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import List, Tuple, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

# paths
INPUT_DIR = Path("data/transcripts")
OUTPUT_DIR = Path("data/index")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = OUTPUT_DIR / "vector.index"
MAPPING_PATH = OUTPUT_DIR / "docs_mapping.pkl"

# model + chunking
EMB_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 750
CHUNK_OVERLAP = 50


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def ingest(paths: List[str] | None = None) -> Tuple[faiss.Index, Dict[int, Dict]]:
    """
    Build a fresh FAISS index (cosine similarity using IP on normalized vectors)
    with chunked transcripts. Saves:
      - FAISS index to data/index/vector.index
      - mapping dict {faiss_id: {file, chunk_id, text}} to data/index/docs_mapping.pkl
    If `paths` is None, ingests all *.txt in data/transcripts.
    """
    files: List[Path] = []
    if paths:
        files = [Path(p) for p in paths]
    else:
        files = sorted([p for p in INPUT_DIR.glob("*.txt")])

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts: List[str] = []
    meta: List[Dict] = []

    for fp in files:
        if not fp.exists() or not fp.is_file():
            continue
        text = fp.read_text(encoding="utf-8", errors="ignore")
        chunks = splitter.split_text(text)
        for i, ch in enumerate(chunks):
            if ch.strip():
                texts.append(ch.strip())
                meta.append({"file": fp.name, "chunk_id": i, "text": ch.strip()})

    if not texts:
        raise ValueError("No chunks produced. Ensure there are .txt files with content.")

    # Embed + normalize (cosine)
    model = SentenceTransformer(EMB_MODEL)
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    emb = _normalize_rows(emb).astype(np.float32)

    index = faiss.IndexFlatIP(emb.shape[1])  # cosine via inner product on unit vectors
    index.add(emb)

    # Persist
    faiss.write_index(index, str(INDEX_PATH))
    id_to_meta = {i: meta[i] for i in range(len(meta))}
    with open(MAPPING_PATH, "wb") as f:
        pickle.dump(id_to_meta, f)

    print(f"✅ Ingestion: {len(texts)} chunks → {INDEX_PATH.name}, {MAPPING_PATH.name}")
    return index, id_to_meta
