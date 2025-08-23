# app/ingest.py
from __future__ import annotations

import os
import re
import pickle
from pathlib import Path
from typing import List, Tuple, Dict
from datetime import date, timedelta

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
# Optional: make transcript-aware splitting prefer timestamped lines like "[00:05]"
SPLIT_SEPARATORS = ["\n[", "\n\n", "\n", ". ", " ", ""]  # safe default; keep if you like

# Regex to capture timestamp + speaker at the start of a line: [mm:ss] Speaker:
TS_SPEAKER_RE = re.compile(r"^\[(\d{2}:\d{2})\]\s*([^:]+):", re.MULTILINE)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def _extract_chunk_meta(text: str):
    """
    From a chunk, extract:
      - first/last timestamps present (mm:ss) → chunk_start / chunk_end
      - ordered unique speakers list
    """
    times, speakers = [], []
    for m in TS_SPEAKER_RE.finditer(text):
        times.append(m.group(1))
        speakers.append(m.group(2).strip())
    speakers = list(dict.fromkeys(speakers))  # de-dupe, preserve order
    start = times[0] if times else None
    end = times[-1] if times else None
    return start, end, speakers


def _assign_consecutive_dates(files: List[Path]) -> Dict[str, str]:
    """
    Assign synthetic dates assuming calls happened on consecutive days:
      newest file -> yesterday; previous -> day before yesterday; etc.
    Returns: { call_id (stem) : ISO date string 'YYYY-MM-DD' }
    """
    if not files:
        return {}
    # Sort by modified time ascending (oldest -> newest), tie-break by name
    files_sorted = sorted(files, key=lambda p: (p.stat().st_mtime, p.name))
    today = date.today()
    base = today - timedelta(days=1)  # newest = yesterday
    mapping: Dict[str, str] = {}
    # Walk newest first and assign decreasing days
    for offset, fp in enumerate(reversed(files_sorted)):
        mapping[fp.stem] = (base - timedelta(days=offset)).isoformat()
    return mapping


def ingest(paths: List[str] | None = None) -> Tuple[faiss.Index, Dict[int, Dict], List[str]]:
    """
    Build a fresh FAISS index (cosine similarity using IP on normalized vectors)
    with chunked transcripts. Saves:
      - FAISS index to data/index/vector.index
      - mapping dict {faiss_id: {...}} to data/index/docs_mapping.pkl
    If `paths` is None, ingests all *.txt in data/transcripts.

    Returns:
      index, id_to_meta, call_ids   (call_ids are filename stems in mtime order)
    """
    # Collect files
    if paths:
        files: List[Path] = [Path(p) for p in paths]
    else:
        files = [p for p in INPUT_DIR.glob("*.txt")]

    # Sort by modified time (oldest → newest), tie-break by name
    files = sorted([fp for fp in files if fp.exists() and fp.is_file()],
                   key=lambda p: (p.stat().st_mtime, p.name))

    if not files:
        raise FileNotFoundError(f"No .txt files found in {INPUT_DIR} (or given paths).")

    # Synthetic dates per call_id (filename stem)
    call_dates = _assign_consecutive_dates(files)

    # Splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SPLIT_SEPARATORS
    )

    texts: List[str] = []
    meta: List[Dict] = []

    # Build chunks + enriched metadata
    for fp in files:
        raw = fp.read_text(encoding="utf-8", errors="ignore")
        parts = splitter.split_text(raw)
        call_id = fp.stem
        call_date = call_dates.get(call_id)

        for i, ch in enumerate(parts):
            ch = (ch or "").strip()
            if not ch:
                continue
            c_start, c_end, speakers = _extract_chunk_meta(ch)
            texts.append(ch)
            meta.append({
                "file": fp.name,
                "call_id": call_id,
                "call_date": call_date,     # synthetic YYYY-MM-DD
                "chunk_id": i,
                "chunk_start": c_start,     # "mm:ss" or None
                "chunk_end": c_end,         # "mm:ss" or None
                "speakers": speakers,       # ["AE", "Prospect", ...]
                "text": ch
            })

    if not texts:
        raise ValueError("No chunks produced. Ensure there are .txt files with content.")

    # Embed + normalize (cosine)
    model = SentenceTransformer(EMB_MODEL)
    emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    emb = _normalize_rows(emb).astype(np.float32)

    # FAISS cosine via Inner Product on unit vectors
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    # Persist artifacts
    faiss.write_index(index, str(INDEX_PATH))
    id_to_meta = {i: meta[i] for i in range(len(meta))}
    with open(MAPPING_PATH, "wb") as f:
        pickle.dump(id_to_meta, f)

    # Call IDs (mtime order, oldest → newest)
    call_ids = [fp.stem for fp in files]

    print(f"✅ Ingestion: {len(texts)} chunks → {INDEX_PATH.name}, {MAPPING_PATH.name}")
    return index, id_to_meta, call_ids
