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

from source.retriever import load_index_and_mapping

# paths
INPUT_DIR = Path("data/transcripts")
OUTPUT_DIR = Path("data/index")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = OUTPUT_DIR / "vector.index"
MAPPING_PATH = OUTPUT_DIR / "docs_mapping.pkl"

# model + chunking
EMB_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500  # (~3-6 dialogue turns)
CHUNK_OVERLAP = 50 # (~1 turn)
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

def _atomic_write_bytes(path: Path, write_fn):
    tmp = Path(str(path) + ".tmp")
    write_fn(tmp)
    os.replace(tmp, path)

def ingest(paths: List[str] | None = None, append: bool = False) -> Tuple[faiss.Index, Dict[int, Dict], List[str]]:
    """
    Build a fresh FAISS index (cosine similarity using IP on normalized vectors)
    with chunked transcripts. Saves:
      - FAISS index to data/index/vector.index
      - mapping dict {faiss_id: {...}} to data/index/docs_mapping.pkl
    If `paths` is None, ingests all *.txt in data/transcripts.

    - append=False: rebuild fresh from `paths` (or all *.txt).
    - append=True: load existing artifacts and ONLY add new call files (by call_id = filename stem).

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

    # Load existing artifacts if appending
    existing_index, existing_mapping, existing_model = load_index_and_mapping() if append else (None, {}, None)
    existing_call_ids = {m.get("call_id") for m in existing_mapping.values()} if append else set()

     # NEW: Build "all_files" from transcripts dir + explicitly provided files
    all_dir_files = [p for p in INPUT_DIR.glob("*.txt") if p.exists() and p.is_file()]
    all_files = list({fp.resolve(): fp for fp in (all_dir_files + files)}.values())  # de-dupe
    all_files = sorted(all_files, key=lambda p: (p.stat().st_mtime, p.name))
    
    # Synthetic dates per call_id (filename stem)
    call_dates = _assign_consecutive_dates(files)

     # Decide which files to process this run
    to_process: List[Path] = []
    for fp in files:
        cid = fp.stem
        if append and cid in existing_call_ids:
            # Skip already ingested call (simple policy).
            continue
        to_process.append(fp)

    if append and not to_process:
        # Nothing new; just return what we have
        if existing_index is None:
            raise RuntimeError("Append requested but no existing index found.")
        # Return call_ids in mtime order from all files present on disk
        call_ids = [fp.stem for fp in all_dir_files]
        return existing_index, existing_mapping, call_ids

    # Splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SPLIT_SEPARATORS
    )

    # Build chunks/meta and track texts for embedding
    new_texts: List[str] = []
    new_meta: List[Dict] = []
    for fp in to_process:
        raw = fp.read_text(encoding="utf-8", errors="ignore")
        chunks = splitter.split_text(raw)
        call_id = fp.stem
        call_date = call_dates.get(call_id)
        for i, ch in enumerate(chunks):
            ch = (ch or "").strip()
            if not ch:
                continue
            c_start, c_end, speakers = _extract_chunk_meta(ch)
            new_texts.append(ch)
            new_meta.append({
                "file": fp.name,
                "call_id": call_id,
                "call_date": call_date,
                "chunk_id": i,
                "chunk_start": c_start,
                "chunk_end": c_end,
                "speakers": speakers,
                "text": ch
            })

    if not new_texts and not (append and existing_index is not None):
        raise ValueError("No new chunks produced. Nothing to ingest.")

    # Embed + normalize (cosine)
    model = SentenceTransformer(EMB_MODEL) if existing_model is None else existing_model
    if new_texts:
        emb = model.encode(new_texts, convert_to_numpy=True, show_progress_bar=True)
        emb = _normalize_rows(emb).astype(np.float32)
    else:
        emb = np.zeros((0, existing_index.d), dtype=np.float32)

    # Build or append FAISS
    if append and existing_index is not None:
        index = existing_index
        if emb.shape[0] > 0:
            if emb.shape[1] != index.d:
                raise ValueError(f"Embedding dim {emb.shape[1]} != index dim {index.d}")
            base = index.ntotal
            index.add(emb)
            # Extend mapping with new ids
            id_to_meta = existing_mapping.copy()
            for j, m in enumerate(new_meta):
                id_to_meta[base + j] = m
        else:
            id_to_meta = existing_mapping
    else:
        # fresh build
        if not new_texts:
            raise ValueError("No texts to build a fresh index.")
        
        # FAISS cosine via Inner Product on unit vectors
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        id_to_meta = {i: new_meta[i] for i in range(len(new_meta))}


    # Persist atomically
    _atomic_write_bytes(INDEX_PATH, lambda p: faiss.write_index(index, str(p)))
    def _dump_map(pth: Path):
        with open(pth, "wb") as f:
            pickle.dump(id_to_meta, f)
    _atomic_write_bytes(MAPPING_PATH, _dump_map)

    # Call IDs (mtime order, oldest → newest)
    call_ids = [fp.stem for fp in all_files]

    print(f"Ingestion ({'Added' if append else 'New'}): {index.ntotal} vectors total → {INDEX_PATH.name}, {MAPPING_PATH.name}")
    return index, id_to_meta, call_ids
