from source import ingest as ing
from source import retriever as ret
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

def test_ingest_and_retrieve(tmp_repo):
    # Build index from both files
    index, mapping, call_ids = ing.ingest(paths=None)
    assert index.ntotal == len(mapping) > 0

    # Load and retrieve
    index2, mapping2, model = ret.load_index_and_mapping()
    q = "Who mentioned pricing?"
    chunks = ret.retrieve_relevant_chunks(q, index2, mapping2, model, k=2)
    assert isinstance(chunks, list) and len(chunks) > 0
    # Result shape sanity
    for c in chunks:
        assert {"rank", "vector_id", "score", "file", "chunk_id", "text"} <= set(c.keys())
