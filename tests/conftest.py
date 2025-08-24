import pytest
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


@pytest.fixture
def tmp_repo(tmp_path, monkeypatch):
    root = tmp_path
    d_transcripts = root / "data" / "transcripts"
    d_index = root / "data" / "index"
    d_transcripts.mkdir(parents=True)
    d_index.mkdir(parents=True)

    # Two tiny transcripts
    (d_transcripts / "1_demo_call.txt").write_text(
        "[00:00] AE: Hello Priya, welcome to the demo.\n"
        "[00:05] Prospect (Priya): Thanks! Pricing?\n",
        encoding="utf-8"
    )
    

    # Point your app paths to this temp repo
    import source.ingest as ing
    import source.retriever as ret
    ing.INPUT_DIR = d_transcripts
    ing.OUTPUT_DIR = d_index
    ing.INDEX_PATH = d_index / "vector.index"
    ing.MAPPING_PATH = d_index / "docs_mapping.pkl"
    ret.INDEX_PATH = str(ing.INDEX_PATH)
    ret.MAPPING_PATH = str(ing.MAPPING_PATH)

    # Make sure LLM won’t call network
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    return {"transcripts": d_transcripts, "index_dir": d_index}
