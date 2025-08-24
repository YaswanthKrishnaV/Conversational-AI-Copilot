import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


from source.llm import ask_llm

def test_llm_returns_stub_without_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    answer = ask_llm("List participants", chunks=[{
        "rank": 1, "file": "1_demo_call.txt", "chunk_id": 0, "score": 0.8,
        "text": "[00:00] AE: Hello Priya, welcome to the demo."
    }], call_ids=["1_demo_call"])
    assert isinstance(answer, str)
    assert "stub" in answer.lower() or "would answer" in answer.lower()
