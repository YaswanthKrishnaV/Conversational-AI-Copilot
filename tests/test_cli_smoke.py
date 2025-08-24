import builtins
from source import ingest as ing
import cli as cli_mod
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


def test_cli_runs_one_turn(tmp_repo, monkeypatch, capsys):
    ing.ingest(paths=None)  # build index

    # Script one query then exit
    inputs = iter(["list all people on the calls", "exit"])
    monkeypatch.setattr(builtins, "input", lambda _="": next(inputs))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)  # avoid real LLM

    cli_mod.cli() if hasattr(cli_mod, "cli") else cli_mod.main()

    out = capsys.readouterr().out.lower()
    assert "ai bot:" in out  # printed an answer (stub is fine)
