from typing import Any, Dict, List

from planner import Planner
from ingest import ingest as run_ingest
from retriever import load_index_and_mapping, retrieve_relevant_chunks
from sentence_transformers import SentenceTransformer
from llm import ask_llm

TOP_K = 10 # how many chunks to retrieve per query
global index, mapping, model, last_chunks
EMB_MODEL = "all-MiniLM-L6-v2"

# runtime state
index = None
mapping = None
model = None
last_chunks: List[Dict[str, Any]] = []


def reload_index_mapping_model():
    
    index, mapping, model = load_index_and_mapping()
    return index, mapping, model


def retrieval_probe_fn(question: str) -> int:
    """
    Tiny check to see if the KB likely has context.
    """
    try:
        chunks = retrieve_relevant_chunks(question, index, mapping, model, k=1)
        return len([c for c in chunks if c.get("text")])
    except Exception:
        return 0


def cli():
    global index, mapping, model, last_chunks

    # Lazy load KB/index on first question
    loaded_once = False
    planner = Planner(top_k=TOP_K)

    if not loaded_once:
        print("Building Index from transcripts in data/transcripts/ ...")
        index, mapping, call_ids = run_ingest(paths=None)
        model = SentenceTransformer(EMB_MODEL)
        print("Data Ingested & loaded.")
        loaded_once = True
    print("\n\nConversational AI Copilot - Please ask your Query (type 'exit' to quit)")

    while True:
        q = input("\n\nUSER: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        # PLAN → ACTIONS
        plan = planner.plan(q, retrieval_probe=retrieval_probe_fn)

        # Single-action outcomes
        single = plan[0]
        if len(plan) == 1 and single["action"] in {"block", "irrelevant", "cannot_ingest", "cannot_answer"}:
            print(f"AI Bot: {single['action'].replace('_',' ')} — {single['reason']}")
            continue

        # Execute plan
        for step in plan:
            action = step["action"]

            if action == "ingest":
                paths = step["args"]["paths"]
                print(f"📥 Ingesting {len(paths)} file(s): {paths}")
                index, mapping, call_ids = run_ingest(paths=paths)        # builds+saves fresh index & mapping
                # reload_index_mapping_model()   # reload in-memory handles
                print("✅ Ingestion complete & index reloaded.\n")
                break  # end this turn after ingest

            if action == "retrieve":
                k = int(step["args"]["top_k"])
                last_chunks = retrieve_relevant_chunks(q, index, mapping, model, k=k)

            if action == "synthesize":
                answer = ask_llm(q, last_chunks, call_ids)
                print(f"\nAI: {answer}\n")
                if last_chunks:
                    print("Top sources:")
                    for c in last_chunks:
                        print(f"  [{c['rank']}] {c['file']} #{c['chunk_id']}  score={c['score']:.3f} text = {c['text']}\n\n")
                print()


if __name__ == "__main__":
    cli()
