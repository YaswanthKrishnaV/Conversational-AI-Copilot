from typing import Any, Dict, List

from source.planner import Planner
from source.ingest import ingest as run_ingest
from source.retriever import load_index_and_mapping, retrieve_relevant_chunks, get_latest_call_id, get_chunks_for_call
import re
from sentence_transformers import SentenceTransformer
from source.llm import ask_llm

TOP_K = 10 # how many chunks to retrieve per query
global index, mapping, model, last_chunks
EMB_MODEL = "all-MiniLM-L6-v2"
SUMMARY_PAT = re.compile(r"\b(summar(ise|ize)|recap|overview)\b", re.I)
LAST_CALL_PAT = re.compile(r"\b(last|latest|yesterday)\b", re.I)
FULL_CALL_PAT = re.compile(r"\b(full|entire|whole)\b", re.I)
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

def choose_context_for_summary(query, last_chunks, mapping):
    q = query.lower()

    # 1) "summarize last call" → pick most recent call_id
    if SUMMARY_PAT.search(q) and LAST_CALL_PAT.search(q):
        cid = get_latest_call_id(mapping)
        return get_chunks_for_call(mapping, cid) if cid else last_chunks

    # 2) "summarize full call" (or "summarize this call"):
    #     use the call of the top-1 retrieved chunk
    if SUMMARY_PAT.search(q) and FULL_CALL_PAT.search(q):
        if last_chunks:
            cid = last_chunks[0].get("call_id")
            if cid:
                return get_chunks_for_call(mapping, cid)

    # 3) plain "summarize <call X>" (optional simple parse)
    m = re.search(r"\bcall\s*(\d+)\b", q)
    if SUMMARY_PAT.search(q) and m:
        want = m.group(1)
        # try to match stem suffix (e.g., "call3" from "3_demo_call")
        for meta in mapping.values():
            cid = meta.get("call_id") or ""
            if cid.endswith(want):
                return get_chunks_for_call(mapping, cid)

    # default: keep retrieved chunks
    return last_chunks

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
                context_chunks = last_chunks
                if SUMMARY_PAT.search(q):
                    context_chunks = choose_context_for_summary(q, last_chunks, mapping)

                answer = ask_llm(q, context_chunks, call_ids)

                print(f"\nAI: {answer}\n")
                if last_chunks:
                    print("Top sources:")
                    for c in context_chunks:
                        print(f"  [{c['rank']}] {c['file']} #{c['chunk_id']}  {c.get('chunk_start','--:--')}→{c.get('chunk_end','--:--')}  speakers={','.join(c.get('speakers',[]))}")
                        print(f"\n{c['text'][:100]}{'...' if len(c['text'])>100 else ''}")
                print("=======================================================END OF QUERY====================================================================\n")


if __name__ == "__main__":
    cli()
