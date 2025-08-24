from typing import Any, Dict, List
from source.llm import ask_llm
from source.planner import Planner
from source.ingest import ingest as run_ingest
from source.retriever import load_index_and_mapping, retrieve_relevant_chunks, get_latest_call_id, get_chunks_for_call, retrieve_topic_chunks
from source.scope_router import scope_route

import re
from sentence_transformers import SentenceTransformer


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


def get_chunks_from_scope(route: Dict[str, Any]) -> List[Dict[str, Any]]:

    scope = route.get("scope", "topk_only")

    # default: topk_only
    chosen_chunks = last_chunks

    if scope == "last_call":
        cid = get_latest_call_id(mapping)
        if cid:
            chosen_chunks = get_chunks_for_call(mapping, cid)

    elif scope == "full_call_of_top_hit":
        if last_chunks:
            cid = last_chunks[0].get("call_id")
            if cid:
                chosen_chunks = get_chunks_for_call(mapping, cid)

    elif scope == "call_by_id":
        # Expect call_id to be something like "1_demo_call" or just a suffix digit.
        req = (route.get("call_id") or "").strip()
        if req:
            # Exact match
            for meta in mapping.values():
                if meta.get("call_id") == req:
                    chosen_chunks = get_chunks_for_call(mapping, req)
                    break
            else:
                # suffix match (e.g., "3" matches "..._3" or endswith 3)
                for meta in mapping.values():
                    cid = meta.get("call_id") or ""
                    if cid.endswith(req):
                        chosen_chunks = get_chunks_for_call(mapping, cid)
                        break

    elif scope == "all_calls":
        # Concatenate every chunk in call order (be careful with long contexts)
        chosen_chunks = [
            {"rank": i, "vector_id": vid, "score": 1.0, **meta}
            for i, (vid, meta) in enumerate(sorted(
                mapping.items(), key=lambda x: (x[1].get("call_date",""), x[1].get("chunk_id",0))
            ), start=1)
        ]

    elif scope == "topic_across_calls":
        topic = (route.get("topic") or "").strip()
        topic_query = topic if topic else q  # fallback to full query if LLM didn’t extract
        chosen_chunks = retrieve_topic_chunks(
            topic_query, index, mapping, model, k_max=100, score_threshold=0.25
        )
        if not chosen_chunks:
            chosen_chunks = last_chunks
        

    return chosen_chunks


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
                index, mapping, call_ids = run_ingest(paths=paths,append=True)        # builds+saves fresh index & mapping
                break  # end this turn after ingest

            if action == "retrieve":
                k = int(step["args"]["top_k"])
                last_chunks = retrieve_relevant_chunks(q, index, mapping, model, k=k)

            if action == "synthesize":

                # 2) ask LLM to pick scope
                route = scope_route(q, call_ids)
                
                context_chunks = get_chunks_from_scope(route)
                
                answer = ask_llm(q, context_chunks, call_ids)

                print(f"\nAI: {answer}\n")
                if last_chunks:
                    print("\n==================================  Top sources:  =====================================\n")
                    print("Rank | File | Chunk ID | Time | Speakers | Text")
                    for c in context_chunks[:5]:
                        
                        print(f"  [{c['rank']}] {c['file']} #{c['chunk_id']}  {c.get('chunk_start','--:--')}→{c.get('chunk_end','--:--')}  speakers={','.join(c.get('speakers',[]))}")
                        print(f"Text -- {c['text'][:100]}{'...' if len(c['text'])>100 else ''}")
                    print(".\n.\n.\n.\n.")
                    for c in context_chunks[-5:]:
                        print(f"  [{c['rank']}] {c['file']} #{c['chunk_id']}  {c.get('chunk_start','--:--')}→{c.get('chunk_end','--:--')}  speakers={','.join(c.get('speakers',[]))}")
                        print(f"Text -- {c['text'][:100]}{'...' if len(c['text'])>100 else ''}")
        
        print("\n=========================================  ASK YOUR NEXT QUERY  ===================================================\n")


if __name__ == "__main__":
    cli()
