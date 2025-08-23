from __future__ import annotations
import asyncio
from typing import Dict, Any, List

from agentic_framework import Orchestrator

def print_sources(chunks: List[Dict[str, Any]]):
    if not chunks:
        return
    print("Top sources:")
    for c in chunks:
        print(f"  [{c.get('rank')}] {c.get('file')} #{c.get('chunk_id')}  score={c.get('score'):.3f} \n\n{c.get('text')}\n")
    print()

def main():
    orch = Orchestrator(top_k=10)
    print("🤖 Agentic Conversational AI Copilot (type 'exit' to quit)")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        result = loop.run_until_complete(orch.handle(q))
        rtype = result.get("type")

        if rtype == "ingest_ok":
            print(f"AI: {result['message']}\n")
            continue

        if rtype == "answer":
            print(f"\nAI: {result['answer']}\n")
            print_sources(result.get("chunks", []))
            continue

        if rtype == "irrelevant":
            print(f"AI: {result['message']}\n")
            continue

        # errors / fallbacks
        print(f"AI: {result.get('message','Sorry, something went wrong.')}\n")

if __name__ == "__main__":
    main()
