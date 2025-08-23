from __future__ import annotations
import json, re
from typing import Any, Dict, List, Optional, Tuple

from source.llm import get_azure_openai_client
from source.planner import _extract_json_block


def scope_route(query: str) -> Dict[str, Any]:
    """
    Ask the LLM to pick a retrieval scope. Returns a dict like:
    {
      "scope": "topk_only" | "last_call" | "full_call_of_top_hit" | "call_by_id" | "all_calls",
      "call_id": "<optional>",
      "reason": "<short>"
    }
    Falls back to a heuristic if no client/keys or on error.
    """
    client, model_name= get_azure_openai_client()
    
    system = (
        "You are a routing assistant for a RAG system over SALES CALL TRANSCRIPTS.\n"
        "Your job is to choose the **context scope** to answer a user request.\n\n"
        "Available scopes:\n"
        "- topk_only: Use the top-k retrieved chunks only (default for narrow Q&A).\n"
        "- full_call_of_top_hit: Use ALL chunks from the call that contains the top-1 retrieved chunk (good for 'summarize this call').\n"
        "- last_call: Use ALL chunks from the most recent call (good for 'summarize the last call').\n"
        "- call_by_id: Use ALL chunks from a specific call_id explicitly mentioned (e.g., 'call 3').\n"
        "- all_calls: Use chunks from ALL calls (rare; broad analytics across every call).\n\n"
        "Rules:\n"
        "1) If the user asks to summarize the last/latest call → last_call.\n"
        "2) If the user asks to summarize the full/entire call without specifying which → full_call_of_top_hit.\n"
        "3) If the user mentions a specific call id/number → call_by_id.\n"
        "4) Else (narrow Q&A) → topk_only.\n"
        "5) Only return JSON. No extra text."
    )
    user = f"User query: {query}"

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role":"system","content":system},
                      {"role":"user","content":user}],
            temperature=0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        json_txt = _extract_json_block(raw)
        data = json.loads(json_txt)
        # Ensure schema presence
        scope = data.get("scope", "topk_only")
        return {
            "scope": scope,
            "call_id": data.get("call_id"),
            "reason": data.get("reason", "LLM-scope"),
        }
    except Exception:
        return {
            "scope": "topk_only",
            "call_id": None,
            "reason": None,
        }