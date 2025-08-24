from __future__ import annotations
import json, re
from typing import Any, Dict, List, Optional, Tuple

from source.llm import get_azure_openai_client
from source.planner import _extract_json_block

def extract_topic_llm(query: str) -> Optional[str]:
    """
    Use LLM to extract the SINGLE main topic from a user query as a short keyword/phrase.
    Return None if not confident.
    """
    client, model_name = get_azure_openai_client()
    
    system = (
        "You extract the SINGLE main TOPIC from a user query about sales call transcripts.\n"
        "Return ONLY JSON: {\"topic\": \"<short phrase>\"}.\n"
        "Keep it concise (1–3 words), lowercase, no punctuation."
    )
    user = f"Query: {query}"

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0
        )
        raw = (resp.choices[0].message.content or "").strip()
        json_txt = _extract_json_block(raw)

        data = json.loads(json_txt)

        topic = (data.get("topic") or "").strip().lower()
        return topic or None
    except Exception:
        return {"scope": "topk_only", "call_id": None, "topic": None, "reason": "Default"}


def scope_route(query: str, files: List) -> Dict[str, Any]:
    """
    Ask the LLM to pick a retrieval scope. Returns a dict like:
    {
      "scope": "topk_only" | "last_call" | "full_call_of_top_hit" | "call_by_id" | "topic_across_calls" |  "all_calls",
      "call_id": "<optional>",
      "topic": "<optional>",
      "reason": "<short>"
    }
    
    """
    client, model_name= get_azure_openai_client()
    
    system = f"""You choose the retrieval scope for a RAG system over SALES CALL TRANSCRIPTS.
List of all files/ call ids - {', '.join(files)}.
Return ONLY JSON with fields: { "scope", "call_id", "topic", "reason" } and nothing else.

Scopes:
- "topk_only": default for narrow Q&A.
- "full_call_of_top_hit": use ALL chunks from the call that contains the top-1 retrieved chunk.
- "last_call": use ALL chunks from the most recent call.
- "call_by_id": use ALL chunks from a specific call the user mentions.
- "topic_across_calls": collect ALL chunks relevant to the user's TOPIC across ALL calls.
- "all_calls": aggregate across every call (rare).

Rules:
1) If the user asks what was discussed/mentioned about a TOPIC across/all calls → scope="topic_across_calls".
   - Extract a concise topic (1-3 words, lowercase) into "topic" (e.g., "pricing", "discounts", "security").
2) If the user asks to "summarize/recap/highlight key takeaways from the last/latest call" → scope="last_call".
3) If the user asks to "summarize/recap/highlight the full/entire call" without naming a call → scope="full_call_of_top_hit".
4) If the user names a call explicitly (e.g., "negotiation call", "demo call", "objection call") or says "call N" asking what was discussed/mentioned about a TOPIC in this call :
   → scope="call_by_id" and set "call_id" to a normalized identifier.
   Normalization guidance (use whichever applies):
     - If a list of known call_ids is provided in context, fuzzy-match the best one (substring match).
     - Else, derive from the phrase (e.g., "negotiation call" → "negotiation_call").
     - If "call N" is used, set call_id to the id that ends with N (e.g., "…_call_4" → "…4"), or just "call_4" or "4_.
5) If none of the above clearly applies → scope="topk_only".
6) Always include a short "reason" explaining the routing decision.
7) Output ONLY JSON. No prose, no backticks."""
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
        scope = data.get("scope", "topk_only")
        call_id = data.get("call_id")
        topic = data.get("topic")
        # If scope is topic_across_calls but topic is missing/empty, ask the LLM to extract it.
        if scope == "topic_across_calls" and (not topic or not topic.strip()):
            topic = extract_topic_llm(query)

        return {
            "scope": scope,
            "call_id": call_id,
            "topic": topic,
            "reason": data.get("reason", "LLM-scope"),
        }
    except Exception:
        return {
            "scope": "topk_only",
            "call_id": None,
            "topic": None,
            "reason": None,
        }