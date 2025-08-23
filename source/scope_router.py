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


def scope_route(query: str) -> Dict[str, Any]:
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
    
    system = (
        "You choose the retrieval scope for a RAG system over SALES CALL TRANSCRIPTS.\n"
        "Return ONLY JSON with fields: scope, call_id, topic, reason.\n\n"
        "Scopes:\n"
        "- topk_only: default for narrow Q&A.\n"
        "- full_call_of_top_hit: summarize the entire call of the top-1 retrieved chunk.\n"
        "- last_call: summarize the most recent call.\n"
        "- call_by_id: summarize a specific call id if user mentions it.\n"
        "- topic_across_calls: collect all chunks about the user's TOPIC across ALL calls.\n"
        "- all_calls: aggregate across every call.\n\n"
        "Rules:\n"
        "1) If user asks what was discussed/mentioned about a topic ACROSS/ALL calls → topic_across_calls.\n"
        "2) If 'summarize the last call' → last_call.\n"
        "3) If 'summarize the full/entire call' (no call specified) → full_call_of_top_hit.\n"
        "4) If a specific call id is mentioned → call_by_id.\n"
        "5) Else → topk_only.\n"
        "6) Only JSON. No prose."
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