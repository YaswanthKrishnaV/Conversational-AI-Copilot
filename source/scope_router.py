from __future__ import annotations
import json, re
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
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


def scope_route(query: str, files: List, router_prompt= Path("data/sys_message/router_scope_prompt.txt")) -> Dict[str, Any]:
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
    
    prompt = f"""You choose the retrieval scope for a RAG system over SALES CALL TRANSCRIPTS.
List of all files/ call ids - {', '.join(files)}.
Return ONLY JSON with fields: { "scope", "call_id", "topic", "reason" } and nothing else."""

    with open(router_prompt, "r", encoding="utf-8") as f:
        sys_message = prompt + "\n\n" + f.read()
    user = f"User query: {query}"

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role":"system","content":sys_message},
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