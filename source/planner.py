# app/planner.py
from __future__ import annotations
from llm import get_azure_openai_client
from pathlib import Path
import os
import re
import json
from typing import Any, Dict, List, Optional

def _ensure_schema(d: dict) -> dict:
    """Guarantee required keys exist with sane defaults."""
    return {
        "label": d.get("label", "irrelevant"),
        "reason": d.get("reason", "Fallback router"),
        "paths": d.get("paths", []) or [],
        "needs_path": bool(d.get("needs_path", False)),
    }

def _extract_json_block(text: str) -> str | None:
    """
    Try to extract a JSON object from messy LLM output.
    Handles code fences and leading/trailing prose.
    """
    # strip code fences if any
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    # if it's already valid json
    try:
        json.loads(text)
        return text
    except Exception:
        pass

     # fallback: grab first {...} block
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        return m.group(0).strip()
    return None


def extract_paths(query: str) -> List[str]:
    """
    Extract .txt paths from a free-form message (Windows/Unix/relative).
    """
    pattern = r'([A-Za-z]:\\[^:*?"<>|\r\n]+\.txt|\/[^:*?"<>|\r\n]+\.txt|(?:\.\.?\/)?[^:*?"<>|\r\n]+\.txt)'
    return re.findall(pattern, query)


def llm_route(query: str, sys_message = Path("data/sys_message/planner_identifier_prompt.txt")) -> str:
    """
    LLM-based router. Returns one of: 'ingest' | 'qna' | 'irrelevant'.
    Falls back to heuristic if OpenAI is unavailable or no API key is set.
    """

    client = get_azure_openai_client()

    with open(sys_message, "r", encoding="utf-8") as f:
        sys_message = f.read()
    
    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=[{"role": "system", "content": sys_message},
                      {"role": "user", "content": query}],
            temperature=0
        )
        raw = resp.choices[0].message.content.strip()
        json_txt = _extract_json_block(raw)
        
        parsed = json.loads(json_txt)
        return _ensure_schema(parsed)
    except Exception as e:
        return _ensure_schema({})


class Planner:
    """
    Produces an executable plan (list of steps) for a given user query.
    Steps are dicts with an 'action' key and optional 'args'.
    """
    def __init__(self, top_k: int = 4):
        self.top_k = top_k

    def plan(self, query: str, retrieval_probe: Optional[callable] = None) -> List[Dict[str, Any]]:
        steps: List[Dict[str, Any]] = []

        # 1) Route intent
        result = llm_route(query)
        intent = result.get("label")
        reason = result.get("reason", "")
        paths = result.get("paths", [])
        needs_path = result.get("needs_path", False)

        # 2) Branch on intent
        if intent == "ingest":
            if not paths and needs_path:
                return [{"action": "cannot_ingest", "reason": "Ingest intent but no file path provided."}]
            return [{"action": "ingest", "args": {"paths": paths}}]

        elif intent == "irrelevant":
            return [{"action": "irrelevant", "reason": reason}]
        
        elif intent == "qna":
            if retrieval_probe is not None:
                try:
                    support_hits = retrieval_probe(query)
                    if support_hits == 0:
                        return [{"action": "cannot_answer",
                                "reason": "No relevant context found in KB for this query."}]
                except Exception:
                    pass

            steps.append({"action": "retrieve", "args": {"query": query, "top_k": self.top_k}})
            steps.append({"action": "synthesize"})

        else:
            [{"action": "block", "reason": "Unable to classify query."}]    
        return steps
