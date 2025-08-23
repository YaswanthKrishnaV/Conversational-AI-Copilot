from __future__ import annotations
import os
import re
import json
from typing import Any, Dict, List, Optional, Tuple
from llm import get_azure_openai_client
from pydantic import BaseModel
from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner, OpenAIChatCompletionsModel
from agents.exceptions import InputGuardrailTripwireTriggered
from openai import AsyncAzureOpenAI

# ==== Azure OpenAI client & model ==========================================
client = AsyncAzureOpenAI(
            azure_endpoint = os.getenv("OPENAI_ENDPOINT"),
            api_key = os.getenv("OPENAI_API_KEY"), 
            api_version = os.getenv("OPENAI_API_VERSION")
        )

model = OpenAIChatCompletionsModel(
    openai_client=client,
    model= os.getenv("OPENAI_MODEL")
)

# ==== Typed outputs =======================================================
class RouteOutput(BaseModel):
    label: str                  # 'ingest' | 'qna' | 'summarize' | 'irrelevant'
    reason: str
    paths: List[str] = []
    needs_path: bool = False

class DomainCheckOutput(BaseModel):
    in_domain: bool
    reason: str

# ==== Helpers =============================================================
PATH_RE = r'([A-Za-z]:\\[^:*?"<>|\r\n]+\.txt|\/[^:*?"<>|\r\n]+\.txt|(?:\.\.?\/)?[^:*?"<>|\r\n]+\.txt)'

def extract_paths(text: str) -> List[str]:
    return re.findall(PATH_RE, text or "")

def router_system_prompt() -> str:
    return (
        "You are an intent router for a Conversational AI Copilot that works ONLY with sales call transcripts.\n"
        "Classify each user query into exactly one of:\n"
        '- "ingest": user wants to add/load/upload/import/index transcript files into the KB (often with a .txt path).\n'
        '- "qna": user asks for information/insights from the transcripts (participants, objections, pricing, competitor mentions, summaries, next steps, etc.).\n'
        '- "summarize": the user asks for a summary of one/many calls (high-level recap, action items).\n'
        '- "irrelevant": query is completely outside sales/transcripts (sports, weather, movies, generic coding, etc.).\n\n'
        "Rules:\n"
        "1) If query mentions adding/uploading/loading transcripts OR contains a .txt path → label 'ingest'. "
        "   If no path is given, set needs_path=true.\n"
        "2) Else if the query asks about transcript content → label 'qna'.\n"
        "3) Else if the query asks for a summary of one/multiple calls → label 'summarize'.\n"
        "4) Else → 'irrelevant'.\n\n"
        "Output ONLY this JSON:\n"
        '{ "label": "ingest" | "qna" | "summarize" | "irrelevant", '
        '"reason": "<short>", "paths": ["<.txt paths if any>"], "needs_path": true | false }'
    )

def answer_system_prompt() -> str:
    return (
        "You are a helpful assistant for sales call transcripts.\n"
        "Use ONLY the provided context to answer. If the answer is not present, reply: "
        '"I don’t know based on the transcripts." Be concise and structured. '
        "Lists/bullets are OK. Do not hallucinate."
    )

# ==== Guardrail: in-domain? ==============================================
domain_guardrail_agent = Agent(
    name="Domain Guardrail",
    model=model,
    output_type=DomainCheckOutput,
    instructions=(
        "Decide if the user message is about SALES CALL TRANSCRIPTS (ingest or QnA). "
        "Return in_domain=true only if it clearly concerns transcripts/calls. "
        "Otherwise in_domain=false with a short reason."
    ),
)

async def domain_guardrail(ctx, agent, input_data):
    r = await Runner.run(domain_guardrail_agent, input_data, context=ctx.context)
    out = r.final_output_as(DomainCheckOutput)
    return GuardrailFunctionOutput(output_info=out, tripwire_triggered=not out.in_domain)

# ==== Specialists (prompt-only; tools handled in orchestrator) ============
ingest_specialist = Agent(
    name="Ingest Specialist",
    model=model,
    instructions=(
        "You assist with ingestion requests. If the user wants to add/load/upload/import/index transcript files into the KB (often with a .txt path) acknowledge them."
        "If not, ask them to provide a valid .txt path."
    ),
)

qna_specialist = Agent(
    name="QnA Specialist",
    model=model,
    handoff_description="Answers questions strictly from provided transcript context. user asks for information/insights/comments from the transcripts (participants, objections, summaries, pricing, competitor mentions, etc.).",
    instructions=answer_system_prompt()
)

summarizer_specialist = Agent(
    name="Summarizer Specialist",
    model=model,
    handoff_description="Summarizes transcript chunks clearly and concisely.",
    instructions="Summarize the provided transcript context into a short, structured recap with action items."
)

# ==== Triage Agent ========================================================
triage_agent = Agent(
    name="Triage Agent",
    model=model,
    instructions=(
        "Classify the user query into exactly one: ingest, qna, summarize, irrelevant. "
        "If ingest is chosen but no path is present, set needs_path=true. "
        "Return ONLY JSON: {label, reason, paths:[...], needs_path:bool}."
    ),
    handoffs=[ingest_specialist, qna_specialist, summarizer_specialist],
    input_guardrails=[InputGuardrail(guardrail_function=domain_guardrail)],
)

# ==== Orchestrator (wires LLM agents to the tools) ======================
from ingest import ingest as run_ingest
from retriever import load_index_and_mapping, retrieve_relevant_chunks
from llm import ask_llm

class AgentContextState:
    def __init__(self):
        self.index = None
        self.mapping = None
        self.emb_model = None
    def load_or_reload(self):
        self.index, self.mapping, self.emb_model = load_index_and_mapping()

class Orchestrator:
    def __init__(self, top_k: int = 4):
        self.ctx = AgentContextState()
        self.top_k = top_k
        try:
            self.ctx.load_or_reload()
        except Exception:
            pass  # will load after first ingest

    async def handle(self, user_query: str) -> Dict[str, Any]:
        try:
            routed = await Runner.run(triage_agent, user_query)
        except InputGuardrailTripwireTriggered as e:
            return {"type": "irrelevant", "message": f"Out of scope: {e}. This copilot only handles queries related to sales call transcripts."}

        # Parse route
        try:
            route = routed.final_output_as(RouteOutput)
        except Exception:
            # try to coerce JSON when model returned raw text
            raw = getattr(routed, "final_output", None)
            try:
                parsed = json.loads(raw) if raw else {}
                route = RouteOutput(**parsed)
            except Exception:
                route = RouteOutput(label="irrelevant", reason="Router returned non-JSON")

        label, paths = route.label, route.paths

        # Ingest
        if label == "ingest":
            if route.needs_path or not paths:
                return {"type":"error","message":"I detected an ingest request, but no .txt path was provided. Please include a valid path."}
            run_ingest(paths=paths, append=True)
            self.ctx.load_or_reload()
            return {"type":"ingest_ok","message":f"Ingested {len(paths)} file(s) and updated the index."}

        # QnA
        if label == "qna":
            if self.ctx.index is None:
                try:
                    self.ctx.load_or_reload()
                except Exception:
                    return {"type":"error","message":"No index found. Please ingest transcripts first."}
            chunks = retrieve_relevant_chunks(user_query, self.ctx.index, self.ctx.mapping, self.ctx.emb_model, k=self.top_k)
            answer = ask_llm(user_query, chunks)
            return {"type":"answer","answer":answer,"chunks":chunks}

        # Summarize
        if label == "summarize":
            if self.ctx.index is None:
                try:
                    self.ctx.load_or_reload()
                except Exception:
                    return {"type":"error","message":"No index found. Please ingest transcripts first."}
            chunks = retrieve_relevant_chunks(user_query, self.ctx.index, self.ctx.mapping, self.ctx.emb_model, k=max(self.top_k,6))
            # Reuse your ask_llm summarization style by asking for a summary explicitly
            summary_q = f"Summarize the following context:\n\n{user_query}"
            answer = ask_llm(summary_q, chunks)
            return {"type":"answer","answer":answer,"chunks":chunks}

        # Irrelevant
        return {"type":"irrelevant","message":f"This request isn't related to sales call transcripts: {route.reason}"}
