import os
from openai import AzureOpenAI
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
from typing import List, Dict

def get_azure_openai_client() -> AzureOpenAI:
    azure_endpoint = os.getenv("OPENAI_ENDPOINT")
    if azure_endpoint is None:
        print("Please set the environment variable OPENAI_ENDPOINT")
        exit(1)

    api_key = os.getenv("OPENAI_API_KEY")
    
    # this is needed if you use Kantar's models
    api_version = os.getenv("OPENAI_API_VERSION")
    if api_key is None:
        print("Please set the environment variable AZURE_OPENAI_API_KEY")
        exit(1)

    # Initialize the Azure OpenAI client
    oai_client = AzureOpenAI(
        azure_endpoint=azure_endpoint, api_key=api_key, api_version=api_version
    )

    return oai_client

def format_context(chunks: List[Dict]) -> str:
    lines = []
    for c in chunks:
        head = f"[{c['rank']}] {c['file']} (chunk {c['chunk_id']}, score={c['score']:.3f})"
        lines.append(head)
        lines.append(c["text"])
        lines.append("")  # blank
    return "\n".join(lines)

def ask_llm(query, chunks, call_ids, sys_message= Path("data/sys_message/context_query_prompt.txt")) -> str:

    context = format_context(chunks)

    with open(sys_message, "r", encoding="utf-8") as f:
        sys_message = f.read()

    prompt =f'''
       {sys_message}
       The files consist of transcripts from calls with the following IDs: {', '.join(call_ids)}.

        ### Context\n{context}\n\n
        ### Question\n{query}\n\n
        ### Answer:
    '''

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        
        return "No API key configured. Would answer using the retrieved context."
    

    try:
         # Initialize the Azure OpenAI client
        oai_client = get_azure_openai_client()

        resp = oai_client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL"),
            messages=[
                {"role": "system", "content": "Answer strictly from the provided context."},
                {"role": "user", "content": prompt},
            ],
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(stub) LLM call failed: {e}. Using retrieved context only."
   
