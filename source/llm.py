import os
from openai import AzureOpenAI
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()
from typing import List, Dict

def get_azure_openai_client() -> AzureOpenAI:

    az_key = os.getenv("OPENAI_API_KEY")
    if az_key:
        from openai import AzureOpenAI
        endpoint = os.getenv("OPENAI_ENDPOINT")
        api_ver = os.getenv("OPENAI_API_VERSION", "2024-05-01-preview")
        model_name = os.getenv("OPENAI_MODEL")
        # If any required Azure bits are missing, return None -> fallback
        if not endpoint or not model_name:
            return None, None
        client = AzureOpenAI(api_key=az_key, azure_endpoint=endpoint, api_version=api_ver)
        return client, model_name

    return None, None

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

    try:
         # Initialize the Azure OpenAI client
        oai_client,model_name = get_azure_openai_client()

        resp = oai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Answer strictly from the provided context."},
                {"role": "user", "content": prompt},
            ],
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(stub) LLM call failed: {e}. Using retrieved context only."
   
