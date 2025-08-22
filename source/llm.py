import os
from openai import AzureOpenAI
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


def ask_llm(query, chunks):
    api_key = os.getenv("OPENAI_API_KEY")
    endpoint = os.getenv("OPENAI_ENDPOINT")
    api_version = os.getenv("OPENAI_API_VERSION")
    model = os.getenv("OPENAI_MODEL")
    if api_key is None:
        print("Please set the environment variable AZURE_OPENAI_API_KEY")
        exit(1)

    # Initialize the Azure OpenAI client
    oai_client = AzureOpenAI(
        azure_endpoint=endpoint, api_key=api_key, api_version=api_version
    )
    context = pd.DataFrame(chunks).to_markdown(index=False)
    prompt = f"""
    Answer the question based on context.\n\n
    Context:\n{context}
    \n\nQuestion: {query}
    \nAnswer:
    """
    
    response = oai_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
