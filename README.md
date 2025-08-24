# Conversational AI Copilot (RAG over Sales Call Transcripts)

This repository implements a **Retrieval-Augmented Generation (RAG)** system tailored for **sales call transcripts**.
It allows you to ingest transcript files, chunk them with metadata, embed them into a FAISS index, and query them via an **LLM-powered copilot**.

# The system supports:

- Narrow Q&A with top-k retrieval
- Summaries of full calls
- “Last call” summaries (synthetic date assignment)
- Topic-based retrieval across all calls
- Append ingestion of new transcripts without rebuilding the index
- Fuzzy matching of call IDs ("summarize negotiation call" → 4_negotiation_call)

# Assumptions

- **Transcript dates**: synthetic consecutive days (mtime order). Last modified → “yesterday”.
- **Embeddings**: we use **all-MiniLM-L6-v2** for speed/accuracy tradeoff. Could swap to larger HF models for slightly better accuracy but heavier compute or use Proprietary models like GPT or Gemini accounting for API cost.
- **Chunk size**: ~500 chars chunk_size (~200–250 tokens, ~3-6 dialogue turns), overlap ~50 ((≈1 turn)). Tuned via EDA (avg dialogue ~90 chars, ~25 tokens ~15 words).
- **Index storage**: in data/index/. Reloaded into memory on ingest.
- **LLM**: Azure OpenAI or OpenAI. If no API key
- **Out-of-scope queries**: e.g., “tell me a joke” are blocked or routed as irrelevant.

# Folder Structure

rag_copilot/
│
├── data/
│   ├── transcripts/        # Input raw transcripts (.txt files)
│   │   ├── 1_demo_call.txt
│   │   ├── 2_objection_call.txt
│   │   └── ...
│   └── index/              # Auto-generated FAISS index + mapping
│       ├── vector.index
│       └── docs_mapping.pkl
│
├── source/
│   ├── ingest.py           # Transcript ingestion & FAISS indexing
│   ├── retriever.py        # Chunk retrieval by query or topic
│   ├── retriever_helpers.py# Helpers (fuzzy call-id matching, etc.)
│   ├── llm.py              # LLM wrappers (Azure OpenAI / OpenAI)
│   ├── planner.py          # Query planner / scope router
│   └── scope_router.py     # Prompt-based intent routing
│
├── cli.py                  # Main CLI entrypoint
│
├── tests/                  # Unit tests
│   ├── test_llm.py
│   ├── test_cli_smoke.py
│   ├── test_ingest_retriever.py
│   └── ...
│
├── .gitignore              # avoids pushing unwanted files or test notebooks
├── .env.sample             # sample Environment variables
├── requirements.txt        # Python dependencies
└── README.md               # This file


# 1. Setup

git https://github.com/YaswanthKrishnaV/Conversational-AI-Copilot.git
cd Conversational-AI-Copilot

# 2. Create a Python environment

using venv:

    python3.10 -m venv .venv
    source .venv/bin/activate   # (Linux/macOS)
    .venv\Scripts\activate      # (Windows)

or using conda:

    conda create -n rag_copilot python=3.10
    conda activate rag_copilot

# 3. Install dependencies
pip install -r requirements.txt

Create a **.env file** in the repo root (or export manually). Use **.env.sample** as reference

# 4. Running the CLI

(Take a few seconds to start in the beginning as it has to obtain index and meta data and then uses lazy loading for any request)

Start the interactive chatbot by running :

**python cli.py** 

Example: 

Conversational AI Copilot - Please ask your Query (type 'exit' to quit)

You: summarize negotiation call
AI: Key takeaways from negotiation call...
Top sources:
  [1] 4_negotiation_call.txt #0  00:01→00:04  speakers=AE, Prospect



# Testing

We use pytest with temp directories and stubs (no real API calls).

Run: **pytest -q**
