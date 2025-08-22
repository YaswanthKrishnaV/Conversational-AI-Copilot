from retriever import retrieve_relevant_chunks
from llm import ask_llm

index = None  # don’t load at startup


def cli():
    global index
    print("Conversational AI Copilot (type 'exit' to quit)")

    while True:
        q = input("You: ")

        if q.lower() in ["exit", "quit"]:
            break

        # Only load index when first needed
        if index is None:
            from ingest import ingest
            print("Loading knowledge base... (only once)")
            index = ingest()

        chunks = retrieve_relevant_chunks(q, index)
        answer = ask_llm(q, chunks)
        print(f"AI: {answer}\nSources: {chunks}")

if __name__ == "__main__":
    cli()
