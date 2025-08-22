from retriever import retrieve_relevant_chunks
from llm import ask_llm

def cli():
    print("Conversational AI Copilot (type 'exit' to quit)")
    while True:
        q = input("You: ")
        if q.lower() == "exit":
            break
        chunks = retrieve_relevant_chunks(q)
        answer = ask_llm(q, chunks)
        print(f"Ans: {answer}\nSources: {chunks}")

if __name__ == "__main__":
    cli()
