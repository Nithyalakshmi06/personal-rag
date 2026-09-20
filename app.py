from retrieval.retriever import Retriever
from generation.generator import Generator


def main():
    print("=" * 60)
    print("PERSONAL RAG ASSISTANT")
    print("=" * 60)

    # Load retriever
    retriever = Retriever()

    # Load Gemini generator
    generator = Generator()

    while True:
        question = input("\nAsk a question about your resume: ")

        if question.lower() in ["exit", "quit"]:
            print("\nGoodbye!")
            break

        # Retrieve relevant chunks
        results = retriever.search(question, top_k=3)

        # Combine retrieved chunks into context
        context = "\n\n".join(
            result["chunk"] for result in results
        )

        # Generate answer using Gemini
        answer = generator.generate_answer(
            question,
            context
        )

        print("\n" + "=" * 60)
        print("ANSWER")
        print("=" * 60)
        print(answer)


if __name__ == "__main__":
    main()