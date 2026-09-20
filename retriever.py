from pathlib import Path
import pickle

import faiss

from embeddings.embedder import Embedder


class Retriever:

    def __init__(
        self,
        index_path="data/processed/index.faiss",
        chunks_path="data/processed/chunks.pkl"
    ):

        print("Loading FAISS index...")

        self.index = faiss.read_index(
            str(index_path)
        )

        print("FAISS index loaded.")

        print("Loading chunks...")

        with open(chunks_path, "rb") as file:
            self.chunks = pickle.load(file)

        print(f"Loaded {len(self.chunks)} chunks.")

        print("Loading embedding model...")

        self.embedder = Embedder()

    def search(self, query, top_k=3):
        """
        Search for the most relevant chunks.
        """

        # Convert question into embedding
        query_embedding = self.embedder.create_embeddings(
            [query]
        )

        # Search FAISS
        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for distance, index in zip(
            distances[0],
            indices[0]
        ):

            if index == -1:
                continue

            results.append({
                "chunk": self.chunks[index],
                "distance": float(distance)
            })

        return results


if __name__ == "__main__":

    print("=" * 60)
    print("PERSONAL RAG RETRIEVER")
    print("=" * 60)

    retriever = Retriever()

    query = input(
        "\nAsk a question about your resume: "
    )

    results = retriever.search(
        query,
        top_k=3
    )

    print("\n" + "=" * 60)
    print("RETRIEVED RESULTS")
    print("=" * 60)

    for i, result in enumerate(results):

        print(f"\nRESULT {i + 1}")
        print("-" * 60)

        print("Distance:", result["distance"])

        print("\nChunk:")
        print(result["chunk"])