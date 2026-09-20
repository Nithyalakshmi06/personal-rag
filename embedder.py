from sentence_transformers import SentenceTransformer


class Embedder:

    def __init__(self):
        print("Loading embedding model...")

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        print("Embedding model loaded!")

    def create_embeddings(self, texts):

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True
        )

        return embeddings


if __name__ == "__main__":

    print("Starting embedding test...")

    embedder = Embedder()

    sample_texts = [
        "I am studying Artificial Intelligence and Data Science.",
        "I am interested in Data Engineering.",
        "I worked on an E-Commerce ETL pipeline."
    ]

    embeddings = embedder.create_embeddings(sample_texts)

    print("\n" + "=" * 60)
    print("EMBEDDING TEST SUCCESSFUL")
    print("=" * 60)

    print("Number of texts:", len(sample_texts))
    print("Embedding shape:", embeddings.shape)

    print("\nFirst embedding:")
    print(embeddings[0])