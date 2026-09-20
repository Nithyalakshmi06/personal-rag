from pathlib import Path
import pickle

import faiss

from ingestion.loader import load_document
from ingestion.chunker import clean_text, create_chunks
from embeddings.embedder import Embedder


class FAISSVectorStore:

    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []

    def add_embeddings(self, embeddings, chunks):
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def save(self, index_path, chunks_path):

        faiss.write_index(
            self.index,
            str(index_path)
        )

        with open(chunks_path, "wb") as file:
            pickle.dump(self.chunks, file)

        print("FAISS index saved!")
        print(f"Index: {index_path}")
        print(f"Chunks: {chunks_path}")


if __name__ == "__main__":

    print("=" * 60)
    print("STARTING FAISS VECTOR STORE")
    print("=" * 60)

    # 1. Load resume
    file_path = Path(
        "data/documents/Nithyalakshmi_K__Resume.docx"
    )

    print("\nLoading document...")

    text = load_document(file_path)

    # 2. Clean text
    print("Cleaning text...")

    cleaned_text = clean_text(text)

    # 3. Create chunks
    print("Creating chunks...")

    chunks = create_chunks(
        cleaned_text,
        chunk_size=500,
        overlap=50
    )

    print(f"Number of chunks: {len(chunks)}")

    # 4. Create embeddings
    print("\nCreating embeddings...")

    embedder = Embedder()

    embeddings = embedder.create_embeddings(chunks)

    print(f"Embedding shape: {embeddings.shape}")

    # 5. Create FAISS index
    print("\nCreating FAISS index...")

    dimension = embeddings.shape[1]

    vector_store = FAISSVectorStore(dimension)

    vector_store.add_embeddings(
        embeddings,
        chunks
    )

    print(
        f"Vectors stored: {vector_store.index.ntotal}"
    )

    # 6. Create processed folder
    processed_dir = Path("data/processed")

    processed_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # 7. Save files
    index_path = processed_dir / "index.faiss"

    chunks_path = processed_dir / "chunks.pkl"

    vector_store.save(
        index_path,
        chunks_path
    )

    print("\n" + "=" * 60)
    print("FAISS VECTOR STORE CREATED SUCCESSFULLY")
    print("=" * 60)