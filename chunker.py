from pathlib import Path

from ingestion.loader import load_document


def clean_text(text):
    """
    Clean extracted document text.
    """

    # Replace multiple spaces/newlines with a single space
    text = " ".join(text.split())

    return text.strip()


def create_chunks(text, chunk_size=500, overlap=50):
    """
    Split text into overlapping chunks.
    """

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":

    file_path = Path(
        "data/documents/Nithyalakshmi_K__Resume.docx"
    )

    # Step 1: Load document
    text = load_document(file_path)

    print(f"Original characters: {len(text)}")

    # Step 2: Clean text
    cleaned_text = clean_text(text)

    print(f"Cleaned characters: {len(cleaned_text)}")

    # Step 3: Create chunks
    chunks = create_chunks(cleaned_text)

    print(f"Number of chunks: {len(chunks)}")

    print("=" * 60)
    print("DOCUMENT CHUNKS")
    print("=" * 60)

    for i, chunk in enumerate(chunks):

        print(f"\nCHUNK {i + 1}")
        print("-" * 60)
        print(chunk)