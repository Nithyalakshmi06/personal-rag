from pathlib import Path
from pypdf import PdfReader
from docx import Document


def load_pdf(file_path):
    """Extract text from a PDF file."""

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def load_docx(file_path):
    """Extract text from a DOCX file."""

    document = Document(file_path)

    text = ""

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            text += paragraph.text + "\n"

    return text


def load_document(file_path):
    """Load a document based on its file type."""

    extension = file_path.suffix.lower()

    if extension == ".pdf":
        return load_pdf(file_path)

    elif extension == ".docx":
        return load_docx(file_path)

    else:
        raise ValueError(f"Unsupported file type: {extension}")


if __name__ == "__main__":

    file_path = Path(
        "E:/anti/personal RAG/data/documents/Nithyalakshmi_K__Resume.docx"
    )

    print(f"Loading: {file_path}")

    text = load_document(file_path)

    print("=" * 60)
    print("EXTRACTED TEXT")
    print("=" * 60)

    print(text)