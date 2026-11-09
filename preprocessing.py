
import re
from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def extract_text_from_pdf(pdf_path):
    """Extract text from all pages of a PDF."""

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        cleaned_text = clean_text(text)

        if cleaned_text.strip():
            pages.append({
                "text": cleaned_text,
                "page": page_number,
                "source": pdf_path.name
            })

    if not pages:
        raise ValueError(
            "No readable text found in the PDF. "
            "The document may contain scanned images."
        )

    return pages


def clean_text(text):
    """Clean unnecessary spaces and line breaks."""

    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\S ]+', ' ', text)

    return text.strip()


def split_text_into_chunks(
    pages,
    chunk_size=800,
    chunk_overlap=150
):
    """Split extracted PDF text into smaller chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []

    for page in pages:
        page_chunks = splitter.split_text(page["text"])

        for chunk in page_chunks:
            chunks.append({
                "text": chunk,
                "metadata": {
                    "source": page["source"],
                    "page": page["page"]
                }
            })

    return chunks


def process_pdf(pdf_path):
    """Complete PDF preprocessing pipeline."""

    print("Extracting PDF text...")

    pages = extract_text_from_pdf(pdf_path)

    print(f"Pages extracted: {len(pages)}")

    print("Splitting text into chunks...")

    chunks = split_text_into_chunks(pages)

    print(f"Total chunks generated: {len(chunks)}")

    return chunks


if __name__ == "__main__":

    pdf_path = Path("dataset/documents/sample.pdf")

    try:
        chunks = process_pdf(pdf_path)

        print("\nPreprocessing completed successfully!")

        if chunks:
            print("\nFirst chunk:")
            print(chunks[0]["text"][:500])

            print("\nMetadata:")
            print(chunks[0]["metadata"])

    except Exception as e:
        print(f"Error: {e}")
