
from pathlib import Path
import chromadb

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document


# Configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DB_PATH = "chroma_db"
COLLECTION_NAME = "education_documents"


def load_embedding_model():
    """Load the Hugging Face embedding model."""

    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    print("Embedding model loaded successfully!")

    return embeddings


def create_vector_store(chunks):
    """Generate embeddings and store them in ChromaDB."""

    if not chunks:
        raise ValueError("No chunks provided for embedding.")

    embeddings = load_embedding_model()

    print("Connecting to ChromaDB...")

    Path(DB_PATH).mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=DB_PATH)

    # Recreate our project collection for a fresh indexing run
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    vector_store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )

    documents = [
        Document(
            page_content=chunk["text"],
            metadata=chunk["metadata"]
        )
        for chunk in chunks
    ]

    print("Generating embeddings and storing documents...")

    vector_store.add_documents(documents)

    print(f"Documents stored: {len(documents)}")

    return vector_store


def load_vector_store():
    """Load the existing ChromaDB vector store."""

    embeddings = load_embedding_model()

    client = chromadb.PersistentClient(path=DB_PATH)

    vector_store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )

    return vector_store


def search_documents(query, vector_store, k=5):
    """Retrieve relevant document chunks."""

    results = vector_store.similarity_search(
        query,
        k=k
    )

    return results


if __name__ == "__main__":

    from preprocessing import process_pdf

    pdf_path = "dataset/documents/sample.pdf"

    try:
        chunks = process_pdf(pdf_path)

        vector_store = create_vector_store(chunks)

        print("\nTesting semantic search...")

        query = "What is artificial intelligence in education?"

        results = search_documents(query, vector_store)

        for index, result in enumerate(results, start=1):
            print(f"\nResult {index}:")
            print("Content:", result.page_content[:300])
            print("Metadata:", result.metadata)

        print("\nEmbedding and vector storage completed!")

    except Exception as e:
        print(f"Error: {e}")
