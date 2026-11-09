
from preprocessing import process_pdf
from embedding import create_vector_store, search_documents


class RAGPipeline:

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.vector_store = None
        self.chunks = []

    def initialize(self):
        """Preprocess PDF and create vector database."""

        print("Initializing RAG pipeline...")

        self.chunks = process_pdf(self.pdf_path)

        self.vector_store = create_vector_store(self.chunks)

        print("RAG pipeline initialized successfully!")

    def retrieve(self, query, k=5):
        """Retrieve relevant document chunks."""

        if self.vector_store is None:
            raise RuntimeError(
                "RAG pipeline is not initialized."
            )

        results = search_documents(
            query,
            self.vector_store,
            k=k
        )

        return results

    def get_context(self, query, k=5):
        """Combine retrieved chunks into context."""

        documents = self.retrieve(query, k)

        context = "\n\n".join(
            [
                f"[Source: {doc.metadata.get('source', 'Unknown')}, "
                f"Page: {doc.metadata.get('page', 'Unknown')}]\n"
                f"{doc.page_content}"
                for doc in documents
            ]
        )

        return context, documents


if __name__ == "__main__":

    try:
        rag = RAGPipeline(
            "dataset/documents/sample.pdf"
        )

        rag.initialize()

        question = "What is artificial intelligence in education?"

        print("\nTesting document retrieval...")

        context, documents = rag.get_context(question)

        print(f"\nRetrieved documents: {len(documents)}")
        print("\nRetrieved context:")
        print(context[:2000])

        print("\nRAG pipeline test completed!")

    except Exception as e:
        print(f"Error: {e}")
