
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


class EducationChatbot:

    def __init__(self, model_name="llama3.2:latest"):

        print("Connecting to Ollama...")

        self.llm = ChatOllama(
            model=model_name,
            temperature=0.2
        )

        self.prompt = ChatPromptTemplate.from_template(
            """
            You are an intelligent Education AI Assistant.

            Answer the user's question using ONLY the
            information provided in the context.

            If the answer is not available in the context,
            clearly say that the uploaded document does
            not contain sufficient information.

            Provide clear, accurate and simple answers.

            Context:
            {context}

            Question:
            {question}

            Answer:
            """
        )

        print("Ollama connected successfully!")

    def generate_answer(self, question, context):

        """Generate an answer using retrieved context."""

        chain = self.prompt | self.llm

        response = chain.invoke({
            "context": context,
            "question": question
        })

        return response.content


if __name__ == "__main__":

    from rag_pipeline import RAGPipeline

    try:

        rag = RAGPipeline(
            "dataset/documents/sample.pdf"
        )

        rag.initialize()

        chatbot = EducationChatbot()

        question = "What is artificial intelligence in education?"

        print("\nQuestion:", question)

        context, documents = rag.get_context(question)

        answer = chatbot.generate_answer(
            question,
            context
        )

        print("\nGenerated Answer:")
        print(answer)

        print("\nSource Documents:")

        for doc in documents:
            print(
                f"{doc.metadata['source']} - "
                f"Page {doc.metadata['page']}"
            )

        print("\nChatbot test completed successfully!")

    except Exception as e:
        print(f"Error: {e}")

