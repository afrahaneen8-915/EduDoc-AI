# EduDoc AI – Education Document Question Answering System

## 1. Project Overview
EduDoc AI is an AI-powered document question-answering application developed using Large Language Models (LLMs), LangChain, and Retrieval-Augmented Generation (RAG).

It allows users to upload educational PDF documents and ask questions based on their content. The system retrieves relevant information and generates answers using a locally running LLM.

## 2. Objectives
- Develop an interactive LLM-based application.
- Extract and process text from PDF documents.
- Generate document embeddings.
- Store and retrieve information using ChromaDB.
- Generate context-based answers using RAG.
- Provide a user-friendly interface using Streamlit.

## 3. Technologies Used
- Python 3.12
- LangChain
- Ollama
- Llama 3.2
- ChromaDB
- Sentence Transformers
- Streamlit
- PyPDF

## 4. Main Features
- PDF document upload.
- Text extraction and preprocessing.
- Text chunking and embedding generation.
- Vector database storage.
- Semantic document search.
- Context-based question answering.
- Chat history storage.
- CSV download options.
- Interactive Streamlit interface.

## 5. Project Workflow
1. Upload an educational PDF.
2. Extract text from the PDF.
3. Split the extracted text into chunks.
4. Generate embeddings.
5. Store embeddings in ChromaDB.
6. Retrieve relevant chunks based on the user's question.
7. Pass the retrieved context to the LLM.
8. Generate and display the answer.

## 6. Project Modules
- preprocessing.py – Extracts and processes PDF text.
- embedding.py – Generates embeddings and manages the vector database.
- rag_pipeline.py – Implements the RAG workflow.
- chatbot.py – Generates answers using the Ollama LLM.
- app.py – Provides the Streamlit user interface.

## 7. Output Files
- dataset/chat_history.csv
- dataset/answers.csv
-  assets/screenshots/ – Application screenshots.

## 8. How to Run the Project
1. Install the required Python packages.
2. Ensure Ollama is running with the llama3.2 model.
3. Open the project folder in VS Code.
4. Open the terminal.
5. Execute:

streamlit run app.py

6. Open the Local URL displayed in the terminal.

## 9. Conclusion
EduDoc AI demonstrates how Large Language Models and Retrieval-Augmented Generation can be combined to create an intelligent document question-answering system. It helps users retrieve relevant educational information from PDF documents through a simple interactive application.