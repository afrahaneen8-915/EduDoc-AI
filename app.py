from pathlib import Path
from datetime import datetime
import csv
import shutil

import streamlit as st

from preprocessing import process_pdf
from embedding import create_vector_store, load_vector_store, search_documents
from chatbot import EducationChatbot

# -------------------- PATHS --------------------
BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "dataset" / "documents"
UPLOADS_DIR = BASE_DIR / "dataset" / "uploaded_files"
CHROMA_DIR = BASE_DIR / "chroma_db"
HISTORY_FILE = BASE_DIR / "dataset" / "chat_history.csv"
ANSWERS_FILE = BASE_DIR / "dataset" / "answers.csv"

for folder in (DOCUMENTS_DIR, UPLOADS_DIR, CHROMA_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="EduDoc AI",
    page_icon="🎓",
    layout="wide"
)

# -------------------- STYLES --------------------
dark_mode = st.sidebar.toggle("🌙 Dark mode", value=False)

if dark_mode:
    st.markdown(
        """
        <style>
        .stApp { background-color: #101820; color: #f4f6f8; }
        [data-testid="stSidebar"] { background-color: #18232d; }
        [data-testid="stChatMessage"] { border-radius: 12px; }
        </style>
        """,
        unsafe_allow_html=True
    )

st.title("🎓 EduDoc AI")
st.subheader("Education Document Question Answering System")
st.write(
    "Upload your study PDFs, build a searchable knowledge base, "
    "and ask questions using Retrieval-Augmented Generation (RAG)."
)

# -------------------- HELPERS --------------------
def save_csv_row(file_path, header, row):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not file_path.exists() or file_path.stat().st_size == 0

    with file_path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        if new_file:
            writer.writerow(header)
        writer.writerow(row)


def get_pdf_files():
    return sorted(DOCUMENTS_DIR.glob("*.pdf")) + sorted(UPLOADS_DIR.glob("*.pdf"))


@st.cache_resource(show_spinner=False)
def get_chatbot():
    return EducationChatbot(model_name="llama3.2:latest")


def build_index():
    pdf_files = get_pdf_files()
    if not pdf_files:
        raise ValueError(
            "No PDF documents found. Add PDF files to dataset/documents "
            "or upload PDFs using the sidebar."
        )

    all_chunks = []
    processed_files = []

    for pdf_file in pdf_files:
        chunks = process_pdf(str(pdf_file))
        all_chunks.extend(chunks)
        processed_files.append(pdf_file.name)

    if not all_chunks:
        raise ValueError("No readable text chunks were extracted from the PDFs.")

    # create_vector_store recreates the project's collection, then stores all
    # chunks together so all selected PDFs are searchable in one index.
    vector_store = create_vector_store(all_chunks)
    return vector_store, len(processed_files), len(all_chunks), processed_files


def load_existing_index():
    # Confirm that the expected Chroma collection already exists.
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    client.get_collection(name="education_documents")
    return load_vector_store()


def answer_question(question, vector_store):
    documents = search_documents(question, vector_store, k=5)

    context = "\n\n".join(
        [
            (
                f"[Source: {doc.metadata.get('source', 'Unknown')}, "
                f"Page: {doc.metadata.get('page', 'Unknown')}]\n"
                f"{doc.page_content}"
            )
            for doc in documents
        ]
    )

    if not context.strip():
        return (
            "The uploaded documents do not contain sufficient information "
            "to answer this question.",
            []
        )

    chatbot = get_chatbot()
    answer = chatbot.generate_answer(question, context)

    sources = []
    for doc in documents:
        source = (
            f"{doc.metadata.get('source', 'Unknown')} "
            f"- Page {doc.metadata.get('page', 'Unknown')}"
        )
        if source not in sources:
            sources.append(source)

    return answer, sources


# -------------------- SIDEBAR: DOCUMENT MANAGEMENT --------------------
st.sidebar.title("📚 Knowledge Base")

uploaded_pdfs = st.sidebar.file_uploader(
    "Upload one or more PDF documents",
    type=["pdf"],
    accept_multiple_files=True,
    help="You can upload multiple PDF study materials."
)

if st.sidebar.button("Save uploaded PDFs", use_container_width=True):
    if not uploaded_pdfs:
        st.sidebar.warning("Please select at least one PDF first.")
    else:
        saved_count = 0
        for uploaded_file in uploaded_pdfs:
            safe_name = Path(uploaded_file.name).name
            destination = UPLOADS_DIR / safe_name
            destination.write_bytes(uploaded_file.getvalue())
            saved_count += 1

        st.sidebar.success(f"Saved {saved_count} PDF file(s). Build the index next.")

st.sidebar.caption("PDF files currently available:")
available_pdfs = get_pdf_files()

if available_pdfs:
    for pdf in available_pdfs:
        st.sidebar.write(f"📄 {pdf.name}")
else:
    st.sidebar.info("No PDF documents found yet.")

if st.sidebar.button("🔄 Build / Refresh Index", use_container_width=True):
    with st.spinner("Extracting PDFs, creating chunks and building ChromaDB..."):
        try:
            vector_store, file_count, chunk_count, names = build_index()
            st.session_state["vector_store"] = vector_store
            st.session_state["index_ready"] = True
            st.sidebar.success(
                f"Index created!\n\nDocuments: {file_count}\n\nChunks: {chunk_count}"
            )
            st.sidebar.caption("Indexed: " + ", ".join(names))
        except Exception as error:
            st.sidebar.error(f"Indexing failed: {error}")

if st.sidebar.button("Load existing index", use_container_width=True):
    try:
        st.session_state["vector_store"] = load_existing_index()
        st.session_state["index_ready"] = True
        st.sidebar.success("Existing ChromaDB index loaded.")
    except Exception as error:
        st.sidebar.warning(
            "No existing compatible index was found. Click Build / Refresh Index. "
            f"Details: {error}"
        )

# Try loading an existing index once per session, without rebuilding it.
if "vector_store" not in st.session_state:
    try:
        st.session_state["vector_store"] = load_existing_index()
        st.session_state["index_ready"] = True
    except Exception:
        st.session_state["index_ready"] = False

st.sidebar.markdown("---")
st.sidebar.write("**LLM:** Ollama – `llama3.2:latest`")
st.sidebar.write("**Vector database:** ChromaDB")
st.sidebar.write("**Embedding model:** all-MiniLM-L6-v2")

# -------------------- MAIN CHAT --------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if st.session_state.get("index_ready"):
    st.success("Knowledge base is ready. Ask a question about your documents.")
else:
    st.warning(
        "Build the document index from the sidebar before asking questions."
    )

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption("Sources: " + " | ".join(message["sources"]))

question = st.chat_input(
    "Ask a question about your uploaded education documents...",
    disabled=not st.session_state.get("index_ready", False)
)

if question:
    st.session_state["messages"].append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating your answer..."):
            try:
                answer, sources = answer_question(
                    question,
                    st.session_state["vector_store"]
                )
                st.markdown(answer)

                if sources:
                    st.caption("Sources: " + " | ".join(sources))

                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    }
                )

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                save_csv_row(
                    HISTORY_FILE,
                    ["Timestamp", "Question", "Answer", "Sources"],
                    [timestamp, question, answer, "; ".join(sources)]
                )
                save_csv_row(
                    ANSWERS_FILE,
                    ["Query", "Answer", "Status"],
                    [question, answer, "Success"]
                )

            except Exception as error:
                error_text = (
                    "I could not generate an answer. Check that Ollama is running, "
                    "the `llama3.2:latest` model is available, and the index is built.\n\n"
                    f"Error: {error}"
                )
                st.error(error_text)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": error_text}
                )

# -------------------- CHAT ACTIONS --------------------
st.markdown("---")
left, middle, right = st.columns(3)

with left:
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

with middle:
    if HISTORY_FILE.exists():
        st.download_button(
            "⬇️ Download chat history",
            data=HISTORY_FILE.read_bytes(),
            file_name="chat_history.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.button("⬇️ Download chat history", disabled=True, use_container_width=True)

with right:
    if ANSWERS_FILE.exists():
        st.download_button(
            "⬇️ Download answers CSV",
            data=ANSWERS_FILE.read_bytes(),
            file_name="answers.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.button("⬇️ Download answers CSV", disabled=True, use_container_width=True)

st.caption(
    "EduDoc AI | Built with LangChain, ChromaDB, Ollama, Sentence Transformers and Streamlit"
)
