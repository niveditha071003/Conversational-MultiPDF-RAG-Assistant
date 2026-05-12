import streamlit as st
import tempfile
import os
import shutil

from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# PAGE TITLE
st.title("Conversational Multi-PDF RAG")

DB_PATH = "faiss_db"

# SIDEBAR
with st.sidebar:
    st.header("About")

    st.write(
        """
        Conversational Multi-PDF RAG Assistant

        Features:
        - Multi-PDF Upload
        - Conversational Memory
        - Source Citations
        - Local LLM
        - FAISS Vector Search
        - Reset Vector Database
        """
    )

    st.divider()

    st.write("Built with:")
    st.write("- Streamlit")
    st.write("- LangChain")
    st.write("- Ollama")
    st.write("- FAISS")
    st.write("- HuggingFace Embeddings")

# RESET VECTOR DB BUTTON
if st.button("Reset Vector Database"):
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH)
        st.session_state.chat_history = []
        st.success("Vector database reset successfully.")
    else:
        st.info("No vector database found.")

# SESSION MEMORY
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Fix old string-based memory
if st.session_state.chat_history and isinstance(st.session_state.chat_history[0], str):
    st.session_state.chat_history = []

# UPLOAD MULTIPLE PDFs
uploaded_files = st.file_uploader(
    "Upload PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    st.success(f"{len(uploaded_files)} PDF(s) uploaded successfully.")

    all_documents = []

    # LOAD ALL PDFs
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        all_documents.extend(documents)

    # SPLIT TEXT
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    docs = text_splitter.split_documents(all_documents)

    # EMBEDDINGS
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # VECTOR DATABASE
    if os.path.exists(DB_PATH):
        db = FAISS.load_local(
            DB_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(DB_PATH)

    retriever = db.as_retriever(
        search_kwargs={"k": 4}
    )

    # LOAD LOCAL MODEL
    llm = ChatOllama(model="phi3")

    # USER QUESTION
    query = st.text_input("Ask a question")

    if query:

        # RETRIEVE CONTEXT
        retrieved_docs = retriever.invoke(query)

        context = ""
        sources = []

        for doc in retrieved_docs:
            page = doc.metadata.get("page", "Unknown")

            if page != "Unknown":
                sources.append(f"Page {page + 1}")
                context += f"\n(Page {page + 1})\n"
            else:
                sources.append("Page Unknown")
                context += "\n(Page Unknown)\n"

            context += doc.page_content
            context += "\n"

        # CHAT HISTORY
        history = "\n".join(
            [
                f"{msg['role']}: {msg['content']}"
                for msg in st.session_state.chat_history
                if isinstance(msg, dict)
            ]
        )

        # STRICT PROMPT
        prompt = f"""
You are a PDF question-answering assistant.

Answer ONLY using the provided PDF context.

If the answer is not found in the context, say:
"I could not find the answer in the uploaded PDFs."

Do NOT use outside knowledge.

Answer clearly and concisely in 5-8 lines.

Previous Conversation:
{history}

Context:
{context}

Question:
{query}
"""

        with st.spinner("Searching PDFs and generating answer..."):

            response = llm.invoke(prompt)
            answer = response.content

            unique_sources = list(set(sources))
            answer += "\n\nSources: " + ", ".join(unique_sources)

            # SAVE MEMORY
            st.session_state.chat_history.append(
                {"role": "user", "content": query}
            )

            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer}
            )

        # CHAT DISPLAY
        for message in st.session_state.chat_history:
            if isinstance(message, dict):
                with st.chat_message(message["role"]):
                    st.write(message["content"])

else:
    st.info("Please upload one or more PDFs to start.")