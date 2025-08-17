import asyncio
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
import tempfile
import os
from dotenv import load_dotenv

# Ensure event loop exists (for Streamlit + async libs)
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Load API key from .env
load_dotenv()
google_api_key = os.getenv("GEMINI_API_KEY")

if not google_api_key:
    st.error("⚠️ GOOGLE_API_KEY not found in .env file.")
    st.stop()

st.set_page_config(page_title="📚 PDF QA Bot (Gemini)", page_icon="🤖", layout="wide")
st.title("📚 PDF Question-Answer Bot — Gemini Edition")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Save uploaded PDF to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.info("Processing PDF... Please wait ⏳")

    # Load and process PDF
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    # Use Google Gemini embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    db = FAISS.from_documents(docs, embeddings)

    # Use Gemini model for Q&A
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)
    qa = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

    st.success("✅ PDF processed! Ask your questions below:")

    query = st.text_input("Your Question:")
    if query:
        with st.spinner("Thinking..."):
            answer = qa.run(query)
        st.markdown(f"**Answer:** {answer}")

        # Optional: Show relevant document chunks
        if st.checkbox("Show relevant document chunks"):
            docs = db.similarity_search(query, k=3)
            for i, doc in enumerate(docs, start=1):
                st.write(f"**Chunk {i}:** {doc.page_content}")
