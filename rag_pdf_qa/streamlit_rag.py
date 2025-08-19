import asyncio
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
# from openai.error import RateLimitError
from openai import RateLimitError
from openai import RateLimitError, APIError, APIConnectionError


import tempfile
import os
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate

# Define a strict prompt
qa_prompt = PromptTemplate(
    template="""
You are a helpful assistant for answering questions based on documents.
Use only the following context to answer the question.
If the answer is not contained in the context, say: "I don't know".

Context:
{context}

Question: {question}
Answer:""",
    input_variables=["context", "question"]
)


# Ensure event loop exists (for Streamlit + async libs)
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Load API keys
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
google_key = os.getenv("GOOGLE_API_KEY")

if not openai_key and not google_key:
    st.error("Please set at least one API key in your .env file (OPENAI_API_KEY or GOOGLE_API_KEY).")
    st.stop()

st.set_page_config(page_title="PDF QA Bot (Failover)", page_icon="🤖", layout="wide")
st.title("PDF Question-Answer Bot — OpenAI → Gemini Failover")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.info("Processing PDF... Please wait ⏳")

    # Load and process PDF
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    # Try OpenAI embeddings first, else fallback to Gemini
    use_gemini = False
    try:
        if openai_key:
            embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
            llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=openai_key)
            st.caption("Using OpenAI")
        else:
            raise ValueError("No OpenAI key found, using Gemini...")
    except Exception as e:
        st.warning(f"OpenAI unavailable ({e}), switching to Gemini.")
        use_gemini = True

    if use_gemini and google_key:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_key)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_key)
        st.caption("Using Gemini")

    db = FAISS.from_documents(docs, embeddings)
    # qa = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
    # Build QA chain with the custom prompt
    qa = RetrievalQA.from_chain_type(
        llm,
        retriever=db.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": qa_prompt}
    )

    st.success("PDF processed! Ask your questions below:")

    query = st.text_input("Your Question:")
    if query:
        with st.spinner("Thinking..."):
            # try:
            #     answer = qa.run(query)
            # except RateLimitError:
            #     st.warning("OpenAI quota exceeded. Switching to Gemini...")
            #     if google_key:
            #         embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_key)
            #         llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_key)
            #         db = FAISS.from_documents(docs, embeddings)
            #         qa = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
            #         answer = qa.run(query)
            #         st.caption("Now using Gemini")
            #     else:
            #         st.error("No Gemini key available for failover.")
            #         st.stop()

            

            try:
                answer = qa.run(query)
            except (RateLimitError, APIError, ServiceUnavailableError) as e:
                st.warning(f"⚠️ OpenAI error: {e}. Switching to Gemini...")
                if google_key:
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_key)
                    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_key)
                    db = FAISS.from_documents(docs, embeddings)
                    qa = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
                    answer = qa.run(query)
                    st.caption("🔄 Now using Gemini")
                else:
                    st.error("❌ No Gemini key available for failover.")
                    st.stop()


        st.markdown(f"**Answer:** {answer}")

        if st.checkbox("Show relevant document chunks"):
            docs = db.similarity_search(query, k=3)
            for i, doc in enumerate(docs, start=1):
                st.write(f"**Chunk {i}:** {doc.page_content}")


# import asyncio
# import streamlit as st
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
# from langchain.chains import RetrievalQA
# import tempfile
# import os
# from dotenv import load_dotenv

# # Ensure event loop exists (for Streamlit + async libs)
# try:
#     asyncio.get_running_loop()
# except RuntimeError:
#     asyncio.set_event_loop(asyncio.new_event_loop())

# # Load API key from .env
# load_dotenv()
# google_api_key = os.getenv("GEMINI_API_KEY")

# if not google_api_key:
#     st.error("⚠️ GOOGLE_API_KEY not found in .env file.")
#     st.stop()

# st.set_page_config(page_title="📚 PDF QA Bot (Gemini)", page_icon="🤖", layout="wide")
# st.title("📚 PDF Question-Answer Bot — Gemini Edition")

# # Upload PDF
# uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# if uploaded_file:
#     # Save uploaded PDF to a temp file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
#         tmp.write(uploaded_file.read())
#         tmp_path = tmp.name

#     st.info("Processing PDF... Please wait ⏳")

#     # Load and process PDF
#     loader = PyPDFLoader(tmp_path)
#     docs = loader.load()

#     # Use Google Gemini embeddings
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
#     db = FAISS.from_documents(docs, embeddings)

#     # Use Gemini model for Q&A
#     llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)
#     qa = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

#     st.success("✅ PDF processed! Ask your questions below:")

#     query = st.text_input("Your Question:")
#     if query:
#         with st.spinner("Thinking..."):
#             answer = qa.run(query)
#         st.markdown(f"**Answer:** {answer}")

#         # Optional: Show relevant document chunks
#         if st.checkbox("Show relevant document chunks"):
#             docs = db.similarity_search(query, k=3)
#             for i, doc in enumerate(docs, start=1):
#                 st.write(f"**Chunk {i}:** {doc.page_content}")
