# PDF-based question answering with RAG
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# Load PDF
loader = PyPDFLoader("your.pdf")
docs = loader.load()

# Create Vector Store
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)

# Create QA Chain
llm = ChatOpenAI(model_name="gpt-4o-mini")
qa = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

while True:
    query = input("Ask: ")
    print(qa.run(query))
