#load the pdf 
#split into chunks
#create te embedding
#store into chromadb

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

data = PyPDFLoader("documents loader/deeplearning.pdf")
docs = data.load()

print(f"Pages: {len(docs)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=10
)
chunks=splitter.split_documents(docs)
print(f"Chunks: {len(chunks)}")

embedding_model = HuggingFaceEmbeddings()

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="mainDB-Chroma"
)
