from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(
        page_content="Deep learning is a subset of machine learning that uses neural networks.",
        metadata={"source": "deeplearning.pdf", "page": 1}
    ),

    Document(
        page_content="Neural networks consist of interconnected layers of artificial neurons.",
        metadata={"source": "deeplearning.pdf", "page": 2}
    ),

    Document(
        page_content="Training a neural network involves adjusting weights to minimize the loss function.",
        metadata={"source": "deeplearning.pdf", "page": 3}
    )
]


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vectore_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="chroma_db"
)


result = vectore_store.similarity_search("What is use for AI/ML?",k=2)

for r in result:
    print(r.page_content)

retriver = vectore_store.as_retriever() 

docs = retriver.invoke("Explian Deep learning")

for d in docs:
    print(d.page_content)