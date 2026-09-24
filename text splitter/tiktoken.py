from langchain_text_splitters import TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader

data=PyPDFLoader("GRU.pdf")
docs=data.load()

spiltter = TokenTextSplitter(
    chunk_size=1000,
    chunk_overlap=10
)

text = spiltter.split_documents(docs)

for i in text:
    print(len(i.page_content))
    print()
    print()