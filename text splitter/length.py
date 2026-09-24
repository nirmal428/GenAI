from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

data=TextLoader("notes.txt")
docs=data.load()

spiltter = CharacterTextSplitter(
    separator="",
    chunk_size=10,
    chunk_overlap=2
)

text = spiltter.split_documents(docs)

for i in text:
    print(i.page_content)
    print()
    print()