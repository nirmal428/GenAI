from langchain_community.document_loaders import TextLoader

data = TextLoader("notes.txt")

doc = data.load()

print(doc[0])