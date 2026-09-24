from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader,PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()

embedding_model = HuggingFaceEmbeddings()

vectorstore = Chroma(
    persist_directory="mainDB-Chroma",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":4,
        "fetch_k":10,
        "lambda_mult":0.5
    }
)

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant that answers questions using the provided PDF context.

Rules:
1. Answer the user's question using only the information provided in the context.
2. Do not make up or assume information that is not present in the context.
3. If the answer cannot be found in the context, clearly say:
   "I could not find the answer in the provided PDF."
4. Give a clear and concise answer.
5. When useful, explain the answer step-by-step.
6. Preserve important technical terms, names, numbers, and formulas from the PDF.

Context:
{context}
"""
    ),
    (
        "human",
        "{question}"
    )
])

print("Rag systema created")
print("press 0 for exit")

while True:
    query = input("You : ")
    if query == "0":
        break
    docs = retriever.invoke(query)

    context = "\n\n" .join(
        [doc.page_content for doc in docs ]
    )

    final_prompt = prompt.invoke({
        "context" : context,
        "question":query
    })

    response = llm.invoke(final_prompt)
    print(f"\n AI : {response.content}")