from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} im simple word"
)

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke("Machine Learning")

print(result)

