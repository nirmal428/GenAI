from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel


load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

parser = StrOutputParser()

sort_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} im 1-2 line"
)

detailed_prompt = ChatPromptTemplate.from_template(
    "Explain {topic} im simple word"
)

topic = "machine learning"

chain = RunnableParallel({
    "sort": sort_prompt | model | parser,
    "detailed":detailed_prompt | model | parser
})

result = chain.invoke({"topic":"Machone learning"})
print(result["sort"])
print(result["detailed"])