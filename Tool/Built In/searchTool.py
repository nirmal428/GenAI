from dotenv import load_dotenv
load_dotenv()

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser



search_tool = TavilySearchResults(max_results=5)


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Summarize the news in clear bullet points.

News:
{news}
""")

chain = prompt | llm | StrOutputParser()
pre_result = search_tool.invoke("latest AI news")

result = chain.invoke({
    "news": pre_result
})
print(result)