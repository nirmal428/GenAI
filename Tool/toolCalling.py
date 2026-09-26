from dotenv import load_dotenv
load_dotenv()
from langchain.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from rich import print
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Create a tool
@tool
def  get_text_lengh(text: str) -> int:
    """Returns the number of charactorin a given text"""
    return len(text)

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

#tool binding
llm_with_tool = llm.bind_tools([get_text_lengh])

result  = llm.invoke("Hello")
result2 = llm_with_tool.invoke("Hello")

print(result)
print()
print()
print(result2)