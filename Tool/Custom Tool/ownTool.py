from langchain.tools import tool

@tool
def get_greet(name : str) -> str:
    """Generate a greeting message for user"""

    return f"Hello {name}, Welcome to the AI world"

result = get_greet.invoke({"name":"Akhil"})
print(result)
