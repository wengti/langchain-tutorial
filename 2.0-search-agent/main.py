from dotenv import load_dotenv

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage, AIMessage, SystemMessage
from langchain_tavily import TavilySearch

load_dotenv()


# Defining a tool
@tool
def search(query: str):
    """Tool that seraches over internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """
    print(f"Searching for {query}")
    tavily_search = TavilySearch(max_results=5, topic="general")
    return tavily_search.invoke({"query": query})


# Setup the agent
tools = [search]
messages = [
    HumanMessage(
        "List out 3 job postings that are still open now that require an AI engineer with knowledge on langchain in Selangor, Malaysia"
    )
]
llm = create_agent(
    model="openai:gpt-5.4-mini",
    tools=tools,
    system_prompt="You are a helpful assistant that answer user's question amicably.",
)


def main():
    print("Hello from 2-0-search-agent!")

    # Call the agent into action
    response = llm.invoke({"messages": messages})

    # View the result
    print(response["messages"][-1].content)


if __name__ == "__main__":
    main()
