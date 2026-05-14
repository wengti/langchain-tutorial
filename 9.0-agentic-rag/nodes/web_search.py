from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_core.documents import Document

from state.state import OverallState

load_dotenv()


def web_search_node(state: OverallState):
    print("=== WEB SEARCH NODE ===")
    user_query = state.user_query
    searched_documents = []

    max_search_results = 5 if state.usefulness_check_count == 0 else 10

    tavily_search = TavilySearch(max_results=max_search_results)
    response = tavily_search.invoke({"query": user_query})
    for result in response["results"]:
        document = Document(
            page_content=result["content"],
            metadata={"source": result["url"]},
        )
        searched_documents.append(document)

    return {
        "searched_documents": searched_documents,
        "hallucination_check_count": 0,
    }
