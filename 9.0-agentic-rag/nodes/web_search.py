from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_core.documents import Document

from state.state import OverallState

load_dotenv()

tavily_search = TavilySearch(max_results=5)


def web_search_node(state: OverallState):
    print("=== WEB SEARCH NODE ===")
    user_query = state.user_query
    cur_documents = state.documents
    response = tavily_search.invoke({"query": user_query})
    searched_documents = []
    for result in response["results"]:
        document = Document(
            page_content=result["content"],
            metadata={"source": result["url"]},
        )
        searched_documents.append(document)
    cur_documents.extend(searched_documents)

    return {"documents": cur_documents}
