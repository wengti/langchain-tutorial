from dotenv import load_dotenv

from state.state import OverallState
from chains.retrieval_grader import (
    RetrievalGraderResponseFormat,
    retrieval_grader_chain,
)


def retrieval_grader_node(state: OverallState):
    print("=== RETRIEVAL GRADER NODE ===")
    user_query = state.user_query
    retrieved_documents = state.retrieved_documents
    filtered_documents = []
    should_web_search = False

    for document in retrieved_documents:
        response: RetrievalGraderResponseFormat = retrieval_grader_chain.invoke(
            {"user_query": user_query, "document_txt": document.page_content}
        )
        if response.binary_flag:
            filtered_documents.append(document)
        else:
            should_web_search = True

    return {
        "retrieved_documents": filtered_documents,
        "should_web_search": should_web_search,
    }
