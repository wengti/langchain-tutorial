from chains.hallucination_grader import (
    HallucinationGraderResponseFormat,
    hallucination_grader_chain,
)
from state.state import OverallState


def hallucination_grader_node(state: OverallState):
    print("=== HALLUCINATION GRADER NODE ===")
    generated_answer = state.generated_answer

    retrieved_documents = state.retrieved_documents
    searched_documents = state.searched_documents
    documents = []
    documents.extend(retrieved_documents)
    documents.extend(searched_documents)

    count = state.hallucination_check_count
    context = "\n\n".join(
        [
            f"Sources: {document.metadata["source"]}\nContent: {document.page_content}"
            for document in documents
        ]
    )
    response: HallucinationGraderResponseFormat = hallucination_grader_chain.invoke(
        {
            "document_txt": context,
            "generated_answer": generated_answer,
        },
    )
    return {
        "is_hallucinating": response.binary_flag,
        "hallucination_check_count": count + 1,
    }
