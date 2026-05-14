from dotenv import load_dotenv

from state.state import OverallState
from chains.generator import GeneratorResponseFormat, generator_chain

load_dotenv()


def generator_node(state: OverallState):
    print("=== GENERATOR NODE ===")
    user_query = state.user_query
    retrieved_documents = state.retrieved_documents
    searched_documents = state.searched_documents
    documents = []
    documents.extend(retrieved_documents)
    documents.extend(searched_documents)

    is_hallucinating = state.is_hallucinating
    if is_hallucinating:
        user_query += (
            "\nYou have tried to generate content that is not based on the provided context before."
            "Make sure you do not do this in this attempt. Generate solely based on the provided context."
        )

    context = "\n\n".join(
        [
            f"Sources: {document.metadata["source"]}\nContent: {document.page_content}"
            for document in documents
        ]
    )
    response: GeneratorResponseFormat = generator_chain.invoke(
        {
            "context": context,
            "user_query": user_query,
        },
    )
    return {"generated_answer": response.generation}
