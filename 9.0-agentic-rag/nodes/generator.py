from dotenv import load_dotenv

from state.state import OverallState
from chains.generator import GeneratorResponseFormat, generator_chain

load_dotenv()


def generator_node(state: OverallState):
    print("=== GENERATOR NODE ===")
    user_query = state.user_query
    documents = state.documents
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
