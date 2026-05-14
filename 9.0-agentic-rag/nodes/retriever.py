from dotenv import load_dotenv

from state.state import OverallState
from vector_db.vector_db import create_vector_store

load_dotenv()

# HYPERPARAMETER
NUM_RETRIEVED_DOCS = 5

vector_store = create_vector_store()
retriever = vector_store.as_retriever(
    search_kwargs={"k": NUM_RETRIEVED_DOCS},
)


def retriever_node(state: OverallState):
    print("=== RETRIEVER NODE ===")
    user_query = state.user_query
    retrived_documents = retriever.invoke(user_query)
    return {"retrieved_documents": retrived_documents}
