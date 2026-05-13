from langgraph.graph import END, START, StateGraph

from nodes.generator import generator_node
from nodes.retrieval_grader import retrieval_grader_node
from nodes.retriever import retriever_node
from nodes.web_search import web_search_node
from state.state import OverallState

# NODE_NAME
RETRIEVER_NODE = "retriver_node"
RETRIEVAL_GRADER_NODE = "retrieval_grader_node"
WEB_SEARCH_NODE = "web_search_node"
GENERATOR_NODE = "generator_node"

builder = StateGraph(OverallState)
builder.add_node(RETRIEVER_NODE, retriever_node)
builder.add_node(RETRIEVAL_GRADER_NODE, retrieval_grader_node)
builder.add_node(WEB_SEARCH_NODE, web_search_node)
builder.add_node(GENERATOR_NODE, generator_node)


def should_web_search(state: OverallState):
    if state.should_web_search:
        return WEB_SEARCH_NODE
    return GENERATOR_NODE


builder.add_edge(START, RETRIEVER_NODE)
builder.add_edge(RETRIEVER_NODE, RETRIEVAL_GRADER_NODE)
builder.add_conditional_edges(
    RETRIEVAL_GRADER_NODE,
    should_web_search,
    {
        WEB_SEARCH_NODE: WEB_SEARCH_NODE,
        GENERATOR_NODE: GENERATOR_NODE,
    },
)
builder.add_edge(WEB_SEARCH_NODE, GENERATOR_NODE)
builder.add_edge(GENERATOR_NODE, END)

graph = builder.compile()

with open("graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())


def main():
    print("Hello from 9-0-agentic-rag!")
    response = graph.invoke(
        {"user_query": "What is an AI agent? Can it make use of Corrective RAG?"}
    )
    print(response["generated_answer"])


if __name__ == "__main__":
    main()
