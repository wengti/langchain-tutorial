from langgraph.graph import END, START, StateGraph

from nodes.generator import generator_node
from nodes.hallucination_grader import hallucination_grader_node
from nodes.retrieval_grader import retrieval_grader_node
from nodes.retriever import retriever_node
from nodes.usefulness_grader import usefulness_grader_node
from nodes.web_search import web_search_node
from nodes.router import router_node
from state.state import OverallState

# NODE_NAME
RETRIEVER_NODE = "retriver_node"
RETRIEVAL_GRADER_NODE = "retrieval_grader_node"
WEB_SEARCH_NODE = "web_search_node"
GENERATOR_NODE = "generator_node"
HALLUCINATION_GRADER_NODE = "hallucination_grader_node"
USEFULNESS_GRADER_NODE = "usefulness_grader_node"
ROUTER_NODE = "router_node"

builder = StateGraph(OverallState)
builder.add_node(RETRIEVER_NODE, retriever_node)
builder.add_node(RETRIEVAL_GRADER_NODE, retrieval_grader_node)
builder.add_node(WEB_SEARCH_NODE, web_search_node)
builder.add_node(GENERATOR_NODE, generator_node)
builder.add_node(HALLUCINATION_GRADER_NODE, hallucination_grader_node)
builder.add_node(USEFULNESS_GRADER_NODE, usefulness_grader_node)
builder.add_node(ROUTER_NODE, router_node)


def should_web_search(state: OverallState):
    if state.should_web_search:
        return WEB_SEARCH_NODE
    return GENERATOR_NODE


def should_regenerate(state: OverallState):
    if not state.is_hallucinating or state.hallucination_check_count >= 3:
        return USEFULNESS_GRADER_NODE
    if state.is_hallucinating:
        return GENERATOR_NODE


def should_retry_web_search(state: OverallState):
    if state.is_useful or state.usefulness_check_count >= 3:
        return "end"
    return WEB_SEARCH_NODE


def choose_starting_point(state: OverallState):
    if state.entry_point == 0:
        return RETRIEVER_NODE
    else:
        return WEB_SEARCH_NODE


builder.add_edge(START, ROUTER_NODE)
builder.add_conditional_edges(
    ROUTER_NODE,
    choose_starting_point,
    {
        RETRIEVER_NODE: RETRIEVER_NODE,
        WEB_SEARCH_NODE: WEB_SEARCH_NODE,
    },
)

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
builder.add_edge(GENERATOR_NODE, HALLUCINATION_GRADER_NODE)
builder.add_conditional_edges(
    HALLUCINATION_GRADER_NODE,
    should_regenerate,
    {
        GENERATOR_NODE: GENERATOR_NODE,
        USEFULNESS_GRADER_NODE: USEFULNESS_GRADER_NODE,
    },
)
builder.add_conditional_edges(
    USEFULNESS_GRADER_NODE,
    should_retry_web_search,
    {
        "end": END,
        WEB_SEARCH_NODE: WEB_SEARCH_NODE,
    },
)

graph = builder.compile()

with open("graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())


def main():
    print("Hello from 9-0-agentic-rag!")
    response = graph.invoke(
        {"user_query": "What is an AI agent and What is a corrective RAG?"}
    )
    print(response["generated_answer"])


if __name__ == "__main__":
    main()
