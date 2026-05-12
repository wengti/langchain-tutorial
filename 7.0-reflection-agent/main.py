from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, MessagesState, StateGraph

load_dotenv()

# ====================

# Creating chains

# Model
llm_model = init_chat_model(
    "openai:gpt-5.4-mini",
    temperature=0,
)

# Generation Chain
with open("generation_prompt.txt") as f:
    generation_prompt = f.read()
generation_prompt_template = ChatPromptTemplate(
    [
        SystemMessage(content=generation_prompt),
        MessagesPlaceholder("messages"),
    ]
)
generation_chain = generation_prompt_template | llm_model

# Reflection Chain
with open("reflection_prompt.txt") as f:
    reflection_prompt = f.read()
reflection_prompt_template = ChatPromptTemplate(
    [
        SystemMessage(content=reflection_prompt),
        MessagesPlaceholder("messages"),
    ]
)
reflection_chain = reflection_prompt_template | llm_model

# ====================

# ====================
# Creating graph


# State
class OverallState(MessagesState):
    extra_field: str


# Node
def generation_node(state: OverallState):
    return {"messages": [generation_chain.invoke({"messages": state["messages"]})]}


def reflection_node(state: OverallState):
    response = reflection_chain.invoke({"messages": state["messages"]})
    return {"messages": [HumanMessage(content=response.content)]}


# Conditional Edge
def should_continue(state: OverallState):
    if len(state["messages"]) > 6:
        return "end"
    return "reflection_node"


# Graph
builder = StateGraph(OverallState)
builder.add_node("generation_node", generation_node)
builder.add_node("reflection_node", reflection_node)

builder.add_edge(START, "generation_node")
builder.add_conditional_edges(
    "generation_node",
    should_continue,
    {"end": END, "reflection_node": "reflection_node"},
)
builder.add_edge("reflection_node", "generation_node")

graph = builder.compile()

# ====================

# Plot graph
with open("graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())


def main():
    print("Hello from 7-0-reflection-agent!")
    user_query = "Write a tweet about LangChain"
    result = graph.invoke({"messages": user_query})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
