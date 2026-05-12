from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, MessagesState, StateGraph
from langchain.tools import tool
from langgraph.prebuilt import ToolNode

load_dotenv()


# ================
# State
# Note: MessagesState inherit from TypedDict
# By default it has an attribute - messages that is typed as following:
# Annotated[list[AnyMessage], add_messages]
# where add_messages is particularly useful, when whenever a node return {messages: [AIMessage(...) or UserMessage(...)]}, it will automatically be appended
class OverallState(MessagesState):
    extra_field: str


# ================

# ================
# Tools
tavily_search = TavilySearch(
    max_results=5,
)


@tool
def triple(input: float) -> float:
    """
    A tool that is called to multiply an input float data and return it.
    Args:
        input: An input number to be multiplied by 3.
    Returns:
        An output number that is the input multiplied by 3.
    """
    return input * 3


tools = [tavily_search, triple]
# ================


# ================
# Node - Reasoning Node
system_prompt = """
You are a helpful AI assistant that has access to tools.
Make use of the tools to answer the user's query whenever if possible or necessary.
"""

base_model = init_chat_model(model="openai:gpt-5.4-mini", temperature=0)

# Pass the available tools to the model
# Since it is only a model and not an agent, it does not have the capabilities of executing ReAct Loop
# A certain graph flow will be implemented subsequently in the code to check the model tries to call a tool
# Which will direct the flow the graph nodes that will execute tool
model_with_tools = base_model.bind_tools(tools)


# The actual reasoning node
# It always unpack the messages that are stored in the state along with the system prompt
# The responses are appended to the messages field inherited from MessagesState
# Usually this will overwrite the messages attribute, but because of MessagesState having the add_messages operator
# So long that the returned value of "messages" is a list of [AIMessage(...) or equivalent], it will be added to the state
def reasoning_node(state: OverallState):
    conversation = [SystemMessage(content=system_prompt), *state["messages"]]
    result = model_with_tools.invoke(conversation)
    return {"messages": [result]}


# ================

# ================

# Node - Acting Node
# This is a prebuilt node provided by the LangGraph
# Refer to: https://reference.langchain.com/python/langgraph.prebuilt/tool_node/ToolNode
# It automatically check the graph state and see if the last entry of "messages" has a tool call
# It then call, execute the call and return the response in a ToolMessage
acting_node = ToolNode(tools)

# ================


# ================


# Conditional Edge - Whether to continue taking actions or returning an answer
# Check the state - whether last message contains a tool call
# If so redirecting it to the acting_node
# Else redirecting it to END
def should_continue(state: OverallState):
    if state["messages"][-1].tool_calls:
        return "acting_node"
    else:
        return "end"


# ================


# Graph
# As seen when defining the builder,
# the OverallState is passed in as the type that governs the type of state in the graph
builder = StateGraph(OverallState)

# .add_node usually will just take the name of the function as the node name for reference in the graph building
# However, for prebuilt node like ToolNode, this doesnt work
# Therefore for standardification, I have assigned name for both of them
builder.add_node("reasoning_node", reasoning_node)
builder.add_node("acting_node", acting_node)

# In particular, for the conditional edges, usually, the 3rd argument is not needed
# The 3rd argument basically maps the output of directing function to the name of the node
# But through trial error, it is found that it is needed to plot the graph nicely
builder.add_edge(START, "reasoning_node")
builder.add_conditional_edges(
    "reasoning_node",
    should_continue,
    {"acting_node": "acting_node", "end": END},
)
builder.add_edge("acting_node", "reasoning_node")


graph = builder.compile()

# Draw and save the graph
with open("graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())


# When the graph is compiled, it is basically just like a runnable interface as a chain.
# It can be streamed, invoked or batched etc...
def main():
    print("Hello from 6-0-langgraph-react!")
    user_input = (
        "What is the temperature in Kuala Lumpur now? List it and multiply by 3."
    )
    result = graph.invoke({"messages": user_input})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
