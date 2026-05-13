from typing import List

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_tavily import TavilySearch
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from openai.types.beta.threads.runs import tool_call
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()


# ===============
# Prompt Template
# Note: General Template will be shared between the initial responder and revisor node with a difference in their chain_instruction.
generation_system_message = """
You are a helpful research assistant that follows the following instructions:

1. {chain_instruction}
2. Reflect on your answers by listing out what is missing and what is superfluous in the generated answer. This will be the directions for future iterations to improve on its quality.
3. After reflection, Suggest 1 to 3 query terms that can be helpful for searching relevant information on the web to help improving the generated answer in the future iterations.
4. The word limitation is only applicable to the actual answer itself. You may generate as much reflection, seach queries and references without being limited by the word limitation.


"""
general_template = ChatPromptTemplate(
    [
        ("system", generation_system_message),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

tool_call_system_message = """
You have access to a search tool to get information from the internet.
Refer to the latest previous iterations of reflection to identify what is missing in the previous reply and what are the search queries suggested.
Make use of the search queries suggested to find those information from the internet.
"""

tool_call_template = ChatPromptTemplate(
    [
        SystemMessage(content=tool_call_system_message),
        MessagesPlaceholder(variable_name="messages"),
    ]
)


# ===============


# ===============
# Chain


# Tools
@tool
def search_tool(queries: List[str]):
    """A tool that is used to search information on the web.
    Args:
        queries: A list of terms that are used to search information on the web.
    Returns:
        Information found on the web.
    """
    tavily_search_tool = TavilySearch(max_results=5)
    result = tavily_search_tool.batch([{"query": query} for query in queries])
    return result


tools_list = [search_tool]


# Models
basic_llm = init_chat_model(
    model="openai:gpt-5.4",
    temperature=0,
)
llm_with_tools = basic_llm.bind_tools(
    tools_list,
    tool_choice="search_tool",
)  # Make sure that the llm always call this tools with `tool_choice`


# First Responder Chain - takes initial prompt and generate an initial response
first_responder_chain_instruction = """
Generate an answer based on the user query in about 250 words. 
"""
first_responder_prompt = general_template.partial(
    chain_instruction=first_responder_chain_instruction,
)
first_responder_chain = first_responder_prompt | basic_llm


# Revisor Chain - revise the previous answer and generate an improved response based on the search results from previous action
revisor_chain_instruction = """
You are going to improve the answer generated in the latest previous iterations using the web search results returned in previous latest iteration.
When generating the answer, you MUST add a numerical citation at the end of a sentence if you have used any. 
Subsequently, at the end of the answer, include a section that lists out the corresponding source of citation.
For an instance:
    This source cites that .... [1].
    References:
        [1]: https://some_link.com
        [2]: https://extra_link.com
Please also refer to the latest reflection to remove what is considered as superfluous in the previous answer.
"""

revisor_prompt = general_template.partial(
    chain_instruction=revisor_chain_instruction,
)
revisor_chain = revisor_prompt | basic_llm

# Tool Call Chain
tool_call_chain = tool_call_template | llm_with_tools


# ===============
# Graph

# Hyperparameter
MAX_TOOLS_COUNT = 2


# State
class OverallState(MessagesState):
    tools_count: int


# Builder
builder = StateGraph(OverallState)


# Node
tool_node = ToolNode(tools=tools_list)


def first_responder_node(state: OverallState):
    raw_response = first_responder_chain.invoke({"messages": state["messages"]})
    return {"messages": [raw_response]}


def revisor_node(state: OverallState):
    raw_response = revisor_chain.invoke({"messages": state["messages"]})
    return {"messages": [raw_response]}


def tool_call_node(state: OverallState):
    cur_tools_count = state["tools_count"]
    raw_response = tool_call_chain.invoke({"messages": state["messages"]})
    return {"messages": [raw_response], "tools_count": cur_tools_count + 1}


builder.add_node("first_responder_node", first_responder_node)
builder.add_node("revisor_node", revisor_node)
builder.add_node("first_responder_tool_call_node", tool_call_node)
builder.add_node("revisor_tool_call_node", tool_call_node)
builder.add_node("tool_node", tool_node)


# Edge
def should_continue(state: OverallState):
    if state["tools_count"] > MAX_TOOLS_COUNT:
        return "end"
    return "revisor_tool_call_node"


builder.add_edge(START, "first_responder_node")
builder.add_edge("first_responder_node", "first_responder_tool_call_node")
builder.add_edge("first_responder_tool_call_node", "tool_node")
builder.add_edge("tool_node", "revisor_node")
builder.add_conditional_edges(
    "revisor_node",
    should_continue,
    {"end": END, "revisor_tool_call_node": "revisor_tool_call_node"},
)
builder.add_edge("revisor_tool_call_node", "tool_node")

# Compile
graph = builder.compile()

# Plot graph
with open("graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())


def main():
    print("Hello from 8-0-reflextion-agent!")
    user_query = "What is Dunning-Kruger effect?"
    result = graph.invoke({"messages": user_query, "tools_count": 0})
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
