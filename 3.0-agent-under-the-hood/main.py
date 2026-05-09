from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langsmith import traceable

load_dotenv()

MODEL_NAME = "openai:gpt-5.4-mini"
TEMPERATURE = 0
MAX_ITER = 10


# Setup system_prompt
system_prompt = """
You are a helpful assistant on an e-commerce that will reply to user's enquiries to the products amicably.
You have access to the product catelog and discount to be applied based on a tier that is either gold, silver of bronze.
Call get_prices to get the price of the product when it is needed.
Call apply_discount to apply the discount and return the discounted price when it is needed.
Do not hallucinate for any of the parameters required in calling tools. Ask the user for more information when needed.
"""


# Setup tools
@tool
def get_prices(name: str) -> float:
    """A tool that can be used to get the price of an item based on the provided name.
    Args:
        name: Name of the item.
    Returns:
        The price of the item if the item exists otherwise will return 0, indicating that the item is non-existent.
    """
    inventory = {"laptop": 1199.99, "headset": 59.99, "mouse": 29.99}
    return inventory.get(name, 0)


@tool
def apply_discount(original_price: float, tier: str) -> float:
    """A tool that takes the original price of an item and apply a discount based on the tier and then return the corresponding prices after discount.
    Args:
        original_price: The original price of the item before discounted.
        tier: The discount tier to be applied, available to be in gold, silver or bronze. For a non-existent tier, it results in 0% discount.
    Returns:
        The discounted price of the item based on the applied tier.
    """
    discount_dict = {"gold": 50, "silver": 30, "bronze": 10}
    discount = discount_dict.get(tier, 0)
    return original_price * (100 - discount) / 100


# Function to call the agent
@traceable(name="LangChain Agent Loop")
def exec_agent(question: str):
    # Setup tools
    tools = [get_prices, apply_discount]
    tools_dict = {t.name: t for t in tools}  # t.name works because of @tool

    # Setup models - bind the tools to the model
    model = init_chat_model(MODEL_NAME, temperature=TEMPERATURE)
    model_with_tools = model.bind_tools(tools)

    # Initialize the conversation
    conversation = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question),
    ]

    # iterate for N times of loop
    for n_iter in range(MAX_ITER):
        print(f"ITERATION: {n_iter}")
        print("=" * 50)

        # Thought process by AI
        ai_response = model_with_tools.invoke(conversation)
        conversation.append(ai_response)
        print("Though process done by AI: ")
        print(ai_response.content + "\n")

        # Resolve tool calls
        tool_call = ai_response.tool_calls

        # if no more tool call return the response
        if not tool_call:
            return ai_response.content

        # otherwise resolve tool call
        tool_call_name = tool_call[0].get("name")
        tool_call_func = tools_dict.get(tool_call_name, None)
        tool_call_args = tool_call[0].get("args")
        tool_call_id = tool_call[0].get("id")
        if tool_call_func is None:
            raise ValueError(f"{tool_call_name}() is not available.")
        observation = tool_call_func.invoke(tool_call_args)

        # append AI message and tool call result to the conversations
        conversation.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )
        print(f"Tools executed by the model: {tool_call_name}({tool_call_args})")
        print(f"Result: str({observation})")

    # if after N times of loop, it still involves tool call, call it off and return an error message
    print(f"Fail to resolve the task in {MAX_ITER} iterations")
    return None


def main():
    print("Hello from 3-0-agent-under-the-hood!")
    exec_agent(question="What is the price of the headset with a dirt tier discount?")


if __name__ == "__main__":
    main()
