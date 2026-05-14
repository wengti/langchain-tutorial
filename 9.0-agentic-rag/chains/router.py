from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

system_prompt = """
You are an expert in choosing which route to take based on the user query.
If the user query is about prompt engineering, ai agent or adversarial attack on LLM, reply 0.
Else, reply 1.
"""

prompt_template = ChatPromptTemplate(
    [
        ("system", system_prompt),
        (
            "human",
            "User query: {user_query}",
        ),
    ]
)


class RouterResponseFormat(BaseModel):
    route_number: int = Field(
        description="This value is 0 if the user query is about prompt engineering, ai agent or adversarial attack, else it is 1"
    )


raw_llm_model = init_chat_model(
    model="openai:gpt-5.4-mini",
    temperature=0,
)

llm_model = raw_llm_model.with_structured_output(
    schema=RouterResponseFormat,
)

router_chain = prompt_template | llm_model
