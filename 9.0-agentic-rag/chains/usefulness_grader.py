from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

system_prompt = """
You are an expert in classifying whether an answer is useful to a user query.
If you detect that the answer is helpful in the context of the user query with detailed explaination, answer True.
If you find that the answer is incomplete and lack details, you MUST answer False.
"""

prompt_template = ChatPromptTemplate(
    [
        ("system", system_prompt),
        (
            "human",
            "User query: {user_query} \n Answer: {generated_answer}",
        ),
    ]
)


class UsefulnessGraderResponseFormat(BaseModel):
    binary_flag: bool = Field(
        description="True if the provided answer is useful for the user query, otherwise False"
    )


raw_llm_model = init_chat_model(
    model="openai:gpt-5.4-mini",
    temperature=0,
)

llm_model = raw_llm_model.with_structured_output(
    schema=UsefulnessGraderResponseFormat,
)

usefulness_grader_chain = prompt_template | llm_model
