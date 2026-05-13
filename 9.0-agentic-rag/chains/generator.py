from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

system_prompt = """
You are a helpful assistant who is an expert at compiling the information.
You will be provided some relevant context with references. Please make use of the context to answer the user's queries.
Whenever you use a reference from the context to answer the question, you must also cite it with numerical citation.
At the end of your answer, you must also include a references section if you have used citation as following:
References:
    [1] https://link.com
    [2] https://link2.com
However, when you found that the context is insufficient to answer the user queries properly. You must admit it honestly and ask for more information.
"""

prompt_template = ChatPromptTemplate(
    [
        ("system", system_prompt),
        ("human", "Context: {context}, User query: {user_query}"),
    ]
)


class GeneratorResponseFormat(BaseModel):
    generation: str = Field(
        description="Your answer to the user queries based on provided context."
    )


raw_llm_model = init_chat_model(
    model="openai:gpt-5.4-mini",
    temperature=0,
)

llm_model = raw_llm_model.with_structured_output(
    schema=GeneratorResponseFormat,
)

generator_chain = prompt_template | llm_model
