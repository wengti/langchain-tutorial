from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

system_prompt = """
You are an expert in classifying whether a document text contain information that is relevant to the user query.
Answer True if the provided document text is relevant otherwise answer No.
"""

prompt_template = ChatPromptTemplate(
    [
        ("system", system_prompt),
        ("human", "User query: {user_query} \n Document Text: {document_txt}"),
    ]
)


class RetrievalGraderResponseFormat(BaseModel):
    binary_flag: bool = Field(
        description="True if the provided document text is relevant to the user query, otherwise False"
    )


raw_llm_model = init_chat_model(
    model="openai:gpt-5.4-mini",
    temperature=0,
)

llm_model = raw_llm_model.with_structured_output(
    schema=RetrievalGraderResponseFormat,
)

retrieval_grader_chain = prompt_template | llm_model
