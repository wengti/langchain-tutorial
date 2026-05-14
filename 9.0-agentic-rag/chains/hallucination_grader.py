from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

load_dotenv()

system_prompt = """
You are an expert in classifying whether an answer is based on the provided document text.
If you detect any form of hallucination or creation of content outside of the provided document text, provide an answer of True, else provide an answer of False.
"""

prompt_template = ChatPromptTemplate(
    [
        ("system", system_prompt),
        (
            "human",
            "Document Text: {document_txt} \n Answer: {generated_answer}",
        ),
    ]
)


class HallucinationGraderResponseFormat(BaseModel):
    binary_flag: bool = Field(
        description="True if the provided answer is based on the provided document text, otherwise False"
    )


raw_llm_model = init_chat_model(
    model="openai:gpt-5.4-mini",
    temperature=0,
)

llm_model = raw_llm_model.with_structured_output(
    schema=HallucinationGraderResponseFormat,
)

hallucination_grader_chain = prompt_template | llm_model
