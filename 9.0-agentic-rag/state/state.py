from pydantic import BaseModel, Field
from langchain_core.documents import Document


class OverallState(BaseModel):
    user_query: str = Field(default="")
    generated_answer: str = Field(default="")
    should_web_search: bool = Field(default=False)
    documents: list[Document] = Field(default_factory=list)
