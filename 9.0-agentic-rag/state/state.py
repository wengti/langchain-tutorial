from pydantic import BaseModel, Field
from langchain_core.documents import Document


class OverallState(BaseModel):
    entry_point: int = Field(default=0)
    user_query: str = Field(default="")
    generated_answer: str = Field(default="")
    should_web_search: bool = Field(default=False)
    is_hallucinating: bool = Field(default=False)
    hallucination_check_count: int = Field(default=0)
    is_useful: bool = Field(default=False)
    usefulness_check_count: int = Field(default=0)
    retrieved_documents: list[Document] = Field(default_factory=list)
    searched_documents: list[Document] = Field(default_factory=list)
