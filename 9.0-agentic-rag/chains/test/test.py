import pytest

from vector_db.vector_db import create_vector_store
from chains.retrieval_grader import (
    RetrievalGraderResponseFormat,
    retrieval_grader_chain,
)
from chains.generator import GeneratorResponseFormat, generator_chain


@pytest.mark.parametrize(
    ("true_user_query", "compare_text", "expected_output"),
    [
        ("prompt_engineering", "prompt_engineering", True),
        ("prompt_engineering", "chicken wing", False),
    ],
)
def test_retrieval_grader_answer_true(true_user_query, compare_text, expected_output):
    user_query = true_user_query
    vector_store = create_vector_store()
    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5},
    )
    retrieved_document = retriever.invoke(user_query)[0]

    response: RetrievalGraderResponseFormat = retrieval_grader_chain.invoke(
        {
            "user_query": compare_text,
            "document_txt": retrieved_document.page_content,
        }
    )

    assert response.binary_flag == expected_output


def test_generation_chain():
    response: GeneratorResponseFormat = generator_chain.invoke(
        {
            "context": "Source: https://link.com Content: Prompt Engineering is Good.",
            "user_query": "Is Prompt Engineering Good?",
        },
    )
    assert type(response.generation) == str
