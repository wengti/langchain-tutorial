import pytest

from vector_db.vector_db import create_vector_store
from chains.retrieval_grader import (
    RetrievalGraderResponseFormat,
    retrieval_grader_chain,
)
from chains.generator import GeneratorResponseFormat, generator_chain
from chains.hallucination_grader import hallucination_grader_chain
from chains.usefulness_grader import usefulness_grader_chain
from chains.router import RouterResponseFormat, router_chain

# @pytest.mark.parametrize(
#     ("true_user_query", "compare_text", "expected_output"),
#     [
#         ("prompt_engineering", "prompt_engineering", True),
#         ("prompt_engineering", "chicken wing", False),
#     ],
# )
# def test_retrieval_grader_answer_true(true_user_query, compare_text, expected_output):
#     user_query = true_user_query
#     vector_store = create_vector_store()
#     retriever = vector_store.as_retriever(
#         search_kwargs={"k": 5},
#     )
#     retrieved_document = retriever.invoke(user_query)[0]

#     response: RetrievalGraderResponseFormat = retrieval_grader_chain.invoke(
#         {
#             "user_query": compare_text,
#             "document_txt": retrieved_document.page_content,
#         }
#     )

#     assert response.binary_flag == expected_output


# def test_generation_chain():
#     response: GeneratorResponseFormat = generator_chain.invoke(
#         {
#             "context": "Source: https://link.com Content: Prompt Engineering is Good.",
#             "user_query": "Is Prompt Engineering Good?",
#         },
#     )
#     assert type(response.generation) == str


# @pytest.mark.parametrize(
#     ("document_txt", "generated_answer", "expected_output"),
#     [
#         (
#             "Source: https://link.com\nContent: RAG is good.",
#             "I love chicken wing.",
#             True,
#         ),
#         (
#             "Source: https://link.com\nContent: RAG is a tool used by AI Agent.",
#             "AI Agent can use RAG",
#             False,
#         ),
#     ],
# )
# def test_hallucination_grader_chain(document_txt, generated_answer, expected_output):
#     response = hallucination_grader_chain.invoke(
#         {
#             "document_txt": document_txt,
#             "generated_answer": generated_answer,
#         }
#     )
#     assert response.binary_flag == expected_output


# @pytest.mark.parametrize(
#     ("user_query", "generated_answer", "expected_output"),
#     [
#         (
#             "What does RAG do?",
#             "RAG is retrieval-augmented generation that allows AI agent to ground their answer based on valid context.",
#             True,
#         ),
#         (
#             "What is RAG?",
#             "I love chicken wing.",
#             False,
#         ),
#     ],
# )
# def test_usefulness_grader_chain(user_query, generated_answer, expected_output):
#     response = usefulness_grader_chain.invoke(
#         {
#             "user_query": user_query,
#             "generated_answer": generated_answer,
#         }
#     )
#     assert response.binary_flag == expected_output


# @pytest.mark.parametrize(
#     ("user_query", "expected_output"),
#     [
#         ("Prompt Engineering", 0),
#         ("Chicken Wing", 1),
#     ],
# )
# def test_router_chain(user_query, expected_output):
#     response: RouterResponseFormat = router_chain.invoke({"user_query": user_query})
#     assert response.route_number == expected_output
