from chains.usefulness_grader import (
    UsefulnessGraderResponseFormat,
    usefulness_grader_chain,
)
from state.state import OverallState


def usefulness_grader_node(state: OverallState):
    print("=== USEFULNESS GRADER NODE ===")
    user_query = state.user_query
    generated_answer = state.generated_answer
    count = state.usefulness_check_count
    response: UsefulnessGraderResponseFormat = usefulness_grader_chain.invoke(
        {
            "user_query": user_query,
            "generated_answer": generated_answer,
        },
    )
    return {"is_useful": response.binary_flag, "usefulness_check_count": count + 1}
