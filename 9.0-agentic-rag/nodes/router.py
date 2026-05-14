from state.state import OverallState
from chains.router import RouterResponseFormat, router_chain


def router_node(state: OverallState):
    print("=== ROUTER NODE ===")
    user_query = state.user_query
    response: RouterResponseFormat = router_chain.invoke({"user_query": user_query})
    return {"entry_point": response.route_number}
