"""Routing helpers for emergency corridor logic."""

import networkx as nx

DEMO_ROUTES = {
    "AIIMS": ["J3", "J2", "J1", "J0"],
    "GTB_HOSPITAL": ["J3", "J2", "J1"],
    "SAFDARJUNG": ["J2", "J1", "J0"],
}

DEMO_ROUTE_LENGTH_KM = {
    "AIIMS": 2.3,
    "GTB_HOSPITAL": 1.8,
    "SAFDARJUNG": 2.0,
}


def compute_route(graph, origin_node, dest_node) -> list[int]:
    return nx.shortest_path(graph, origin_node, dest_node, weight="length")


def get_demo_route(destination: str) -> list[str]:
    key = destination.strip().upper()
    if key not in DEMO_ROUTES:
        raise ValueError(f"Unknown destination: {destination}")
    return list(DEMO_ROUTES[key])


def route_to_prompt_text(route: list[str], destination: str = "AIIMS") -> str:
    key = destination.strip().upper()
    km = DEMO_ROUTE_LENGTH_KM.get(key)
    route_str = " -> ".join(route)
    if km is None:
        return f"{route_str} ({destination})"
    return f"{route_str} ({destination}, {km:.1f} km)"
