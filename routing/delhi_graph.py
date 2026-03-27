"""Random routing environment generator for local development.

This replaces Delhi OSM dependency for now. It provides a weighted random graph
that can be cached and reloaded for repeatable routing tests.
"""

from __future__ import annotations

import os
import pickle
import random
from pathlib import Path

import networkx as nx

GRAPH_PATH = Path("data/random_test_graph.gpickle")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def build_random_graph(
    num_nodes: int = 120,
    num_edges: int = 320,
    seed: int = 42,
    min_length: float = 60.0,
    max_length: float = 1600.0,
    traffic_light_probability: float = 0.25,
) -> nx.Graph:
    rng = random.Random(seed)
    g = nx.gnm_random_graph(num_nodes, num_edges, seed=seed, directed=False)

    # Ensure connectivity by linking components if needed.
    components = [list(c) for c in nx.connected_components(g)]
    while len(components) > 1:
        a = components[0][0]
        b = components[1][0]
        g.add_edge(a, b)
        components = [list(c) for c in nx.connected_components(g)]

    for u, v in g.edges():
        g[u][v]["length"] = round(rng.uniform(min_length, max_length), 2)

    for node in g.nodes():
        has_light = rng.random() < traffic_light_probability
        g.nodes[node]["has_traffic_light"] = has_light
        if has_light:
            g.nodes[node]["signal_cycle_s"] = rng.choice([45, 60, 75, 90])

    return g


def save_graph(graph: nx.Graph, path: str | Path = GRAPH_PATH) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        pickle.dump(graph, f)
    return p


def load_graph(path: str | Path = GRAPH_PATH) -> nx.Graph:
    p = Path(path)
    with p.open("rb") as f:
        return pickle.load(f)


def get_graph(
    cache_path: str | Path | None = None,
    force_new: bool = False,
    num_nodes: int = 120,
    num_edges: int = 320,
    seed: int = 42,
) -> nx.Graph:
    path = Path(cache_path) if cache_path else GRAPH_PATH

    if path.exists() and not force_new:
        g = load_graph(path)
    else:
        env_nodes = max(2, _env_int("RANDOM_GRAPH_NODES", num_nodes))
        env_edges = max(1, _env_int("RANDOM_GRAPH_EDGES", num_edges))
        env_light_prob = _env_float("RANDOM_GRAPH_LIGHT_PROB", 0.25)
        env_light_prob = max(0.0, min(1.0, env_light_prob))
        g = build_random_graph(
            num_nodes=env_nodes,
            num_edges=env_edges,
            seed=seed,
            traffic_light_probability=env_light_prob,
        )
        save_graph(g, path)

    light_nodes = sum(1 for _, d in g.nodes(data=True) if d.get("has_traffic_light", False))
    print(
        "Random test graph: "
        f"{g.number_of_nodes()} nodes, {g.number_of_edges()} edges, {light_nodes} signalized junctions"
    )
    return g


def sample_route_test(graph: nx.Graph, seed: int = 42) -> dict:
    rng = random.Random(seed)
    nodes = list(graph.nodes())
    if len(nodes) < 2:
        raise ValueError("Graph must contain at least two nodes")

    origin = rng.choice(nodes)
    destination = rng.choice(nodes)
    while destination == origin:
        destination = rng.choice(nodes)

    route = nx.shortest_path(graph, origin, destination, weight="length")
    route_length = 0.0
    for u, v in zip(route, route[1:]):
        route_length += float(graph[u][v].get("length", 1.0))

    return {
        "origin": origin,
        "destination": destination,
        "route": route,
        "route_length": round(route_length, 2),
        "signalized_nodes_on_route": int(
            sum(1 for n in route if graph.nodes[n].get("has_traffic_light", False))
        ),
    }


if __name__ == "__main__":
    g = get_graph(force_new=True)
    result = sample_route_test(g)
    print(result)
