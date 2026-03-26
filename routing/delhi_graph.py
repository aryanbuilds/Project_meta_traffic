"""Delhi road-graph fetch and cache helpers."""

from pathlib import Path

GRAPH_PATH = Path("data/delhi_graph.graphml")


def get_graph(cache_path: str | None = None):
    import osmnx as ox

    path = Path(cache_path) if cache_path else GRAPH_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        graph = ox.load_graphml(path)
    else:
        graph = ox.graph_from_place("New Delhi, India", network_type="drive")
        ox.save_graphml(graph, path)

    print(f"Delhi graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    return graph
