from routing.delhi_graph import get_graph


def test_random_graph_invalid_env_values_fall_back(monkeypatch, tmp_path):
    monkeypatch.setenv("RANDOM_GRAPH_NODES", "not_an_int")
    monkeypatch.setenv("RANDOM_GRAPH_EDGES", "bad")
    monkeypatch.setenv("RANDOM_GRAPH_LIGHT_PROB", "oops")

    graph = get_graph(
        cache_path=tmp_path / "graph_fallback.gpickle",
        force_new=True,
        num_nodes=11,
        num_edges=22,
    )

    assert graph.number_of_nodes() == 11
    assert graph.number_of_edges() >= 10


def test_random_graph_light_prob_clamped(monkeypatch, tmp_path):
    monkeypatch.setenv("RANDOM_GRAPH_NODES", "10")
    monkeypatch.setenv("RANDOM_GRAPH_EDGES", "15")
    monkeypatch.setenv("RANDOM_GRAPH_LIGHT_PROB", "1.7")

    graph = get_graph(
        cache_path=tmp_path / "graph_clamped.gpickle",
        force_new=True,
    )

    assert all(data.get("has_traffic_light", False) for _, data in graph.nodes(data=True))
