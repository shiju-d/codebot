from unittest.mock import MagicMock, patch

from llama_index.core.schema import NodeWithScore, TextNode, QueryBundle

from graph_postprocessor import GraphExpansionPostprocessor


def _make_node(file_path: str, text: str = "code snippet") -> NodeWithScore:
    return NodeWithScore(
        node=TextNode(text=text, metadata={"file_path": file_path}),
        score=0.8,
    )


def test_returns_original_nodes_when_driver_is_none():
    pp = GraphExpansionPostprocessor(
        service_name="rover-ifc",
        chroma_collection=MagicMock(),
        driver=None,
    )
    nodes = [_make_node("rover-ifc/test/integration/channex_test.rb")]
    result = pp._postprocess_nodes(nodes)
    assert result == nodes


def test_returns_original_nodes_when_nodes_empty():
    pp = GraphExpansionPostprocessor(
        service_name="rover-ifc",
        chroma_collection=MagicMock(),
        driver=MagicMock(),
    )
    result = pp._postprocess_nodes([])
    assert result == []


def test_merges_expanded_nodes():
    mock_collection = MagicMock()
    mock_collection.get.return_value = {
        "documents": ["def build_occupancy_rates..."],
        "metadatas": [{"file_path": "rover-ifc/lib/snt/channex/exporters/rate_exporter.rb"}],
    }

    with patch("graph_postprocessor.expand_file_paths") as mock_expand:
        mock_expand.return_value = ["rover-ifc/lib/snt/channex/exporters/rate_exporter.rb"]
        pp = GraphExpansionPostprocessor(
            service_name="rover-ifc",
            chroma_collection=mock_collection,
            driver=MagicMock(),
        )
        nodes = [_make_node("rover-ifc/test/integration/channex_test.rb")]
        result = pp._postprocess_nodes(nodes)

    assert len(result) == 2
    paths = [n.node.metadata["file_path"] for n in result]
    assert "rover-ifc/lib/snt/channex/exporters/rate_exporter.rb" in paths


def test_degrades_gracefully_on_neo4j_failure():
    with patch("graph_postprocessor.expand_file_paths", side_effect=Exception("Neo4j down")):
        pp = GraphExpansionPostprocessor(
            service_name="rover-ifc",
            chroma_collection=MagicMock(),
            driver=MagicMock(),
        )
        nodes = [_make_node("rover-ifc/lib/foo.rb")]
        result = pp._postprocess_nodes(nodes)

    assert result == nodes


def test_does_not_fetch_when_expanded_paths_are_all_seeds():
    mock_collection = MagicMock()

    with patch("graph_postprocessor.expand_file_paths") as mock_expand:
        mock_expand.return_value = ["rover-ifc/test/integration/channex_test.rb"]
        pp = GraphExpansionPostprocessor(
            service_name="rover-ifc",
            chroma_collection=mock_collection,
            driver=MagicMock(),
        )
        nodes = [_make_node("rover-ifc/test/integration/channex_test.rb")]
        result = pp._postprocess_nodes(nodes)

    mock_collection.get.assert_not_called()
    assert len(result) == 1


def test_degrades_gracefully_on_chromadb_failure():
    mock_collection = MagicMock()
    mock_collection.get.side_effect = Exception("ChromaDB error")

    with patch("graph_postprocessor.expand_file_paths") as mock_expand:
        mock_expand.return_value = ["rover-ifc/lib/rate_exporter.rb"]
        pp = GraphExpansionPostprocessor(
            service_name="rover-ifc",
            chroma_collection=mock_collection,
            driver=MagicMock(),
        )
        nodes = [_make_node("rover-ifc/test/channex_test.rb")]
        result = pp._postprocess_nodes(nodes)

    assert result == nodes


def test_expanded_nodes_get_score_zero():
    mock_collection = MagicMock()
    mock_collection.get.return_value = {
        "documents": ["expanded code"],
        "metadatas": [{"file_path": "rover-ifc/lib/rate_exporter.rb"}],
    }

    with patch("graph_postprocessor.expand_file_paths") as mock_expand:
        mock_expand.return_value = ["rover-ifc/lib/rate_exporter.rb"]
        pp = GraphExpansionPostprocessor(
            service_name="rover-ifc",
            chroma_collection=mock_collection,
            driver=MagicMock(),
        )
        nodes = [_make_node("rover-ifc/test/channex_test.rb")]
        result = pp._postprocess_nodes(nodes)

    expanded = [n for n in result if n.node.metadata.get("file_path") == "rover-ifc/lib/rate_exporter.rb"]
    assert len(expanded) == 1
    assert expanded[0].score == 0.0
