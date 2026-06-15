from typing import Any, Optional

import neo4j
from llama_index.core.bridge.pydantic import PrivateAttr
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from graph_retriever import expand_file_paths


class GraphExpansionPostprocessor(BaseNodePostprocessor):
    """Expand retrieval results by 1 hop in the code knowledge graph.

    For each file path in the vector-search hits, queries Neo4j for directly
    required/requiring files and method-call targets, fetches their chunks from
    ChromaDB, and merges them with the original nodes before reranking.

    If Neo4j is unreachable the original nodes are returned unchanged.
    """

    max_expanded_files: int = 5
    max_expanded_chunks: int = 15

    _service_name: str = PrivateAttr()
    _collection: Any = PrivateAttr()   # chromadb.Collection
    _driver: Any = PrivateAttr()       # neo4j.Driver | None

    def __init__(
        self,
        service_name: str,
        chroma_collection: Any,
        driver: Any,
        max_expanded_files: int = 5,
        max_expanded_chunks: int = 15,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            max_expanded_files=max_expanded_files,
            max_expanded_chunks=max_expanded_chunks,
            **kwargs,
        )
        self._service_name = service_name
        self._collection = chroma_collection
        self._driver = driver

    @classmethod
    def class_name(cls) -> str:
        return "GraphExpansionPostprocessor"

    def _postprocess_nodes(
        self,
        nodes: list[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> list[NodeWithScore]:
        if not self._driver or not nodes:
            return nodes

        seed_paths = list({
            n.node.metadata.get("file_path")
            for n in nodes
            if n.node.metadata.get("file_path")
        })
        if not seed_paths:
            return nodes

        try:
            expanded_paths = expand_file_paths(
                seed_paths=seed_paths,
                service=self._service_name,
                driver=self._driver,
                max_files=self.max_expanded_files,
            )
        except Exception as exc:
            print(f"[graph_postprocessor] Neo4j expansion failed: {exc}")
            return nodes

        new_paths = [p for p in expanded_paths if p not in set(seed_paths)]
        if not new_paths:
            return nodes

        try:
            results = self._collection.get(
                where={"file_path": {"$in": new_paths}},
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            print(f"[graph_postprocessor] ChromaDB fetch failed: {exc}")
            return nodes

        documents = (results.get("documents") or [])[:self.max_expanded_chunks]
        metadatas = (results.get("metadatas") or [])[:self.max_expanded_chunks]

        expanded_nodes = [
            NodeWithScore(node=TextNode(text=doc, metadata=meta or {}), score=0.0)
            for doc, meta in zip(documents, metadatas)
            if doc
        ]

        return nodes + expanded_nodes
