from collections import OrderedDict

import chromadb
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core.chat_engine import ContextChatEngine

from config import load_services
import graph as _graph
from graph_postprocessor import GraphExpansionPostprocessor
from llms import local_llm, bedrock_llm, neo4j_driver, get_reranker, SERVICES_CONFIG_PATH
from indexer import build_service_index

MAX_SESSIONS = 100

# { service_name: { "index": VectorStoreIndex, "system_prompt": str,
#                   "sessions": { "local": OrderedDict, "bedrock": OrderedDict } } }
services: dict = {}


def get_engine(session_id: str, service_name: str, llm, llm_key: str):
    sessions = services[service_name]["sessions"][llm_key]
    if session_id not in sessions:
        if len(sessions) >= MAX_SESSIONS:
            sessions.popitem(last=False)
        memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
        svc = services[service_name]
        base_retriever = svc["index"].as_retriever(similarity_top_k=10)
        fusion_retriever = QueryFusionRetriever(
            retrievers=[base_retriever],
            llm=llm,
            num_queries=3,
            mode=FUSION_MODES.RECIPROCAL_RANK,
            use_async=True,
            similarity_top_k=30,
        )
        postprocessors = []
        if svc.get("graph_postprocessor"):
            postprocessors.append(svc["graph_postprocessor"])
        postprocessors.append(get_reranker())
        sessions[session_id] = {
            "memory": memory,
            "engine": ContextChatEngine.from_defaults(
                retriever=fusion_retriever,
                llm=llm,
                memory=memory,
                node_postprocessors=postprocessors,
                system_prompt=svc["system_prompt"],
            ),
        }
    return sessions[session_id]["engine"]


def _make_service_entry(svc, index, chroma_client=None) -> dict:
    graph_pp = None
    if neo4j_driver:
        if chroma_client is None:
            chroma_client = chromadb.PersistentClient(path="/app/chroma_db")
        collection = chroma_client.get_or_create_collection(f"{svc.name}_codebase")
        graph_pp = GraphExpansionPostprocessor(
            service_name=svc.name,
            chroma_collection=collection,
            driver=neo4j_driver,
        )
    return {
        "index": index,
        "system_prompt": svc.system_prompt,
        "graph_postprocessor": graph_pp,
        "sessions": {
            "local": OrderedDict(),
            "bedrock": OrderedDict(),
        },
    }


def init_all_services():
    global services
    configs = load_services(SERVICES_CONFIG_PATH)
    chroma_client = chromadb.PersistentClient(path="/app/chroma_db")
    for svc in configs:
        index = build_service_index(svc)

        if neo4j_driver:
            try:
                _graph.build_service_graph(svc, neo4j_driver)
            except Exception as exc:
                print(f"[graph:{svc.name}] Graph build failed ({exc}); continuing without graph")

        services[svc.name] = _make_service_entry(svc, index, chroma_client)
    print(f"codebot ready. Services: {list(services.keys())}")


def reindex_all_services() -> list[str]:
    global services
    chroma_client = chromadb.PersistentClient(path="/app/chroma_db")
    configs = load_services(SERVICES_CONFIG_PATH)
    for svc in configs:
        try:
            chroma_client.delete_collection(f"{svc.name}_codebase")
        except Exception as e:
            print(f"[reindex] Warning: could not delete collection {svc.name}_codebase: {e}")
        if neo4j_driver:
            try:
                _graph.clear_service_graph(svc.name, neo4j_driver)
            except Exception as e:
                print(f"[reindex] Warning: could not clear graph {svc.name}: {e}")

    new_services: dict = {}
    for svc in configs:
        index = build_service_index(svc)
        if neo4j_driver:
            try:
                _graph.build_service_graph(svc, neo4j_driver)
            except Exception as exc:
                print(f"[graph:{svc.name}] Graph build failed ({exc}); continuing without graph")
        new_services[svc.name] = _make_service_entry(svc, index, chroma_client)
    services = new_services
    print(f"codebot reindexed. Services: {list(services.keys())}")
    return list(services.keys())


def reindex_one_service(service_name: str):
    configs = load_services(SERVICES_CONFIG_PATH)
    svc_config = next((s for s in configs if s.name == service_name), None)
    if not svc_config:
        valid = ", ".join(s.name for s in configs)
        raise ValueError(f"Unknown service '{service_name}'. Valid services: {valid}")

    chroma_client = chromadb.PersistentClient(path="/app/chroma_db")
    try:
        chroma_client.delete_collection(f"{service_name}_codebase")
    except Exception as e:
        print(f"[reindex] Warning: could not delete collection {service_name}_codebase: {e}")

    index = build_service_index(svc_config)

    if neo4j_driver:
        try:
            _graph.clear_service_graph(service_name, neo4j_driver)
            _graph.build_service_graph(svc_config, neo4j_driver)
        except Exception as exc:
            print(f"[graph:{service_name}] Graph rebuild failed ({exc}); continuing without graph")

    services[service_name] = _make_service_entry(svc_config, index, chroma_client)
