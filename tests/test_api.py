from collections import OrderedDict
from unittest.mock import patch, MagicMock, AsyncMock
import pytest
from fastapi.testclient import TestClient
import runner


def _fresh_mock_services():
    return {
        "ibe": {
            "index": MagicMock(),
            "system_prompt": "You are IBE expert.",
            "sessions": {
                "local": OrderedDict(),
                "claude": OrderedDict(),
                "bedrock": OrderedDict(),
            },
        }
    }


@pytest.fixture
def client():
    runner.services = _fresh_mock_services()
    with patch("runner._init_all_services"):
        with TestClient(runner.app) as c:
            yield c


def test_list_services(client):
    response = client.get("/services")
    assert response.status_code == 200
    assert response.json() == {"services": ["ibe"]}


def test_chat_missing_prefix(client):
    response = client.post("/chat", json={"message": "why is checkout failing?", "session_id": "t"})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Missing service prefix" in detail
    assert "ibe" in detail


def test_chat_unknown_service(client):
    response = client.post("/chat", json={"message": "xyz: hello", "session_id": "t"})
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Unknown service 'xyz'" in detail
    assert "ibe" in detail


def test_chat_bedrock_not_configured(client):
    original = runner.bedrock_llm
    runner.bedrock_llm = None
    try:
        response = client.post("/chat/bedrock", json={"message": "ibe: hello", "session_id": "t"})
        assert response.status_code == 503
        assert "AWS_ACCESS_KEY_ID" in response.json()["detail"]
    finally:
        runner.bedrock_llm = original


def test_reindex_unknown_service(client):
    response = client.post("/reindex/xyz")
    assert response.status_code == 400
    assert "Unknown service 'xyz'" in response.json()["detail"]


def test_clear_session(client):
    runner.services["ibe"]["sessions"]["local"]["my-session"] = {"memory": None, "engine": None}
    response = client.delete("/session/my-session")
    assert response.status_code == 200
    assert response.json() == {"cleared": "my-session"}
    assert "my-session" not in runner.services["ibe"]["sessions"]["local"]


def test_chat_valid_service_returns_response(client):
    mock_node = MagicMock()
    mock_node.metadata = {"file_path": "/repos/ibe-api/src/services/cart.service.ts"}

    mock_response = MagicMock()
    mock_response.response = "The bug is in cart.service.ts line 42."
    mock_response.source_nodes = [mock_node]

    mock_engine = MagicMock()
    mock_engine.chat.return_value = mock_response

    with patch("runner._get_engine", return_value=mock_engine):
        response = client.post("/chat", json={"message": "ibe: why is checkout failing?", "session_id": "t"})

    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "The bug is in cart.service.ts line 42."
    assert "/repos/ibe-api/src/services/cart.service.ts" in data["sources"]


def test_get_reranker_returns_singleton():
    runner._reranker = None  # ensure clean state
    with patch("runner.FlagEmbeddingReranker") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        first = runner._get_reranker()
        second = runner._get_reranker()
    assert first is second
    mock_cls.assert_called_once_with(model="BAAI/bge-reranker-base", top_n=12)
    runner._reranker = None  # restore clean state
