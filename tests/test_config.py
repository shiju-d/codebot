import os
import tempfile
import pytest
import yaml
from config import load_services, ServiceConfig


def _write_yaml(data):
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.dump(data, f)
    f.close()
    return f.name


def test_load_single_service():
    path = _write_yaml({"services": [
        {"name": "ibe", "system_prompt": "IBE prompt.", "repos": ["/repos/ibe-api", "/repos/ibe-frontend"]}
    ]})
    try:
        result = load_services(path)
        assert len(result) == 1
        assert isinstance(result[0], ServiceConfig)
        assert result[0].name == "ibe"
        assert result[0].system_prompt == "IBE prompt."
        assert result[0].repos == ["/repos/ibe-api", "/repos/ibe-frontend"]
    finally:
        os.unlink(path)


def test_load_multiple_services():
    path = _write_yaml({"services": [
        {"name": "ibe", "system_prompt": "IBE prompt.", "repos": ["/repos/ibe-api"]},
        {"name": "pms", "system_prompt": "PMS prompt.", "repos": ["/repos/pms-api", "/repos/pms-frontend"]},
    ]})
    try:
        result = load_services(path)
        assert len(result) == 2
        assert result[0].name == "ibe"
        assert result[1].name == "pms"
        assert len(result[1].repos) == 2
    finally:
        os.unlink(path)


def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        load_services("/nonexistent/path/services.yaml")


def test_system_prompt_preserved_verbatim():
    path = _write_yaml({"services": [
        {"name": "test", "system_prompt": "  prompt with spaces  ", "repos": ["/r/a"]}
    ]})
    try:
        result = load_services(path)
        assert result[0].system_prompt == "  prompt with spaces  "
    finally:
        os.unlink(path)
