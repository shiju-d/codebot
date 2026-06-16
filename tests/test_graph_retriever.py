from unittest.mock import MagicMock
from graph_retriever import expand_file_paths


def _make_mock_driver(requires=None, required_by=None, call_targets=None):
    record = MagicMock()
    record.get.side_effect = lambda k: {
        "requires": requires or [],
        "required_by": required_by or [],
        "call_targets": call_targets or [],
    }.get(k, [])

    mock_result = MagicMock()
    mock_result.single.return_value = record

    session = MagicMock()
    session.run.return_value = mock_result
    session.__enter__ = MagicMock(return_value=session)
    session.__exit__ = MagicMock(return_value=False)

    driver = MagicMock()
    driver.session.return_value = session
    return driver


def test_expand_returns_requires_paths():
    driver = _make_mock_driver(
        requires=["rover-ifc/lib/snt/channex/exporters/rate_exporter.rb"]
    )
    result = expand_file_paths(
        seed_paths=["rover-ifc/test/integration/channex_test.rb"],
        service="rover-ifc",
        driver=driver,
        max_files=5,
    )
    assert "rover-ifc/lib/snt/channex/exporters/rate_exporter.rb" in result


def test_expand_excludes_seed_paths():
    driver = _make_mock_driver(
        required_by=["rover-ifc/test/integration/channex_test.rb"]
    )
    result = expand_file_paths(
        seed_paths=["rover-ifc/test/integration/channex_test.rb"],
        service="rover-ifc",
        driver=driver,
        max_files=5,
    )
    assert "rover-ifc/test/integration/channex_test.rb" not in result


def test_expand_caps_at_max_files():
    many_paths = [f"rover-ifc/lib/file{i}.rb" for i in range(20)]
    driver = _make_mock_driver(requires=many_paths)
    result = expand_file_paths(
        seed_paths=["rover-ifc/test/integration/channex_test.rb"],
        service="rover-ifc",
        driver=driver,
        max_files=3,
    )
    assert len(result) <= 3


def test_expand_returns_empty_on_no_neighbours():
    driver = _make_mock_driver()
    result = expand_file_paths(
        seed_paths=["rover-ifc/lib/isolated.rb"],
        service="rover-ifc",
        driver=driver,
        max_files=5,
    )
    assert result == []


def test_expand_returns_empty_on_no_seed_paths():
    driver = _make_mock_driver(requires=["some/path.rb"])
    result = expand_file_paths(
        seed_paths=[],
        service="rover-ifc",
        driver=driver,
        max_files=5,
    )
    assert result == []


def test_expand_deduplicates_paths():
    driver = _make_mock_driver(
        requires=["rover-ifc/lib/foo.rb"],
        call_targets=["rover-ifc/lib/foo.rb"],
    )
    result = expand_file_paths(
        seed_paths=["rover-ifc/test/integration/channex_test.rb"],
        service="rover-ifc",
        driver=driver,
        max_files=5,
    )
    assert result.count("rover-ifc/lib/foo.rb") == 1


def test_expand_prioritises_requires_over_callers():
    driver = _make_mock_driver(
        requires=["rover-ifc/lib/dep.rb"],
        required_by=["rover-ifc/lib/caller.rb"],
    )
    result = expand_file_paths(
        seed_paths=["rover-ifc/lib/seed.rb"],
        service="rover-ifc",
        driver=driver,
        max_files=1,
    )
    assert result == ["rover-ifc/lib/dep.rb"]
