import os
import pytest

from config import ServiceConfig

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "codebot-secret")

pytestmark = pytest.mark.skipif(
    not NEO4J_URI,
    reason="NEO4J_URI not set — skipping graph integration tests",
)


@pytest.fixture
def driver():
    import neo4j as _neo4j
    d = _neo4j.GraphDatabase.driver(NEO4J_URI, auth=("neo4j", NEO4J_PASSWORD))
    yield d
    d.close()


@pytest.fixture
def fixture_repo(tmp_path):
    """Create a minimal repo structure mimicking rover-ifc."""
    lib_dir = tmp_path / "lib" / "snt" / "channex" / "exporters"
    lib_dir.mkdir(parents=True)
    test_dir = tmp_path / "test" / "integration"
    test_dir.mkdir(parents=True)

    (lib_dir / "rate_exporter.rb").write_text("""
class RateExporter
  def build_occupancy_rates(room_rate_data)
    single_rate = room_rate_data[:single_amount]&.to_f
  end
end
""")

    (test_dir / "channex_test.rb").write_text("""
require 'snt/channex/exporters/rate_exporter'

class ChannexTest
  def test_rate_building
    exporter = RateExporter.new
  end
end
""")

    return tmp_path


@pytest.fixture
def svc(fixture_repo):
    return ServiceConfig(
        name="test-service",
        repos=[str(fixture_repo)],
        system_prompt="Test",
        file_extensions=[".rb"],
        jira_project_key="TEST",
    )


def test_rate_exporter_surfaces_from_channex_test(driver, svc):
    from graph import build_service_graph, clear_service_graph
    from graph_retriever import expand_file_paths

    clear_service_graph("test-service", driver)
    build_service_graph(svc, driver)

    repo_path = svc.repos[0]
    repos_prefix = os.path.dirname(repo_path).rstrip("/") + "/"

    test_abs = os.path.join(repo_path, "test/integration/channex_test.rb")
    seed_rel = test_abs.removeprefix(repos_prefix)

    expanded = expand_file_paths(
        seed_paths=[seed_rel],
        service="test-service",
        driver=driver,
        max_files=5,
    )

    exporter_abs = os.path.join(repo_path, "lib/snt/channex/exporters/rate_exporter.rb")
    exporter_rel = exporter_abs.removeprefix(repos_prefix)

    assert exporter_rel in expanded, (
        f"rate_exporter.rb ({exporter_rel!r}) not found in expansion: {expanded}"
    )

    clear_service_graph("test-service", driver)
