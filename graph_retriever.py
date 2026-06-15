import neo4j

_EXPANSION_QUERY = """
MATCH (f:File)
WHERE f.path IN $seed_paths AND f.service = $service
OPTIONAL MATCH (f)-[:REQUIRES]->(dep:File)
OPTIONAL MATCH (caller:File)-[:REQUIRES]->(f)
OPTIONAL MATCH (f)-[:DEFINES]->(:Class)-[:HAS_METHOD]->(m:Method)
               -[:CALLS]->(target:Method)<-[:HAS_METHOD]-(:Class)
               <-[:DEFINES]-(target_file:File)
RETURN
  COLLECT(DISTINCT dep.path)         AS requires,
  COLLECT(DISTINCT caller.path)      AS required_by,
  COLLECT(DISTINCT target_file.path) AS call_targets
"""


def expand_file_paths(
    seed_paths: list[str],
    service: str,
    driver: neo4j.Driver,
    max_files: int = 5,
) -> list[str]:
    """Return up to max_files unique file paths reachable in 1 hop from seed_paths.

    Ranked: direct REQUIRES dependencies first, then CALLS targets, then callers.
    Seed paths are excluded from the result.
    """
    if not seed_paths:
        return []

    seed_set = set(seed_paths)

    with driver.session() as session:
        result = session.run(
            _EXPANSION_QUERY,
            seed_paths=seed_paths,
            service=service,
        )
        record = result.single()

    if not record:
        return []

    requires = record.get("requires") or []
    call_targets = record.get("call_targets") or []
    required_by = record.get("required_by") or []

    # Merge in priority order, deduplicate, strip None and seeds
    seen: set[str] = set()
    ordered: list[str] = []
    for path in requires + call_targets + required_by:
        if path and path not in seed_set and path not in seen:
            seen.add(path)
            ordered.append(path)

    return ordered[:max_files]
