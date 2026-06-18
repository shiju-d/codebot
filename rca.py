import re

from elastic import TimeSpec, extract_time_spec


def parse_rca_input(input_str: str) -> tuple[str, str, str, 'TimeSpec | None']:
    """Parse '<service>: <jira_key> [time keys] [optional context]'
    → (service, issue_key, additional_context, TimeSpec | None).

    Time range is expressed with explicit key:"value" pairs anywhere after
    the ticket key:
      window:"2h"                              — relative to now (for Elasticsearch lookup)
      from:"2024-01-15T10:00" to:"2024-01-15T12:00"  — absolute range
    """
    colon_idx = input_str.find(':')
    if colon_idx == -1:
        raise ValueError("Format: <service>: <jira_url_or_key>  e.g. ibe: IBE-1152")
    service = input_str[:colon_idx].strip().lower()
    if not service:
        raise ValueError("Format: <service>: <jira_url_or_key>  e.g. ibe: IBE-1152")
    rest = input_str[colon_idx + 1:].strip()
    match = re.search(r'([A-Z]+-\d+)', rest)
    if not match:
        raise ValueError(f"Could not find a Jira issue key (e.g. IBE-1152) in: {rest}")
    issue_key = match.group(1)
    after_key = rest[match.end():].strip()
    after_key = re.sub(r'^/[^\s]*', '', after_key).strip()

    time_spec, additional_context = extract_time_spec(after_key)
    return service, issue_key, additional_context, time_spec


def build_rca_message(
    service: str,
    issue_key: str,
    summary: str,
    description: str,
    additional_context: str = '',
    log_context: str = '',
) -> str:
    """Build the RAG query message sent to the LLM for RCA."""
    parts = [
        f"{service}: You are performing Root Cause Analysis (RCA) for a production bug. Analyse the codebase carefully.",
        "",
        f"Ticket: {issue_key}",
        f"Summary: {summary}",
        "",
        "Description:",
        description,
    ]
    if log_context:
        parts.extend(["", "Error logs from Elasticsearch:", log_context])
    if additional_context:
        parts.extend(["", "Additional context provided by reporter:", additional_context])
    parts.extend([
        "",
        "IMPORTANT: Base your entire analysis on the code snippets retrieved for you. Only reference",
        "file paths that appear verbatim in those snippets. If a relevant file is not in the context,",
        'say "not retrieved" rather than guessing a path.',
        "",
        "Based on the retrieved code, answer:",
        "1. Which files and functions (visible in the context) are involved in this flow?",
        "2. Where exactly is the root cause (file path and function from the context)?",
        "3. What is the precise fix with a code snippet from the retrieved code?",
    ])
    return "\n".join(parts)
