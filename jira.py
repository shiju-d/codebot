import re
import httpx


def parse_rca_input(input_str: str) -> tuple[str, str, str]:
    """Parse '<service>: <jira_url_or_key> [optional context]' → (service, issue_key, additional_context)."""
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
    additional_context = re.sub(r'^/[^\s]*', '', after_key).strip()
    return service, issue_key, additional_context


def extract_adf_text(node: dict | None) -> str:
    """Recursively extract plain text from Jira's Atlassian Document Format (ADF)."""
    if not node:
        return ''
    if node.get('type') == 'text':
        return node.get('text', '')
    if 'content' in node:
        return ' '.join(extract_adf_text(child) for child in node['content'])
    return ''


def build_rca_message(
    service: str,
    issue_key: str,
    summary: str,
    description: str,
    additional_context: str = '',
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


def md_to_jira(text: str) -> str:
    """Convert Markdown to Jira wiki markup."""
    # Extract fenced code blocks first so backticks inside them are not
    # converted to {{inline}} by the inline-code substitution below.
    fenced: list[str] = []

    def _store(m: re.Match) -> str:
        fenced.append(m.group(0))
        return f'__FENCED_{len(fenced) - 1}__'

    text = re.sub(r'```\w*\n[\s\S]*?```', _store, text)

    # Apply all other transformations on the placeholder-safe text.
    text = re.sub(r'`([^`\n]+)`', r'{{\1}}', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)
    text = re.sub(r'^### (.+)$', r'h3. \1', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'h2. \1', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$', r'h1. \1', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\. (.+)$', r'# \1', text, flags=re.MULTILINE)
    text = re.sub(r'^[ \t]*[-*] (.+)$', r'* \1', text, flags=re.MULTILINE)
    text = re.sub(r'^---+$', r'----', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Restore fenced blocks, converting them to Jira {code} syntax.
    def _convert(block: str) -> str:
        m = re.match(r'```(\w+)\n([\s\S]*?)```', block)
        if m:
            return f'{{code:{m.group(1)}}}\n{m.group(2)}{{code}}'
        m = re.match(r'```\n?([\s\S]*?)```', block)
        if m:
            return f'{{code}}\n{m.group(1)}{{code}}'
        return block

    for i, block in enumerate(fenced):
        text = text.replace(f'__FENCED_{i}__', _convert(block))

    return text.strip()


async def fetch_jira_issue(base_url: str, email: str, token: str, issue_key: str) -> dict:
    """Fetch a Jira issue via REST API v3. Raises httpx.HTTPStatusError on failure."""
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{base_url}/rest/api/3/issue/{issue_key}",
            auth=(email, token),
            timeout=30.0,
        )
        r.raise_for_status()
        return r.json()


async def post_jira_comment(base_url: str, email: str, token: str, issue_key: str, body: str) -> None:
    """Post a plain-text comment to a Jira issue via REST API v2. Raises httpx.HTTPStatusError on failure."""
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{base_url}/rest/api/2/issue/{issue_key}/comment",
            auth=(email, token),
            json={"body": body},
            timeout=30.0,
        )
        r.raise_for_status()
