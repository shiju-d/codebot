import re
import httpx


def extract_adf_text(node: dict | None) -> str:
    """Recursively extract plain text from Jira's Atlassian Document Format (ADF)."""
    if not node:
        return ''
    if node.get('type') == 'text':
        return node.get('text', '')
    if 'content' in node:
        return ' '.join(extract_adf_text(child) for child in node['content'])
    return ''


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
