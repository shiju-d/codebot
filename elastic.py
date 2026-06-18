import re
import httpx
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass
class TimeSpec:
    """Describes a log time range — either relative to now or absolute."""
    window: str | None = None     # relative: "2h", "30m", "1d"
    from_str: str | None = None   # absolute ISO string, e.g. "2024-01-15T10:00"
    to_str: str | None = None     # absolute ISO string, e.g. "2024-01-15T12:00"

    @property
    def is_relative(self) -> bool:
        return self.window is not None

    @property
    def is_absolute(self) -> bool:
        return self.from_str is not None and self.to_str is not None


_KV_RE = re.compile(r'(\w+):"([^"]*)"')


def extract_time_spec(text: str) -> tuple['TimeSpec | None', str]:
    """Extract time spec key-value pairs from text.

    Recognised keys:
      window:"2h"                               — relative, anchored to now
      from:"2024-01-15T10:00" to:"2024-01-15T12:00"  — absolute range

    Returns (TimeSpec | None, remaining_text_with_keys_stripped).
    """
    kvs = dict(_KV_RE.findall(text))
    remaining = _KV_RE.sub('', text).strip()

    if 'window' in kvs:
        return TimeSpec(window=kvs['window']), remaining
    if 'from' in kvs and 'to' in kvs:
        return TimeSpec(from_str=kvs['from'], to_str=kvs['to']), remaining
    return None, text


def parse_time_window(tw: str) -> timedelta:
    """Parse a relative time window string like '30m', '2h', '1d' into a timedelta."""
    m = re.match(r'^(\d+)([mhd])$', tw)
    if not m:
        raise ValueError(f"Invalid time window '{tw}'. Use format: 30m, 2h, 1d")
    n = int(m.group(1))
    unit = m.group(2)
    if unit == 'm':
        return timedelta(minutes=n)
    if unit == 'h':
        return timedelta(hours=n)
    return timedelta(days=n)


def resolve_time_spec(spec: TimeSpec) -> tuple[datetime, datetime]:
    """Resolve a TimeSpec to (from_dt, to_dt).

    Relative specs (window:"2h") anchor to now — independent of any Jira ticket.
    Absolute specs use the provided ISO strings directly.
    """
    if spec.is_relative:
        now = datetime.now(timezone.utc)
        return now - parse_time_window(spec.window), now
    return (
        datetime.fromisoformat(spec.from_str),
        datetime.fromisoformat(spec.to_str),
    )


async def fetch_elastic_logs(
    elastic_url: str,
    api_key: str,
    index: str,
    from_dt: datetime,
    to_dt: datetime,
    max_lines: int = 50,
) -> str:
    """Query Elasticsearch for ERROR/FATAL logs between from_dt and to_dt.
    Returns formatted log lines, or an empty string if none found.
    """
    query = {
        "query": {
            "bool": {
                "filter": [
                    {"range": {"@timestamp": {
                        "gte": from_dt.isoformat(),
                        "lte": to_dt.isoformat(),
                    }}},
                    {"terms": {"level": ["ERROR", "FATAL", "error", "fatal"]}},
                ]
            }
        },
        "size": max_lines,
        "sort": [{"@timestamp": {"order": "asc"}}],
        "_source": ["@timestamp", "message", "level"],
    }
    async with httpx.AsyncClient() as client:
        r = await client.post(
            f"{elastic_url.rstrip('/')}/{index}/_search",
            headers={"Authorization": f"ApiKey {api_key}"},
            json=query,
            timeout=15.0,
        )
        r.raise_for_status()

    hits = r.json().get("hits", {}).get("hits", [])
    lines = []
    for hit in hits:
        src = hit.get("_source", {})
        ts = src.get("@timestamp", "")
        level = src.get("level", "").upper()
        msg = src.get("message", "").strip()
        if msg:
            lines.append(f"[{ts}] {level}: {msg}")
    return "\n".join(lines)
