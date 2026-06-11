def parse_message(raw: str) -> tuple:
    """Parse 'service: message' into (service_name, message). Raises ValueError if prefix missing."""
    if ":" not in raw:
        raise ValueError("missing_prefix")
    service_name, _, message = raw.partition(":")
    service_name = service_name.strip().lower()
    message = message.strip()
    if not service_name or not message:
        raise ValueError("missing_prefix")
    return service_name, message
