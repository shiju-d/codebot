import pytest
from elastic import parse_time_window, resolve_time_spec, TimeSpec
from jira import extract_adf_text, md_to_jira
from rca import parse_rca_input, build_rca_message
from datetime import datetime, timedelta, timezone


# --- parse_rca_input ---

def test_parse_rca_bare_key():
    service, key, ctx, tw = parse_rca_input("ibe: IBE-1152")
    assert service == "ibe"
    assert key == "IBE-1152"
    assert ctx == ""
    assert tw is None

def test_parse_rca_full_url():
    service, key, ctx, tw = parse_rca_input(
        "ibe: https://stayntouch.atlassian.net/browse/IBE-1152"
    )
    assert service == "ibe"
    assert key == "IBE-1152"
    assert ctx == ""
    assert tw is None

def test_parse_rca_with_additional_context():
    service, key, ctx, tw = parse_rca_input(
        "ibe: IBE-1152\nStack trace: TypeError at line 45"
    )
    assert service == "ibe"
    assert key == "IBE-1152"
    assert ctx == "Stack trace: TypeError at line 45"
    assert tw is None

def test_parse_rca_uppercase_service_normalised():
    service, key, _, _tw = parse_rca_input("IBE: IBE-1152")
    assert service == "ibe"

def test_parse_rca_missing_colon_raises():
    with pytest.raises(ValueError, match="Format"):
        parse_rca_input("ibe IBE-1152")

def test_parse_rca_missing_key_raises():
    with pytest.raises(ValueError, match="Jira issue key"):
        parse_rca_input("ibe: some random text")

def test_parse_rca_empty_service_raises():
    with pytest.raises(ValueError):
        parse_rca_input(": IBE-1152")

def test_parse_rca_context_after_url():
    service, key, ctx, tw = parse_rca_input(
        "ibe: IBE-1152\nStack trace line 1\nStack trace line 2"
    )
    assert service == "ibe"
    assert key == "IBE-1152"
    assert "Stack trace line 1" in ctx
    assert tw is None

def test_parse_rca_relative_window():
    service, key, ctx, spec = parse_rca_input('rover-ifc: CICO-134027 window:"2h"')
    assert service == "rover-ifc"
    assert key == "CICO-134027"
    assert spec.is_relative
    assert spec.window == "2h"
    assert ctx == ""

def test_parse_rca_relative_window_minutes():
    _, _, _, spec = parse_rca_input('ibe: IBE-1152 window:"30m"')
    assert spec.window == "30m"

def test_parse_rca_relative_window_with_context():
    _, _, ctx, spec = parse_rca_input('rover-ifc: CICO-134027 window:"2h" focus on channex')
    assert spec.window == "2h"
    assert ctx == "focus on channex"

def test_parse_rca_absolute_timeframe():
    _, _, ctx, spec = parse_rca_input(
        'rover-ifc: CICO-134027 from:"2024-01-15T10:00" to:"2024-01-15T12:00"'
    )
    assert spec.is_absolute
    assert spec.from_str == "2024-01-15T10:00"
    assert spec.to_str == "2024-01-15T12:00"
    assert ctx == ""

def test_parse_rca_absolute_timeframe_with_context():
    _, _, ctx, spec = parse_rca_input(
        'rover-ifc: CICO-134027 from:"2024-01-15T10:00" to:"2024-01-15T12:00" focus on rates'
    )
    assert spec.is_absolute
    assert ctx == "focus on rates"

def test_parse_rca_keys_anywhere_in_string():
    _, _, ctx, spec = parse_rca_input(
        'rover-ifc: CICO-134027 focus on rates window:"1h"'
    )
    assert spec.window == "1h"
    assert ctx == "focus on rates"

def test_parse_rca_no_time_spec():
    _, _, ctx, spec = parse_rca_input("rover-ifc: CICO-134027 focus on channex")
    assert spec is None
    assert ctx == "focus on channex"


# --- parse_time_window ---

def test_parse_time_window_hours():
    assert parse_time_window("2h") == timedelta(hours=2)

def test_parse_time_window_minutes():
    assert parse_time_window("30m") == timedelta(minutes=30)

def test_parse_time_window_days():
    assert parse_time_window("1d") == timedelta(days=1)

def test_parse_time_window_invalid_raises():
    with pytest.raises(ValueError, match="Invalid time window"):
        parse_time_window("2x")


# --- resolve_time_spec ---

def test_resolve_time_spec_relative():
    before = datetime.now(timezone.utc)
    from_dt, to_dt = resolve_time_spec(TimeSpec(window="2h"))
    after = datetime.now(timezone.utc)
    assert before <= to_dt <= after
    assert abs((to_dt - from_dt).total_seconds() - 7200) < 2

def test_resolve_time_spec_absolute():
    spec = TimeSpec(from_str="2024-01-15T08:00", to_str="2024-01-15T10:00")
    from_dt, to_dt = resolve_time_spec(spec)
    assert from_dt == datetime(2024, 1, 15, 8, 0, 0)
    assert to_dt == datetime(2024, 1, 15, 10, 0, 0)


# --- extract_adf_text ---

def test_extract_adf_plain_text():
    node = {"type": "text", "text": "hello"}
    assert extract_adf_text(node) == "hello"

def test_extract_adf_nested():
    node = {
        "type": "doc",
        "content": [
            {"type": "paragraph", "content": [
                {"type": "text", "text": "Error "},
                {"type": "text", "text": "occurs here"},
            ]},
        ],
    }
    result = extract_adf_text(node)
    assert "Error" in result
    assert "occurs here" in result
    assert result == "Error  occurs here"

def test_extract_adf_none_returns_empty():
    assert extract_adf_text(None) == ""

def test_extract_adf_no_content_returns_empty():
    assert extract_adf_text({"type": "hardBreak"}) == ""


# --- build_rca_message ---

def test_build_rca_message_contains_service_prefix():
    msg = build_rca_message("ibe", "IBE-1152", "Checkout fails", "User gets 500", "")
    assert msg.startswith("ibe: You are performing Root Cause Analysis")

def test_build_rca_message_contains_ticket_info():
    msg = build_rca_message("ibe", "IBE-1152", "Checkout fails", "User gets 500", "")
    assert "Ticket: IBE-1152" in msg
    assert "Summary: Checkout fails" in msg
    assert "User gets 500" in msg

def test_build_rca_message_contains_grounding_instruction():
    msg = build_rca_message("ibe", "IBE-1152", "Checkout fails", "Desc", "")
    assert "IMPORTANT" in msg
    assert "not retrieved" in msg

def test_build_rca_message_includes_additional_context():
    msg = build_rca_message("ibe", "IBE-1152", "Summary", "Desc", "Stack trace: ...")
    assert "Additional context provided by reporter" in msg
    assert "Stack trace: ..." in msg

def test_build_rca_message_no_additional_context_section_when_empty():
    msg = build_rca_message("ibe", "IBE-1152", "Summary", "Desc", "")
    assert "Additional context" not in msg

def test_build_rca_message_includes_log_context():
    msg = build_rca_message("ibe", "IBE-1152", "Summary", "Desc", "", "[2024-01-15] ERROR: boom")
    assert "Error logs from Elasticsearch" in msg
    assert "[2024-01-15] ERROR: boom" in msg

def test_build_rca_message_no_log_section_when_empty():
    msg = build_rca_message("ibe", "IBE-1152", "Summary", "Desc", "")
    assert "Elasticsearch" not in msg


# --- md_to_jira ---

def test_md_to_jira_heading_h1():
    assert md_to_jira("# H1") == "h1. H1"

def test_md_to_jira_heading_h2():
    assert md_to_jira("## H2") == "h2. H2"

def test_md_to_jira_heading_h3():
    assert md_to_jira("### H3") == "h3. H3"

def test_md_to_jira_bold():
    assert md_to_jira("**bold**") == "*bold*"

def test_md_to_jira_inline_code():
    assert md_to_jira("`foo`") == "{{foo}}"

def test_md_to_jira_fenced_code_with_language():
    result = md_to_jira("```python\nprint('hi')\n```")
    assert "{code:python}" in result
    assert "print('hi')" in result
    assert result.endswith("{code}")

def test_md_to_jira_fenced_code_no_language():
    result = md_to_jira("```\nsome code\n```")
    assert "{code}" in result
    assert "some code" in result

def test_md_to_jira_fenced_code_backticks_not_converted():
    # Backticks inside a fenced block must not be converted to {{}}
    text = "```python\nresult = obj.method(`key`)\n```"
    result = md_to_jira(text)
    assert "{code:python}" in result
    # The backtick inside the code block must not become {{key}}
    assert "{{key}}" not in result

def test_md_to_jira_unordered_list_dash():
    assert "* item" in md_to_jira("- item")

def test_md_to_jira_unordered_list_star():
    assert "* item" in md_to_jira("* item")

def test_md_to_jira_horizontal_rule():
    assert "----" in md_to_jira("---")
