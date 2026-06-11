import pytest
from jira import parse_rca_input, extract_adf_text, build_rca_message, md_to_jira


# --- parse_rca_input ---

def test_parse_rca_bare_key():
    service, key, ctx = parse_rca_input("ibe: IBE-1152")
    assert service == "ibe"
    assert key == "IBE-1152"
    assert ctx == ""

def test_parse_rca_full_url():
    service, key, ctx = parse_rca_input(
        "ibe: https://stayntouch.atlassian.net/browse/IBE-1152"
    )
    assert service == "ibe"
    assert key == "IBE-1152"
    assert ctx == ""

def test_parse_rca_with_additional_context():
    service, key, ctx = parse_rca_input(
        "ibe: IBE-1152\nStack trace: TypeError at line 45"
    )
    assert service == "ibe"
    assert key == "IBE-1152"
    assert ctx == "Stack trace: TypeError at line 45"

def test_parse_rca_uppercase_service_normalised():
    service, key, _ = parse_rca_input("IBE: IBE-1152")
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
    assert "Error" in extract_adf_text(node)
    assert "occurs here" in extract_adf_text(node)

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

def test_md_to_jira_unordered_list_dash():
    assert "* item" in md_to_jira("- item")

def test_md_to_jira_unordered_list_star():
    assert "* item" in md_to_jira("* item")

def test_md_to_jira_horizontal_rule():
    assert "----" in md_to_jira("---")
