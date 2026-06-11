import pytest
from message import parse_message


def test_parse_basic_prefix():
    service, msg = parse_message("ibe: why is checkout failing?")
    assert service == "ibe"
    assert msg == "why is checkout failing?"


def test_parse_strips_whitespace():
    service, msg = parse_message("  ibe  :  why is checkout failing?  ")
    assert service == "ibe"
    assert msg == "why is checkout failing?"


def test_parse_message_containing_colon():
    service, msg = parse_message("ibe: error: something went wrong")
    assert service == "ibe"
    assert msg == "error: something went wrong"


def test_parse_missing_colon_raises():
    with pytest.raises(ValueError, match="missing_prefix"):
        parse_message("why is checkout failing?")


def test_parse_empty_service_raises():
    with pytest.raises(ValueError, match="missing_prefix"):
        parse_message(": why is checkout failing?")


def test_parse_empty_message_raises():
    with pytest.raises(ValueError, match="missing_prefix"):
        parse_message("ibe:")


def test_parse_empty_string_raises():
    with pytest.raises(ValueError, match="missing_prefix"):
        parse_message("")


def test_parse_uppercased_service():
    service, msg = parse_message("IBE: why is checkout failing?")
    assert service == "ibe"
    assert msg == "why is checkout failing?"
