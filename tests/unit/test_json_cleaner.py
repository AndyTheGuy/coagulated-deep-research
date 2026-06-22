import pytest
from core.utils.json_cleaner import clean_json_string, parse_json_safely

def test_clean_json_string_basic():
    assert clean_json_string('{"key": "value"}') == '{"key": "value"}'
    assert clean_json_string('   {"key": "value"}   ') == '{"key": "value"}'

def test_clean_json_string_markdown_blocks():
    assert clean_json_string('```json\n{"key": "value"}\n```') == '{"key": "value"}'
    assert clean_json_string('```\n{"key": "value"}\n```') == '{"key": "value"}'

def test_clean_json_string_boundaries():
    assert clean_json_string('Some text prefix {"key": "value"} some text suffix') == '{"key": "value"}'
    assert clean_json_string('Prefix [1, 2, 3] Suffix') == '[1, 2, 3]'
    # Outer bracket wrapping braces
    assert clean_json_string('Prefix [ {"key": "value"} ] suffix') == '[ {"key": "value"} ]'

def test_clean_json_string_trailing_commas():
    assert clean_json_string('{"key": "value",}') == '{"key": "value"}'
    assert clean_json_string('[1, 2, 3,]') == '[1, 2, 3]'
    assert clean_json_string('{"list": [1, 2, {"a": "b",},],}') == '{"list": [1, 2, {"a": "b"}]}'

def test_parse_json_safely_success():
    data = parse_json_safely('Some LLM response ```json\n{"val": 123,}\n```')
    assert data == {"val": 123}

def test_parse_json_safely_failure():
    with pytest.raises(Exception):
        parse_json_safely('{"invalid_json": ')
