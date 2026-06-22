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

def test_clean_json_string_control_characters():
    # 1. Literal newline inside string value
    raw_with_newline = '{"text": "Line 1\nLine 2"}'
    assert clean_json_string(raw_with_newline) == '{"text": "Line 1\\nLine 2"}'
    
    # 2. Literal tab inside string value
    raw_with_tab = '{"text": "Col 1\tCol 2"}'
    assert clean_json_string(raw_with_tab) == '{"text": "Col 1\\tCol 2"}'
    
    # 3. Preserves newlines outside string values
    formatted_json = '{\n  "key": "value"\n}'
    assert clean_json_string(formatted_json) == '{\n  "key": "value"\n}'
    
    # 4. Correctly handles escaped double quotes without toggling in_string incorrectly
    escaped_quotes = '{"text": "He said \\"hello\\" to me"}'
    assert clean_json_string(escaped_quotes) == '{"text": "He said \\"hello\\" to me"}'
    
    # 5. Correctly handles double backslashes
    double_backslash = '{"text": "Double \\\\ backslash"}'
    assert clean_json_string(double_backslash) == '{"text": "Double \\\\ backslash"}'

    # 6. End-to-end safe parsing
    parsed = parse_json_safely('{"report": "Line 1\nLine 2", "score": 1.0}')
    assert parsed == {"report": "Line 1\nLine 2", "score": 1.0}
