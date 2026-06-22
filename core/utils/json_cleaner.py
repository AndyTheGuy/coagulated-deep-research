import re
import json
from typing import Any
import structlog

logger = structlog.get_logger("deep-research")

def escape_control_chars_in_json_strings(s: str) -> str:
    """Escapes literal control characters (like actual unescaped newlines, tabs, and carriage returns)
    found inside double-quoted JSON string values, while preserving valid JSON syntax outside string values.
    """
    chars = []
    in_string = False
    escaped = False
    i = 0
    n = len(s)
    while i < n:
        char = s[i]
        if char == '"' and not escaped:
            in_string = not in_string
            chars.append(char)
        elif in_string:
            if escaped:
                chars.append(char)
                escaped = False
            elif char == '\\':
                escaped = True
                chars.append(char)
            else:
                if char == '\n':
                    chars.append('\\n')
                elif char == '\t':
                    chars.append('\\t')
                elif char == '\r':
                    chars.append('\\r')
                else:
                    chars.append(char)
        else:
            chars.append(char)
        i += 1
    return "".join(chars)

def clean_json_string(s: str) -> str:
    """Cleans a string containing JSON output from an LLM.
    Strips markdown code blocks, extracts the JSON object, and fixes common syntax issues.
    """
    if not s:
        return ""
    
    # 1. Strip general whitespace
    s = s.strip()
    
    # 2. Strip Markdown code blocks if they wrap the content
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\n", "", s)
        s = re.sub(r"\n```$", "", s)
        s = s.strip()
        
    # 3. Find the boundaries of the JSON block (either { } or [ ])
    first_brace = s.find('{')
    last_brace = s.rfind('}')
    
    first_bracket = s.find('[')
    last_bracket = s.rfind(']')
    
    if first_brace != -1 and last_brace != -1:
        # Check if there is an outer bracket wrapping the braces
        if first_bracket != -1 and first_bracket < first_brace and last_bracket != -1 and last_bracket > last_brace:
            s = s[first_bracket:last_bracket+1]
        else:
            s = s[first_brace:last_brace+1]
    elif first_bracket != -1 and last_bracket != -1:
        s = s[first_bracket:last_bracket+1]
        
    # Escape literal control characters in JSON strings before comma cleanup
    s = escape_control_chars_in_json_strings(s)
    
    # 4. Repair trailing commas in objects and arrays: e.g. {"a": 1,} -> {"a": 1}
    s = re.sub(r',\s*([\]}])', r'\1', s)
    
    return s

def parse_json_safely(s: str) -> Any:
    """Attempts to parse a JSON string, applying robust cleaning first."""
    cleaned = clean_json_string(s)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error("JSON parsing failed after cleaning", original=s, cleaned=cleaned, error=str(e))
        raise
