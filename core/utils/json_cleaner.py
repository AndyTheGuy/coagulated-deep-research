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

def find_outermost_balanced_block(s: str, start_char: str = '{', end_char: str = '}') -> str:
    """Finds the largest substring starting with start_char that forms a balanced structure."""
    first_idx = s.find(start_char)
    if first_idx == -1:
        return ""
        
    # Scan from first occurrence and count balancing depth
    n = len(s)
    for start in range(first_idx, n):
        if s[start] != start_char:
            continue
            
        depth = 0
        in_string = False
        escaped = False
        
        for i in range(start, n):
            char = s[i]
            
            if char == '"' and not escaped:
                in_string = not in_string
                escaped = False
                continue
                
            if in_string:
                if char == '\\':
                    escaped = not escaped
                else:
                    escaped = False
                continue
                
            # Keep track of escape sequences outside strings (usually irrelevant but safe)
            escaped = False
            
            if char == start_char:
                depth += 1
            elif char == end_char:
                depth -= 1
                if depth == 0:
                    return s[start:i+1]
                    
    return ""

def clean_json_string(s: str) -> str:
    """Cleans a string containing JSON output from an LLM.
    Strips markdown code blocks, extracts the JSON object, and fixes common syntax issues.
    """
    if not s:
        return ""
    
    # 1. Strip general whitespace
    s = s.strip()
    
    # 2. Strip Markdown code blocks if they wrap the content
    pattern = r"```(?:json)?\s*(.*?)\s*```"
    match = re.search(pattern, s, re.DOTALL)
    if match:
        s = match.group(1).strip()
        
    # 3. Extract balanced JSON block
    balanced_brace = find_outermost_balanced_block(s, '{', '}')
    balanced_bracket = find_outermost_balanced_block(s, '[', ']')
    
    if balanced_brace:
        # Check if bracket wraps the brace
        if balanced_bracket and s.find('[') < s.find('{') and s.rfind(']') > s.rfind('}'):
            s = balanced_bracket
        else:
            s = balanced_brace
    elif balanced_bracket:
        s = balanced_bracket
        
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
