import re
import json
from typing import Any
import structlog

logger = structlog.get_logger("deep-research")

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
