import json
import os
from typing import Any, Dict, List, Optional

# Premium HSL/RGB Colors for Modern Dark Theme
AGENT_COLORS = {
    "ScopingAgent": "#ff66ff",     # Magenta
    "SupervisorAgent": "#b366ff",  # Purple
    "ResearcherAgent": "#33ffff",  # Cyan
    "VerifierAgent": "#ffff66",    # Yellow
    "WriterAgent": "#66ff66",      # Green
    "EvaluatorAgent": "#3399ff",   # Blue
    "System": "#e0e0e0",           # Light Grey
    "MCTSPlanner": "#ffaa66"       # Warm Orange for MCTS
}

LEVEL_COLORS = {
    "info": "#c9d1d9",      # Sleek off-white
    "warning": "#ff9933",   # Vibrant Orange
    "warn": "#ff9933",      # Vibrant Orange (compatibility)
    "error": "#ff3333"      # Vivid Red
}

def parse_log_line(line_str: str) -> Optional[Dict[str, Any]]:
    """Parse a single JSON log line and extract standard fields.
    Returns None if the line is not valid JSON.
    """
    if not line_str.strip():
        return None
    try:
        data = json.loads(line_str)
        if not isinstance(data, dict):
            return None
        
        # Extract and sanitize standard fields
        timestamp = data.get("timestamp", "")
        level = data.get("level", "info").lower()
        event = data.get("event", "")
        
        # Map structured fields
        agent = data.get("agent")
        node = data.get("node")
        
        # Heuristic fallbacks if agent/node fields are not explicitly present
        if not agent:
            if "scoping" in event.lower() or "clarify" in event.lower() or "brief" in event.lower():
                agent = "ScopingAgent"
            elif "supervisor" in event.lower() or "route" in event.lower() or "routing" in event.lower():
                agent = "SupervisorAgent"
            elif "researcher" in event.lower() or "search" in event.lower() or "scrape" in event.lower() or "diversify" in event.lower():
                agent = "ResearcherAgent"
            elif "verifier" in event.lower() or "critique" in event.lower() or "claim" in event.lower() or "quote" in event.lower():
                agent = "VerifierAgent"
            elif "writer" in event.lower() or "report" in event.lower():
                agent = "WriterAgent"
            elif "evaluator" in event.lower() or "dream" in event.lower() or "kic" in event.lower():
                agent = "EvaluatorAgent"
            elif "mcts" in event.lower() or "planner" in event.lower():
                agent = "MCTSPlanner"
            else:
                agent = "System"
                
        if not node:
            node = "-"
            
        return {
            "timestamp": timestamp,
            "level": level,
            "event": event,
            "agent": agent,
            "node": node,
            "raw": data
        }
    except Exception:
        return None

def format_log_line_html(parsed: Dict[str, Any]) -> str:
    """Format the parsed log dictionary into a beautiful premium HTML string."""
    timestamp = parsed["timestamp"]
    level = parsed["level"]
    event = parsed["event"]
    agent = parsed["agent"]
    node = parsed["node"]
    
    agent_color = AGENT_COLORS.get(agent, AGENT_COLORS["System"])
    level_color = LEVEL_COLORS.get(level, LEVEL_COLORS["info"])
    
    # Format: [timestamp] [agent_name] [node_name] action_description
    html = (
        f'<span style="color: #8b949e; font-family: monospace;">[{timestamp}]</span> '
        f'<span style="color: {agent_color}; font-weight: bold; font-family: sans-serif;">[{agent}]</span> '
        f'<span style="color: #58a6ff; font-family: monospace;">[{node}]</span> '
        f'<span style="color: {level_color}; font-family: monospace;">{event}</span>'
    )
    return html

def read_logs_from_file(file_path: str = "agent.json.log") -> List[str]:
    """Read logs from the JSON file and return them formatted as HTML lines."""
    if not os.path.exists(file_path):
        return []
    
    formatted_lines = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parsed = parse_log_line(line)
                if parsed:
                    formatted_lines.append(format_log_line_html(parsed))
    except Exception as e:
        formatted_lines.append(f'<span style="color: #ff3333;">[Error reading log file: {str(e)}]</span>')
        
    return formatted_lines
