import os
import json
import pytest
from ui.log_streamer import parse_log_line, format_log_line_html, read_logs_from_file, AGENT_COLORS, LEVEL_COLORS

def test_parse_log_line_valid_json():
    """Verify parse_log_line successfully parses standard structured fields."""
    line = '{"timestamp": "2026-06-21T12:00:00Z", "level": "WARN", "event": "Over-temperature warning", "agent": "ResearcherAgent", "node": "web_scraps"}'
    parsed = parse_log_line(line)
    
    assert parsed is not None
    assert parsed["timestamp"] == "2026-06-21T12:00:00Z"
    assert parsed["level"] == "warn"  # Normalized to lowercase
    assert parsed["event"] == "Over-temperature warning"
    assert parsed["agent"] == "ResearcherAgent"
    assert parsed["node"] == "web_scraps"
    assert parsed["raw"] == {
        "timestamp": "2026-06-21T12:00:00Z",
        "level": "WARN",
        "event": "Over-temperature warning",
        "agent": "ResearcherAgent",
        "node": "web_scraps"
    }

def test_parse_log_line_invalid_or_malformed():
    """Verify parse_log_line handles malformed JSON, empty lines, and non-dicts gracefully."""
    # Malformed JSON
    assert parse_log_line("this is not json") is None
    
    # Empty/whitespace lines
    assert parse_log_line("") is None
    assert parse_log_line("   \n") is None
    
    # JSON list (not dict)
    assert parse_log_line('[1, 2, 3]') is None

def test_parse_log_line_agent_heuristics():
    """Verify parse_log_line heuristic fallback behavior when 'agent' or 'node' is omitted."""
    # 1. ScopingAgent heuristics
    l1 = '{"timestamp": "t1", "level": "info", "event": "Running scoping checks"}'
    p1 = parse_log_line(l1)
    assert p1["agent"] == "ScopingAgent"
    assert p1["node"] == "-"

    # 2. SupervisorAgent heuristics
    l2 = '{"timestamp": "t1", "level": "info", "event": "Supervisor task assignment"}'
    p2 = parse_log_line(l2)
    assert p2["agent"] == "SupervisorAgent"

    # 3. ResearcherAgent heuristics
    l3 = '{"timestamp": "t1", "level": "info", "event": "Initiated search query"}'
    p3 = parse_log_line(l3)
    assert p3["agent"] == "ResearcherAgent"

    # 4. VerifierAgent heuristics
    l4 = '{"timestamp": "t1", "level": "info", "event": "Starting quote validation"}'
    p4 = parse_log_line(l4)
    assert p4["agent"] == "VerifierAgent"

    # 5. WriterAgent heuristics
    l5 = '{"timestamp": "t1", "level": "info", "event": "Drafting final report"}'
    p5 = parse_log_line(l5)
    assert p5["agent"] == "WriterAgent"

    # 6. EvaluatorAgent heuristics
    l6 = '{"timestamp": "t1", "level": "info", "event": "Running dream evaluation quality gate"}'
    p6 = parse_log_line(l6)
    assert p6["agent"] == "EvaluatorAgent"

    # 7. MCTSPlanner heuristics
    l7 = '{"timestamp": "t1", "level": "info", "event": "Running mcts expansion step"}'
    p7 = parse_log_line(l7)
    assert p7["agent"] == "MCTSPlanner"

    # 8. Unknown keywords -> System
    l8 = '{"timestamp": "t1", "level": "info", "event": "Initializing application contexts"}'
    p8 = parse_log_line(l8)
    assert p8["agent"] == "System"

def test_format_log_line_html():
    """Verify format_log_line_html generates high-quality styled HTML with correct colors."""
    parsed = {
        "timestamp": "12:05:00",
        "level": "error",
        "event": "Database connection lost",
        "agent": "ResearcherAgent",
        "node": "scraping_node"
    }
    html = format_log_line_html(parsed)
    
    # HTML must include the expected tags and colors
    assert "style=\"color: #8b949e;" in html  # Timestamp color
    assert f"style=\"color: {AGENT_COLORS['ResearcherAgent']}; font-weight: bold;" in html  # ResearcherAgent color
    assert "style=\"color: #58a6ff;" in html  # Node color
    assert f"style=\"color: {LEVEL_COLORS['error']};" in html  # Level color
    
    assert "[12:05:00]" in html
    assert "[ResearcherAgent]" in html
    assert "[scraping_node]" in html
    assert "Database connection lost" in html

def test_read_logs_from_file(tmp_path):
    """Verify read_logs_from_file reads, parses, and formats lines from a physical file."""
    # 1. Non-existent file should return empty list
    non_existent = str(tmp_path / "missing.log")
    assert read_logs_from_file(non_existent) == []

    # 2. Setup mock log file
    log_file = tmp_path / "test_run.log"
    lines_to_write = [
        '{"timestamp": "12:00", "level": "info", "event": "Task started", "agent": "ScopingAgent", "node": "scoping_node"}\n',
        'invalid_non_json_line_should_be_skipped\n',
        '{"timestamp": "12:01", "level": "error", "event": "Failed to compile", "agent": "WriterAgent", "node": "writer_node"}\n'
    ]
    
    with open(log_file, "w", encoding="utf-8") as f:
        f.writelines(lines_to_write)
        
    formatted = read_logs_from_file(str(log_file))
    
    # Should skip the malformed line and return 2 parsed HTML strings
    assert len(formatted) == 2
    
    assert "Task started" in formatted[0]
    assert "ScopingAgent" in formatted[0]
    
    assert "Failed to compile" in formatted[1]
    assert "WriterAgent" in formatted[1]
