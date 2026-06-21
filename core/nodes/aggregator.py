import structlog
from typing import Any, Dict, List
from core.models import GraphState, Report, VerifiedSource

logger = structlog.get_logger("deep-research")

async def context_aggregator_node(state: GraphState) -> Dict[str, Any]:
    """Context Aggregator node. Merges synthesized findings from all completed parallel researcher agents,
    filters duplicates/redundancies, and compiles them into a structured draft Report with properly
    formatted metadata and bibliography.
    """
    topic = state.topic or (state.research_brief.topic if state.research_brief else "Research Topic")
    logger.info("Running context_aggregator_node", topic=topic)
    
    # Filter completed sub-questions
    completed_qs = [
        q for q in state.sub_questions_state 
        if q.status == "completed" and q.results_summary and q.results_summary.strip()
    ]
    
    if not completed_qs:
        logger.warning("No completed sub-question findings found to aggregate")
        fallback_report = Report(
            title=f"Preliminary Aggregated Findings: {topic}",
            content="No completed researcher agent findings were available to aggregate.",
            citations=[]
        )
        return {
            "draft_report": fallback_report,
            "logs": ["Context aggregator completed with no completed sub-questions found."]
        }
        
    # Merge findings into structured sections
    sections = []
    for q in completed_qs:
        sections.append(
            f"## {q.id.upper()}: {q.question}\n\n"
            f"{q.results_summary.strip()}"
        )
        
    merged_sections_text = "\n\n---\n\n".join(sections)
    
    # Compile bibliography from verified sources
    seen_urls = set()
    unique_sources: List[VerifiedSource] = []
    for src in state.verified_sources:
        normalized_url = src.url.strip().lower().rstrip("/")
        if normalized_url not in seen_urls:
            seen_urls.add(normalized_url)
            unique_sources.append(src)
            
    bibliography_lines = ["## Sources & References\n"]
    citations_list = []
    
    accessible_sources = [s for s in unique_sources if s.accessible]
    
    for src in accessible_sources:
        ref_text = f"- [{src.title}]({src.url})"
        bibliography_lines.append(ref_text)
        citations_list.append(src.url)
        
    if not accessible_sources:
        bibliography_lines.append("No successful external source citations were compiled.")
        
    bibliography_text = "\n".join(bibliography_lines)
    
    # Build full aggregated content
    aggregated_content = (
        f"# Aggregated Research Findings: {topic}\n\n"
        f"This document consolidates and organizes the research findings from parallel academic and technical searches "
        f"for the topic: **{topic}**.\n\n"
        f"{merged_sections_text}\n\n"
        f"{bibliography_text}"
    )
    
    report = Report(
        title=f"Aggregated Findings: {topic}",
        content=aggregated_content,
        citations=citations_list
    )
    
    logger.info("Context aggregation complete", completed_count=len(completed_qs), sources_count=len(citations_list))
    
    return {
        "draft_report": report,
        "logs": [f"Context aggregator successfully compiled {len(completed_qs)} sub-question findings with {len(citations_list)} citations."]
    }
