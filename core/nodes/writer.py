from typing import Any, Dict
import structlog
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from core.models import GraphState, Report
from core.nodes.scoping import router, get_router

logger = structlog.get_logger("deep-research")

class WriterOutput(BaseModel):
    """Pydantic model representing the output of the report writer."""
    title: str = Field(description="The professional academic title of the final report")
    content: str = Field(description="The full final research report formatted in clean Markdown")

async def report_writer_node(state: GraphState) -> Dict[str, Any]:
    """Report Writer node. Compiles verified findings, quotes, and citations 
    into a structured academic-grade final markdown report.
    """
    logger.info("Running report_writer_node")
    
    # 1. Fallback if no draft report exists
    draft = state.draft_report
    if not draft:
        logger.warning("No draft report found, compiling fallback content")
        draft = Report(
            title=f"Research Report: {state.topic or 'Topic'}",
            content="No draft report was generated."
        )
        
    # 2. Prepare verified claims metadata to inject
    claims_text_list = []
    for claim in draft.claims:
        claims_text_list.append(
            f"- Claim: \"{claim.claim_text}\"\n"
            f"  Supporting Quotes: {claim.supporting_quotes}\n"
            f"  Source URLs: {claim.source_urls}\n"
            f"  Status: {claim.verification_status} (Confidence: {claim.confidence_score})"
        )
    claims_details = "\n".join(claims_text_list) if claims_text_list else "No verified claims metadata."

    # 3. Compile prompt and call LLM
    parser = JsonOutputParser(pydantic_object=WriterOutput)
    
    system_prompt = (
        "You are an elite research analyst and academic writer. Your job is to compile a peer-review grade "
        "final research report in clean Markdown based on the provided draft report and its verified claim details.\n\n"
        "Your final report must adhere to these strict requirements:\n"
        "1. Academic Rigor: Write in an objective, professional, and dense analytical style.\n"
        "2. Comprehensive Content: Ensure all sub-questions are thoroughly addressed using verified facts.\n"
        "3. Literal Quotes & Citations: Integrate literal supporting quotes seamlessly in the text with inline citations "
        "linking to their source URLs, e.g. [Source Title](URL).\n"
        "4. Bibliography: Include a comprehensive, formatted 'Sources & References' section at the end of the report, "
        "listing all cited source URLs (aim for at least 20 sources if available in the context).\n"
        "5. Confidence Summary: Append a 'Report Confidence Assessment' section highlighting the overall report "
        "confidence score (which is {overall_score}) and summarizing the verification statistics (verified/failed/gaps).\n\n"
        "Format the output strictly as a JSON object matching the following schema:\n"
        "{format_instructions}"
    )
    
    user_prompt = (
        "Draft Report Title: {title}\n"
        "Draft Report Content:\n{draft_content}\n\n"
        "Verified Claims & Quotes Details:\n{claims_details}\n\n"
        "Please generate the final polished Markdown report."
    )
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt)
    ])
    
    formatted_prompt = prompt_template.format_prompt(
        overall_score=draft.confidence_score,
        title=draft.title,
        draft_content=draft.content,
        claims_details=claims_details,
        format_instructions=parser.get_format_instructions()
    )
    
    response = await router.ainvoke(
        messages=formatted_prompt.to_messages(),
        tier="CRITICAL",
        agent_name="WriterAgent",
        node_name="report_writer"
    )
    
    try:
        parsed = parser.parse(response.content)
        writer_out = WriterOutput(**parsed)
        logger.info("Report writing complete", title=writer_out.title)
        
        # Instantiate final report model, carrying over claims list and confidence score
        final_report = Report(
            title=writer_out.title,
            content=writer_out.content,
            claims=draft.claims,
            citations=draft.citations,
            confidence_score=draft.confidence_score
        )
        
        return {
            "final_report": final_report,
            "logs": [f"Report writer compiled final report: \"{final_report.title}\" with confidence score {final_report.confidence_score}."],
            "token_usage": get_router().token_usage
        }
        
    except Exception as e:
        logger.error("Failed to parse report writer output, defaulting to draft report as final", error=str(e))
        fallback_final = Report(
            title=draft.title,
            content=draft.content,
            claims=draft.claims,
            citations=draft.citations,
            confidence_score=draft.confidence_score
        )
        return {
            "final_report": fallback_final,
            "logs": [f"Report writer failed to parse final output ({str(e)}), fell back to verified draft."],
            "errors": [f"Writer output parsing failed: {str(e)}"],
            "token_usage": get_router().token_usage
        }
