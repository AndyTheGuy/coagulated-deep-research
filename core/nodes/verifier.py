from typing import Any, Dict, List, Optional
import structlog
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from core.models import GraphState, Report, SubQuestion
from core.nodes.scoping import router, get_router
from verification.pipeline import VerificationPipeline

logger = structlog.get_logger("deep-research")

class VerifierCritique(BaseModel):
    """Pydantic model representing the output of the adversarial verifier's review."""
    gaps_found: bool = Field(
        description="True if critical gaps, unverified claims, or logical holes are found and need more research; False otherwise"
    )
    critique_text: str = Field(
        description="A detailed critical evaluation of the draft report, pointing out specific unverified claims or information gaps"
    )
    suggested_queries: List[str] = Field(
        default_factory=list,
        description="List of 1 to 3 specific research questions or search queries to run to fill the identified gaps"
    )

async def verifier_node(state: GraphState) -> Dict[str, Any]:
    """Adversarial Verifier node. Runs the verification pipeline on the draft report,
    and then invokes a CRITICAL-tier LLM to critique the results and decide on remediation.
    """
    logger.info("Running verifier_node")
    
    # 1. Check if draft report exists
    if not state.draft_report:
        logger.warning("No draft report found to verify, bypassing verification")
        return {
            "logs": ["Verifier node: No draft report available. Bypassed verification."]
        }
        
    # 2. Run the verification pipeline (which updates claims and score in place)
    pipeline = VerificationPipeline()
    verified_report, stats = await pipeline.run_verification_pipeline(state.draft_report)
    
    # Check if we have already looped too many times to prevent infinite loops
    # Let's count existing gap sub-questions
    gap_count = sum(1 for q in state.sub_questions_state if q.id.startswith("gap_"))
    max_gaps_allowed = 4 # Safety cap to prevent infinite loop
    
    if gap_count >= max_gaps_allowed:
        logger.info("Verifier node: Reached maximum gap-filling loops, forcing transition to report writer")
        return {
            "draft_report": verified_report,
            "logs": [f"Verifier node: Verification complete with score {verified_report.confidence_score}. Maximum loops reached."]
        }

    # 3. Compile prompt for LLM critique
    parser = JsonOutputParser(pydantic_object=VerifierCritique)
    
    # Format claims details for the LLM
    claims_text_list = []
    for claim in verified_report.claims:
        claims_text_list.append(
            f"- Claim [{claim.claim_id}]: \"{claim.claim_text}\"\n"
            f"  Status: {claim.verification_status}\n"
            f"  Confidence: {claim.confidence_score}\n"
            f"  Notes: {claim.remediation_notes}"
        )
    claims_summary = "\n".join(claims_text_list) if claims_text_list else "No claims extracted."
    
    system_prompt = (
        "You are an elite academic editor and adversarial verifier. Your job is to critically review "
        "the draft report and its claim verification results to identify any remaining informational gaps, "
        "logical inconsistencies, or unverified claims.\n\n"
        "If the report has multiple 'failed' or 'gap' claims, or is missing critical context on the overall "
        "topic, you MUST mark gaps_found as True and provide 1 to 3 targeted follow-up research questions "
        "to fill those gaps.\n\n"
        "If the overall confidence score is high (>= 0.80) and there are no critical missing parts of the "
        "user's query, mark gaps_found as False.\n\n"
        "Format the output strictly as a JSON object matching the following schema:\n"
        "{format_instructions}"
    )
    
    user_prompt = (
        "User's Overall Research Topic: {topic}\n\n"
        "Draft Report Title: {title}\n"
        "Draft Report Snippet (First 2000 chars):\n{content_snippet}\n\n"
        "Claim Verification Results:\n"
        "Overall Score: {overall_score}\n"
        "Verified Claims: {verified_count}\n"
        "Failed Claims: {failed_count}\n"
        "Gap Claims: {gap_count}\n"
        "Claims Details:\n{claims_summary}\n\n"
        "Please analyze this report and output your critique."
    )
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt)
    ])
    
    content_snippet = verified_report.content[:2000]
    formatted_prompt = prompt_template.format_prompt(
        topic=state.topic or (state.research_brief.topic if state.research_brief else "Research"),
        title=verified_report.title,
        content_snippet=content_snippet,
        overall_score=stats.overall_score,
        verified_count=stats.verified_claims_count,
        failed_count=stats.failed_claims_count,
        gap_count=stats.gaps_count,
        claims_summary=claims_summary,
        format_instructions=parser.get_format_instructions()
    )
    
    response = await router.ainvoke(
        messages=formatted_prompt.to_messages(),
        tier="CRITICAL",
        agent_name="VerifierAgent",
        node_name="verifier_critique"
    )
    
    try:
        parsed = parser.parse(response.content)
        critique = VerifierCritique(**parsed)
        logger.info("Adversarial verifier analysis complete", gaps_found=critique.gaps_found)
        
        new_sub_questions = []
        logs = [f"Verifier node: Verified report confidence score {verified_report.confidence_score}."]
        
        if critique.gaps_found and critique.suggested_queries:
            logger.info("Verifier node: Gaps detected. Preparing follow-up questions", queries=critique.suggested_queries)
            logs.append(f"Gaps found in verifier critique: {critique.critique_text}")
            
            # Generate pending sub-questions for the gaps
            for i, query in enumerate(critique.suggested_queries):
                q_id = f"gap_{gap_count + i + 1}"
                new_sub_questions.append(
                    SubQuestion(
                        id=q_id,
                        question=query,
                        status="pending"
                    )
                )
                logs.append(f"Added follow-up sub-question {q_id}: \"{query}\"")
                
        else:
            logger.info("Verifier node: No gaps found or no queries suggested.")
            logs.append("Adversarial verifier approved the report. No gaps found.")
            
        return {
            "draft_report": verified_report,
            "sub_questions_state": new_sub_questions, # This will be appended due to list reducer
            "logs": logs,
            "token_usage": get_router().token_usage
        }
        
    except Exception as e:
        logger.error("Failed to parse verifier critique, defaulting to proceeding with no gaps", error=str(e))
        return {
            "draft_report": verified_report,
            "logs": [f"Verifier node: Failed to parse critique ({str(e)}), proceeding without gaps."],
            "token_usage": get_router().token_usage
        }
