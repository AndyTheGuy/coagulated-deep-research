import structlog
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from core.models import Report, Claim
from core.nodes.scoping import router, get_router

logger = structlog.get_logger("deep-research")

class ExtractedClaims(BaseModel):
    """Pydantic model representing the list of claims extracted from the report."""
    claims: List[Claim] = Field(description="List of extracted factual claims")

class ClaimExtractor:
    """LLM-powered module that extracts factual claims from a research report."""

    def __init__(self) -> None:
        self.parser = JsonOutputParser(pydantic_object=ExtractedClaims)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an elite research verifier and fact-checker.\n"
                "Your task is to analyze the provided markdown report and extract all individual, concrete, and verifiable factual claims.\n\n"
                "For each claim, populate:\n"
                "- claim_id: A unique short identifier (e.g. 'c1', 'c2', 'c3')\n"
                "- claim_text: The precise factual statement of the claim (e.g. 'Company X grew by 15% in Q3')\n"
                "- section: The name of the section or header where this claim is found (e.g. 'Executive Summary')\n"
                "- supporting_quotes: A list of literal, verbatim quotes from the report section that state or support this claim\n"
                "- source_urls: A list of URLs that are cited or referenced in proximity to this claim in the report\n\n"
                "Format the output strictly as a JSON object matching this schema:\n"
                "{format_instructions}"
            )),
            ("user", "Extract all verifiable factual claims from the following report:\n\n{report_content}")
        ])

    async def extract_claims(self, report: Report) -> List[Claim]:
        """Extract factual claims from the report content."""
        logger.info("Extracting claims from report", report_title=report.title)
        
        formatted_prompt = self.prompt.format_prompt(
            report_content=report.content,
            format_instructions=self.parser.get_format_instructions()
        )
        
        try:
            response = await router.ainvoke(
                messages=formatted_prompt.to_messages(),
                tier="STANDARD",
                agent_name="VerifierAgent",
                node_name="claim_extraction"
            )
            
            parsed = self.parser.parse(response.content)
            claims_data = parsed.get("claims", [])
            
            claims: List[Claim] = []
            for item in claims_data:
                # Ensure validation and default values are loaded correctly
                claim = Claim(
                    claim_id=item.get("claim_id", f"c{len(claims) + 1}"),
                    claim_text=item.get("claim_text", ""),
                    section=item.get("section", "General"),
                    supporting_quotes=item.get("supporting_quotes", []),
                    source_urls=item.get("source_urls", []),
                    verification_status=item.get("verification_status", "unverified"),
                    confidence_score=item.get("confidence_score", 0.0),
                    remediation_notes=item.get("remediation_notes")
                )
                claims.append(claim)
                
            logger.info("Successfully extracted claims", count=len(claims))
            return claims
            
        except Exception as e:
            logger.error("Failed to extract claims from report, returning empty list", error=str(e))
            return []
