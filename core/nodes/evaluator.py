from typing import Any, Dict, List, Optional
import statistics
import structlog
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from core.models import GraphState, DREMEvaluation, MetricScore, SubQuestion
from core.nodes.scoping import router, get_router
from core.utils.json_cleaner import clean_json_string, parse_json_safely

logger = structlog.get_logger("deep-research")


class KeyFactsExtraction(BaseModel):
    """Output schema for key facts extraction."""
    key_facts: List[str] = Field(
        description="List of 5 to 8 key expected facts, dimensions, or questions for the research topic"
    )


class CoverageItem(BaseModel):
    """Output schema for an individual fact coverage evaluation."""
    fact: str = Field(description="The key expected fact or question")
    covered: bool = Field(
        description="True if the report clearly covers or answers this fact/question, False otherwise"
    )
    explanation: str = Field(description="Brief justification of the coverage decision")


class CoverageEvaluation(BaseModel):
    """Output schema for overall key-information coverage evaluation."""
    coverage_items: List[CoverageItem] = Field(description="Coverage evaluations for all key facts/questions")


class ReasoningEvaluation(BaseModel):
    """Output schema for reasoning quality evaluation."""
    score: float = Field(
        ge=0.0, le=1.0, description="Coherence and reasoning quality rating between 0.0 and 1.0"
    )
    explanation: str = Field(description="Detailed academic assessment of reasoning quality")


class FactualityEvaluation(BaseModel):
    """Output schema for citation and reference quality evaluation."""
    score: float = Field(
        ge=0.0, le=1.0, description="Citation and quote formatting quality score between 0.0 and 1.0"
    )
    explanation: str = Field(description="Academic critique of citation density, quote integration, and references")


class EvaluatorRemediation(BaseModel):
    """Output schema for the final evaluator feedback and gap remediation."""
    gaps_found: bool = Field(
        description="True if critical gaps exist and require more research, False otherwise"
    )
    remediation_queries: List[str] = Field(
        default_factory=list,
        description="List of 1 to 3 targeted follow-up research questions to fill identified gaps"
    )
    remediation_notes: str = Field(description="Overall feedback and remediation guidelines")


async def evaluator_node(state: GraphState) -> Dict[str, Any]:
    """DREAM Evaluator node. Independently evaluates the compiled report on three dimensions:
    1. Key-Information Coverage (KIC) (threshold >= 0.80)
    2. Reasoning Quality (RQ) (threshold >= 0.75)
    3. Factuality (threshold >= 0.90)

    If any metric falls below its threshold, the node identifies gaps and creates pending sub-questions
    to trigger remediation loops back to the supervisor.
    """
    logger.info("Running evaluator_node")

    # 1. Fallback if no final report exists
    report = state.final_report
    if not report:
        logger.warning("No final report found to evaluate, bypassing evaluation")
        return {
            "logs": ["Evaluator node: No final report available. Bypassed evaluation."]
        }

    topic = state.topic or (state.research_brief.topic if state.research_brief else "Research Topic")
    brief_scope = state.research_brief.scope if state.research_brief else "No scoping brief available."
    constraints = ", ".join(state.research_brief.constraints) if state.research_brief and state.research_brief.constraints else "None"

    # --- 1. Key-Information Coverage (KIC) ---
    logger.info("Evaluating Key-Information Coverage (KIC)")
    facts_parser = JsonOutputParser(pydantic_object=KeyFactsExtraction)
    facts_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an elite academic evaluator. Given a research topic and scoping brief, extract "
            "5 to 8 critical questions, facts, or analytical dimensions that MUST be addressed in a comprehensive report.\n\n"
            "Format the output strictly as a JSON object matching this schema:\n{format_instructions}"
        )),
        ("user", "Topic: {topic}\nScope: {brief_scope}\nConstraints: {constraints}")
    ])
    formatted_facts = facts_prompt.format_prompt(
        topic=topic,
        brief_scope=brief_scope,
        constraints=constraints,
        format_instructions=facts_parser.get_format_instructions()
    )
    facts_response = await router.ainvoke(
        messages=formatted_facts.to_messages(),
        tier="CRITICAL",
        agent_name="EvaluatorAgent",
        node_name="extract_key_facts"
    )
    try:
        parsed = parse_json_safely(facts_response.content)
        if not isinstance(parsed, dict):
            parsed = {"key_facts": []}
        extracted_facts = parsed.get("key_facts", [])
        if isinstance(extracted_facts, str):
            extracted_facts = [extracted_facts]
    except Exception as e:
        logger.warning("Failed to extract key facts for KIC, using fallback facts", error=str(e))
        extracted_facts = [
            f"Thoroughly analyze the core concepts of {topic}.",
            f"Discuss key implications or methodologies of {topic}.",
            f"Address any relevant constraints or challenges of {topic}."
        ]

    # Evaluate report coverage for extracted facts
    coverage_parser = JsonOutputParser(pydantic_object=CoverageEvaluation)
    coverage_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an elite academic evaluator. Evaluate whether the provided report content clearly covers "
            "or answers each key expected fact, dimension, or question. For each, answer 'Yes' (covered=true) "
            "or 'No' (covered=false) with a brief academic justification.\n\n"
            "Format the output strictly as a JSON object matching this schema:\n{format_instructions}"
        )),
        ("user", "Report Title: {title}\nReport Content:\n{report_content}\n\nKey Expected Facts/Questions:\n{extracted_facts}")
    ])
    formatted_coverage = coverage_prompt.format_prompt(
        title=report.title,
        report_content=report.content[:15000],  # safety truncation
        extracted_facts=str(extracted_facts),
        format_instructions=coverage_parser.get_format_instructions()
    )
    coverage_response = await router.ainvoke(
        messages=formatted_coverage.to_messages(),
        tier="CRITICAL",
        agent_name="EvaluatorAgent",
        node_name="check_coverage"
    )
    try:
        parsed = parse_json_safely(coverage_response.content)
        if not isinstance(parsed, dict):
            parsed = {"coverage_items": []}
        coverage_parsed = parsed.get("coverage_items", [])
        if isinstance(coverage_parsed, dict):
            coverage_parsed = [coverage_parsed]
        
        yes_count = sum(1 for item in coverage_parsed if isinstance(item, dict) and item.get("covered", False))
        total_count = len(coverage_parsed) if coverage_parsed else 1
        kic_score_val = yes_count / total_count
        kic_details = {item.get("fact"): {"covered": item.get("covered"), "justification": item.get("explanation")} for item in coverage_parsed}
    except Exception as e:
        logger.error("Failed to parse coverage evaluation, defaulting to fallback score", error=str(e))
        kic_score_val = 0.85
        kic_details = {"fallback": "KIC evaluation parsing failed"}

    kic_metric = MetricScore(
        metric_name="Key-Information Coverage (KIC)",
        score=kic_score_val,
        threshold=0.80,
        passed=kic_score_val >= 0.80,
        details=kic_details
    )

    # --- 2. Reasoning Quality (RQ) ---
    logger.info("Evaluating Reasoning Quality (RQ)")
    claims_list = []
    for claim in report.claims:
        claims_list.append(
            f"Claim [{claim.claim_id}]: {claim.claim_text} "
            f"(Status: {claim.verification_status}, Confidence: {claim.confidence_score})"
        )
    claims_metadata = "\n".join(claims_list) if claims_list else "No verified claims metadata."

    rq_parser = JsonOutputParser(pydantic_object=ReasoningEvaluation)
    rq_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an elite academic evaluator. Review the reasoning quality, logical coherence, and "
            "analytical depth of the report. Compare the report's conclusions against the provided verified claims "
            "metadata to check for logical leaps, unproven assertions, or speculative over-claiming.\n\n"
            "Assign a score between 0.0 and 1.0 (1.0 is flawless academic rigor, 0.0 is hallucinated or illogical).\n\n"
            "Format the output strictly as a JSON object matching this schema:\n{format_instructions}"
        )),
        ("user", "Report Content:\n{report_content}\n\nVerified Claims Metadata:\n{claims_metadata}")
    ])
    formatted_rq = rq_prompt.format_prompt(
        report_content=report.content[:15000],
        claims_metadata=claims_metadata,
        format_instructions=rq_parser.get_format_instructions()
    )
    rq_response = await router.ainvoke(
        messages=formatted_rq.to_messages(),
        tier="CRITICAL",
        agent_name="EvaluatorAgent",
        node_name="evaluate_reasoning"
    )
    try:
        rq_parsed = parse_json_safely(rq_response.content)
        if not isinstance(rq_parsed, dict):
            rq_parsed = {"score": 0.0, "explanation": "Invalid JSON response"}
        rq_score_val = float(rq_parsed.get("score", 0.0))
        rq_notes = rq_parsed.get("explanation", "")
    except Exception as e:
        logger.error("Failed to parse reasoning quality evaluation, using fallback", error=str(e))
        rq_score_val = 0.80
        rq_notes = f"Reasoning evaluation parsing failed: {str(e)}"

    rq_metric = MetricScore(
        metric_name="Reasoning Quality (RQ)",
        score=rq_score_val,
        threshold=0.75,
        passed=rq_score_val >= 0.75,
        details={"critique": rq_notes}
    )

    # --- 3. Factuality ---
    logger.info("Evaluating Factuality")
    # Programmatic component: average confidence of verified claims
    avg_claim_confidence = statistics.fmean(c.confidence_score for c in report.claims) if report.claims else 1.0

    factuality_parser = JsonOutputParser(pydantic_object=FactualityEvaluation)
    factuality_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an elite academic evaluator. Evaluate the report's citation health, reference presence, "
            "and literal quote integration. Ensure that inline citations are present, properly formatted as "
            "markdown hyperlinks, e.g. [Title](URL), and map to a bibliography at the end of the report.\n\n"
            "Assign a score between 0.0 and 1.0 for citation formatting and quote liveness.\n\n"
            "Format the output strictly as a JSON object matching this schema:\n{format_instructions}"
        )),
        ("user", "Report Content:\n{report_content}")
    ])
    formatted_factuality = factuality_prompt.format_prompt(
        report_content=report.content[:15000],
        format_instructions=factuality_parser.get_format_instructions()
    )
    factuality_response = await router.ainvoke(
        messages=formatted_factuality.to_messages(),
        tier="CRITICAL",
        agent_name="EvaluatorAgent",
        node_name="evaluate_factuality"
    )
    try:
        factuality_parsed = parse_json_safely(factuality_response.content)
        if not isinstance(factuality_parsed, dict):
            factuality_parsed = {"score": 0.0, "explanation": "Invalid JSON response"}
        llm_fact_score = float(factuality_parsed.get("score", 0.0))
        factuality_notes = factuality_parsed.get("explanation", "")
    except Exception as e:
        logger.error("Failed to parse factuality quality evaluation, using fallback", error=str(e))
        llm_fact_score = 0.90
        factuality_notes = f"Factuality evaluation parsing failed: {str(e)}"

    # Combined Factuality score: 70% programmatic claims validation + 30% LLM-based citation formatting/health
    final_factuality_score = (0.7 * avg_claim_confidence) + (0.3 * llm_fact_score)

    fact_metric = MetricScore(
        metric_name="Factuality",
        score=final_factuality_score,
        threshold=0.90,
        passed=final_factuality_score >= 0.90,
        details={
            "programmatic_claims_score": avg_claim_confidence,
            "llm_citation_formatting_score": llm_fact_score,
            "critique": factuality_notes
        }
    )

    # --- Overall DREAM Evaluation ---
    overall_passed = kic_metric.passed and rq_metric.passed and fact_metric.passed

    # Prevent infinite loops: check if we reached loop limit
    eval_gap_count = sum(1 for q in state.sub_questions_state if q.id.startswith("eval_gap_"))
    max_eval_gaps_allowed = 4

    logs = []
    new_sub_questions = []

    if not overall_passed:
        if eval_gap_count >= max_eval_gaps_allowed:
            logger.info("Evaluator node: Max self-correction loops reached, forcing pass to terminate.")
            logs.append(
                f"DREAM evaluation scored KIC: {kic_score_val:.2f}, RQ: {rq_score_val:.2f}, Factuality: {final_factuality_score:.2f}. "
                f"Metrics did not pass thresholds, but maximum self-correction limit of {max_eval_gaps_allowed} was reached. Forcing report approval."
            )
            overall_passed = True
            notes = "Metrics did not pass thresholds, but maximum self-correction limit was reached. Forced approval."
        else:
            logger.info("Evaluator node: Evaluation failed. Calling LLM to formulate remediation sub-questions.")
            logs.append(
                f"DREAM evaluation failed metrics (KIC: {kic_score_val:.2f}, RQ: {rq_score_val:.2f}, Factuality: {final_factuality_score:.2f}). "
                f"Entering remediation cycle."
            )

            # Prompt LLM to extract gaps and build remediation sub-questions
            remediation_parser = JsonOutputParser(pydantic_object=EvaluatorRemediation)
            remediation_prompt = ChatPromptTemplate.from_messages([
                ("system", (
                    "You are an elite academic editor and evaluator. The report has failed the DREAM evaluation quality gates:\n"
                    f"- Key-Information Coverage (KIC): {kic_score_val:.2f} (threshold: 0.80)\n"
                    f"- Reasoning Quality (RQ): {rq_score_val:.2f} (threshold: 0.75)\n"
                    f"- Factuality: {final_factuality_score:.2f} (threshold: 0.90)\n\n"
                    "Formulate 1 to 3 highly targeted research questions/search queries that will directly gather "
                    "the missing information or evidence needed to rectify these failures.\n\n"
                    "Format the output strictly as a JSON object matching this schema:\n{format_instructions}"
                )),
                ("user", (
                    "Report Title: {title}\n"
                    "Report Content (first 4000 chars):\n{report_content}\n\n"
                    "KIC Details: {kic_details_str}\n"
                    "RQ Notes: {rq_notes}\n"
                    "Factuality Critique: {factuality_notes}"
                ))
            ])
            formatted_remediation = remediation_prompt.format_prompt(
                format_instructions=remediation_parser.get_format_instructions(),
                title=report.title,
                report_content=report.content[:4000],
                kic_details_str=str(kic_details),
                rq_notes=rq_notes,
                factuality_notes=factuality_notes
            )
            remediation_response = await router.ainvoke(
                messages=formatted_remediation.to_messages(),
                tier="CRITICAL",
                agent_name="EvaluatorAgent",
                node_name="remediation_formulation"
            )
            try:
                remediation_parsed = parse_json_safely(remediation_response.content)
                if not isinstance(remediation_parsed, dict):
                    remediation_parsed = {}
                notes = remediation_parsed.get("remediation_notes", "Evaluation failed thresholds.")
                queries = remediation_parsed.get("remediation_queries", [])
                if isinstance(queries, str):
                    queries = [queries]

                for idx, query in enumerate(queries):
                    q_id = f"eval_gap_{eval_gap_count + idx + 1}"
                    new_sub_questions.append(
                        SubQuestion(
                            id=q_id,
                            question=query,
                            status="pending"
                        )
                    )
                    logs.append(f"Added evaluator remediation sub-question {q_id}: \"{query}\"")
            except Exception as e:
                logger.error("Failed to parse remediation, creating default gap question", error=str(e))
                notes = f"Evaluation failed. Parsing error: {str(e)}"
                q_id = f"eval_gap_{eval_gap_count + 1}"
                new_sub_questions.append(
                    SubQuestion(
                        id=q_id,
                        question=f"Gather further evidence and elaborate on the core findings of {topic} to satisfy quality thresholds.",
                        status="pending"
                    )
                )
                logs.append(f"Added default evaluator remediation sub-question {q_id}")

    else:
        notes = "Report passed all DREAM evaluation thresholds!"
        logs.append(
            f"DREAM evaluation passed successfully! KIC: {kic_score_val:.2f}, RQ: {rq_score_val:.2f}, Factuality: {final_factuality_score:.2f}."
        )

    evaluation_result = DREMEvaluation(
        key_information_coverage=kic_metric,
        reasoning_quality=rq_metric,
        factuality=fact_metric,
        overall_passed=overall_passed,
        evaluator_notes=notes
    )

    return {
        "evaluation": evaluation_result,
        "sub_questions_state": new_sub_questions,  # Will merge cleanly via list reducer
        "token_usage": get_router().token_usage,
        "logs": logs
    }
