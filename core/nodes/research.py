import asyncio
import structlog
from typing import Any, Dict, List
from langchain_core.prompts import ChatPromptTemplate

from core.models import ResearcherInput, SubQuestion, SearchResult, VerifiedSource
from core.llm_router import LLMRouter
from db.embeddings import LocalEmbeddings
from search.searxng import search_searxng
from search.ddg import search_ddg
from search.scraper import scrape_url
from search.dedup import deduplicate_by_url, deduplicate_semantically
from search.fusion import reciprocal_rank_fusion
from search.diversify import diversify_query

logger = structlog.get_logger("deep-research")

# Lazy-loaded embeddings model to avoid redundant initialization
_embeddings = None

def get_embeddings() -> LocalEmbeddings:
    global _embeddings
    if _embeddings is None:
        logger.info("Initializing LocalEmbeddings for Researcher Nodes")
        _embeddings = LocalEmbeddings()
    return _embeddings

async def search_variant(query: str) -> List[SearchResult]:
    """Execute search for a single query variant, trying SearXNG first, then falling back to DDG."""
    try:
        results = await search_searxng(query, num_results=10)
        return results
    except Exception as e:
        logger.warning("SearXNG search failed, falling back to DDG", query=query, error=str(e))
        try:
            results = await search_ddg(query, num_results=10)
            return results
        except Exception as e2:
            logger.error("DDG fallback search failed too", query=query, error=str(e2))
            return []

async def researcher_node(state: ResearcherInput) -> Dict[str, Any]:
    """Researcher agent node. Executes async parallel search queries, scrapes pages,
    deduplicates content, and uses LLM to synthesize/summarize findings relative to the sub-question.
    """
    sub_q = state.sub_question
    topic = state.topic
    constraints = state.constraints
    
    logger.info("Starting researcher agent", sub_question_id=sub_q.id, question=sub_q.question)
    
    # Initialize a local LLMRouter for isolated and safe token tracking
    router = LLMRouter()
    
    # Update state log
    logs = [f"Researcher node started for sub-question '{sub_q.id}': {sub_q.question}"]
    
    # Generate search queries (diversification)
    try:
        variants = await diversify_query(sub_q.question, num_variants=3, router=router)
    except Exception as e:
        logger.error("Failed to diversify query", sub_question_id=sub_q.id, error=str(e))
        variants = [sub_q.question]
        
    logs.append(f"Diversified sub-question '{sub_q.id}' into: {', '.join(variants)}")
    
    # Execute searches in parallel across all variants
    search_tasks = [search_variant(v) for v in variants]
    search_results_lists = await asyncio.gather(*search_tasks)
    
    # Merge and rank using Reciprocal Rank Fusion (RRF)
    fused_results = reciprocal_rank_fusion(search_results_lists)
    
    # Deduplicate URL-based
    deduped_by_url = deduplicate_by_url(fused_results)
    
    # Semantic deduplication using sentence-transformers
    embeddings_service = get_embeddings()
    deduped_results = await deduplicate_semantically(deduped_by_url, embeddings_service)
    
    # Select top results to scrape (max 5)
    to_scrape = deduped_results[:5]
    logs.append(f"Selected {len(to_scrape)} URLs to scrape for sub-question '{sub_q.id}'")
    
    # Parallel scrape
    scrape_tasks = [scrape_url(res.url) for res in to_scrape]
    scrape_results = await asyncio.gather(*scrape_tasks, return_exceptions=True)
    
    verified_sources: List[VerifiedSource] = []
    errors_list: List[str] = []
    
    for res, scrape_res in zip(to_scrape, scrape_results):
        if isinstance(scrape_res, Exception):
            err_msg = str(scrape_res)
            logger.warning("Failed to scrape URL", url=res.url, error=err_msg)
            errors_list.append(f"Scraping failed for {res.url}: {err_msg}")
            verified_sources.append(
                VerifiedSource(
                    url=res.url,
                    title=res.title or "Unknown Title",
                    content="",
                    accessible=False,
                    error_message=err_msg
                )
            )
        else:
            title, content = scrape_res
            verified_sources.append(
                VerifiedSource(
                    url=res.url,
                    title=title or res.title or "Unknown Title",
                    content=content,
                    accessible=True
                )
            )
            
    accessible_sources = [s for s in verified_sources if s.accessible and s.content.strip()]
    
    results_summary = ""
    status = "completed"
    
    if not accessible_sources:
        status = "failed"
        results_summary = "Failed to scrape any valid or accessible source content to answer this sub-question."
        logger.error("Researcher node failed - no accessible sources", sub_question_id=sub_q.id)
        logs.append(f"Researcher node failed for sub-question '{sub_q.id}': No accessible sources found")
    else:
        # Synthesize source documents using STANDARD tier LLM
        sources_text = ""
        for idx, src in enumerate(accessible_sources):
            sources_text += f"--- Source {idx+1} ---\nTitle: {src.title}\nURL: {src.url}\nContent:\n{src.content}\n\n"
            
        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an academic/analyst-grade researcher. Your task is to write a comprehensive, "
                "detailed research summary that directly answers the assigned sub-question, based on the provided source contents.\n\n"
                "Overarching Research Topic: {topic}\n"
                "Research Constraints: {constraints}\n\n"
                "Guidelines:\n"
                "1. Ground your answer strictly in the provided sources. Do NOT make up any facts or rely on external knowledge.\n"
                "2. Cite your sources inline using standard markdown links pointing to the exact source URLs, e.g., [Source Title](URL).\n"
                "3. Cite ALL sources that support a claim. If different sources have conflicting information, explicitly present both viewpoints.\n"
                "4. Keep the tone professional, objective, and analytical.\n"
                "5. If the provided sources do not contain enough information to answer the question, state this clearly and summarize what *is* available."
            )),
            ("user", "Sub-question to answer: {question}\n\nSource Documents:\n{sources_text}")
        ])
        
        formatted_prompt = prompt.format_prompt(
            topic=topic,
            constraints=", ".join(constraints) if constraints else "None",
            question=sub_q.question,
            sources_text=sources_text
        )
        
        try:
            logger.info("Synthesizing research summary", sub_question_id=sub_q.id)
            response = await router.ainvoke(
                messages=formatted_prompt.to_messages(),
                tier="STANDARD",
                agent_name="ResearcherAgent",
                node_name="summarize_sources"
            )
            results_summary = response.content
            logger.info("Research summary synthesis complete", sub_question_id=sub_q.id)
            logs.append(f"Successfully researched and synthesized findings for sub-question '{sub_q.id}'")
        except Exception as e:
            logger.error("Failed to synthesize summary", sub_question_id=sub_q.id, error=str(e))
            status = "failed"
            results_summary = f"Failed to synthesize research summary due to LLM error: {str(e)}"
            errors_list.append(f"Synthesis failed for sub-question {sub_q.id}: {str(e)}")
            logs.append(f"Researcher node failed for sub-question '{sub_q.id}': LLM synthesis failure")
            
    # Create the updated SubQuestion object
    completed_sub_q = SubQuestion(
        id=sub_q.id,
        question=sub_q.question,
        status=status,
        assigned_researcher="researcher_node",
        results_summary=results_summary
    )
    
    return {
        "sub_questions_state": [completed_sub_q],
        "search_results": fused_results,
        "verified_sources": verified_sources,
        "token_usage": router.token_usage,
        "logs": logs,
        "errors": errors_list
    }
