import asyncio
import re
import structlog
from typing import Any, Dict, List, Tuple
from langchain_core.prompts import ChatPromptTemplate

from core.models import ResearcherInput, SubQuestion, SearchResult, VerifiedSource
from core.llm_router import LLMRouter
from db.embeddings import LocalEmbeddings
from search.searxng import search_searxng
from search.ddg import search_ddg
from search.dedup import deduplicate_by_url, deduplicate_semantically
from search.fusion import reciprocal_rank_fusion
from planning.mcts_engine import PCTSEngine
from planning.mango_router import MangoRouter
from planning.browser_explorer import BrowserExplorer
from core.mcp_client import MCPHub
from search.scraper import scrape_url

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

async def evaluate_scraped_relevance(router: LLMRouter, question: str, content: str) -> float:
    """Evaluate factual density/relevance of scraped text relative to target question."""
    if not content or not content.strip():
        return 0.0
        
    # Truncate content to avoid token blowup during fast evaluation
    truncated_content = content[:3000]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an objective research auditor. Rate the factual density and relevance of the scraped text "
            "relative to the target question. Provide a single numerical score between 0.0 (useless/unrelated) "
            "and 1.0 (extremely helpful, dense facts answering the question).\n"
            "Respond with ONLY the numerical score. Do not explain your score."
        )),
        ("user", (
            "Target Question: {question}\n\n"
            "Scraped Content:\n{content}\n\n"
            "Score (0.0 to 1.0):"
        ))
    ])
    
    try:
        response = await router.ainvoke(
            messages=prompt.format_messages(question=question, content=truncated_content),
            tier="STANDARD",
            agent_name="ResearcherAgent",
            node_name="evaluate_relevance"
        )
        match = re.search(r"([0-9]*\.[0-9]+|[0-9]+)", response.content.strip())
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
        return 0.5
    except Exception as e:
        logger.warning("Failed to evaluate scraped relevance, using fallback 0.5", error=str(e))
        return 0.5

async def researcher_node(state: ResearcherInput) -> Dict[str, Any]:
    """Advanced Researcher agent node.
    Executes Plan-Space MCTS to find the optimal search queries, retrieves search results,
    deduplicates them, and optimizes starting-point URL selection using Thompson Sampling (Mango).
    """
    sub_q = state.sub_question
    topic = state.topic
    constraints = state.constraints
    
    logger.info("Starting advanced researcher agent", sub_question_id=sub_q.id, question=sub_q.question)
    
    # Initialize a local LLMRouter for isolated and safe token tracking
    router = LLMRouter()
    
    # Update state log
    logs = [f"Advanced researcher node started for sub-question '{sub_q.id}': {sub_q.question}"]
    
    # 1. Plan-MCTS Planning Phase
    logger.info("Starting Plan-MCTS Planning Phase", sub_question_id=sub_q.id)
    mcts_planner = PCTSEngine(router=router)
    try:
        best_intent, queries = await mcts_planner.search(
            sub_question=sub_q.question,
            topic=topic,
            max_iterations=2  # Snap iteration count for fast research loops
        )
        logs.append(f"MCTS Selected Intent: {best_intent}")
    except Exception as e:
        logger.error("MCTS search planning failed, falling back to basic query", sub_question_id=sub_q.id, error=str(e))
        queries = [sub_q.question]
        best_intent = "Direct search fallback"
        
    logs.append(f"MCTS generated {len(queries)} optimal search queries: {', '.join(queries)}")
    
    # 2. Execute parallel searches
    search_tasks = [search_variant(q) for q in queries]
    search_results_lists = await asyncio.gather(*search_tasks)
    
    # 3. Reciprocal Rank Fusion & Deduplication
    fused_results = reciprocal_rank_fusion(search_results_lists)
    deduped_by_url = deduplicate_by_url(fused_results)
    
    embeddings_service = get_embeddings()
    deduped_results = await deduplicate_semantically(deduped_by_url, embeddings_service)
    
    # 4. Multi-Armed Bandit starting point selection (Mango)
    logger.info("Starting Mango starting-point optimization", url_count=len(deduped_results))
    mango = MangoRouter()
    candidates = {}
    url_to_result = {}
    
    for i, res in enumerate(deduped_results[:8]): # Pick top 8 candidates for arms
        # Prior score bootstrapped from rank
        score = max(0.1, min(0.9, 1.0 - (i * 0.1)))
        candidates[res.url] = score
        url_to_result[res.url] = res
        
    mango.add_candidates(candidates)
    
    # Connect MCP Hub for browser automation (e.g., Puppeteer)
    mcp_hub = MCPHub()
    try:
        await mcp_hub.connect_all()
    except Exception as e:
        logger.warning("Failed to connect MCP servers in researcher node, using fallback scraping", error=str(e))
        
    explorer = BrowserExplorer(mcp_hub=mcp_hub, scrape_func=scrape_url)
    
    verified_sources: List[VerifiedSource] = []
    errors_list: List[str] = []
    
    # Thompson Sampling selection & exploration loop
    # Scrape up to 4 URLs
    scraped_count = 0
    for step in range(4):
        url = mango.select_url()
        if not url:
            break
            
        logger.info("Mango selecting arm to scrape", step=step, url=url)
        scraped_count += 1
        
        try:
            explore_res = await explorer.explore_url(url)
            if explore_res.get("success", False):
                reward = await evaluate_scraped_relevance(router, sub_q.question, explore_res["content"])
                mango.update_reward(url, reward)
                
                verified_sources.append(
                    VerifiedSource(
                        url=url,
                        title=url_to_result[url].title if explore_res.get("title") in (None, "", "Extracted Page", "Unknown Title") else explore_res["title"],
                        content=explore_res["content"],
                        accessible=True
                    )
                )
                logs.append(f"Scraped URL: {url} | Method: {explore_res['method']} | Mango Reward: {reward:.2f}")
            else:
                mango.update_reward(url, 0.0)
                err_msg = explore_res.get("error_message", "Exploration failed")
                errors_list.append(f"Scraping failed for {url}: {err_msg}")
                verified_sources.append(
                    VerifiedSource(
                        url=url,
                        title=url_to_result[url].title or "Unknown Title",
                        content="",
                        accessible=False,
                        error_message=err_msg
                    )
                )
                logs.append(f"Scraping failed for {url} | Error: {err_msg}")
        except Exception as e:
            logger.error("Exception during Mango URL exploration", url=url, error=str(e))
            mango.update_reward(url, 0.0)
            errors_list.append(f"Exploration error for {url}: {str(e)}")
            verified_sources.append(
                VerifiedSource(
                    url=url,
                    title=url_to_result[url].title or "Unknown Title",
                    content="",
                    accessible=False,
                    error_message=str(e)
                )
            )
            logs.append(f"Exploration error for {url}: {str(e)}")

    await mcp_hub.shutdown()
    logs.append(f"Completed starting-point URL selection via Mango (scraped {scraped_count} candidates)")

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
            sources_text += f"--- Source {idx+1} ---\nTitle: {src.title}\nURL: {src.url}\nContent:\n{src.content[:10000]}\n\n" # Safe content truncation for synthesis
            
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
