# Phase 2 Walkthrough: Search & Research

This document preserves the changes, architecture choices, and testing verification for completed tasks in Phase 2.

---

## 1. Search & Scraper Clients

### Task 2.1: SearXNG API Client Wrapper
- **Implementation**: Developed `search/searxng.py` implementing the async SearXNG client wrapper `search_searxng` to query the self-hosted SearXNG Docker instance on `http://localhost:8080`.
- **Error Handling**: Raised custom exceptions (`SearchError` and `SearXNGError`) to handle timeouts, HTTP errors, and JSON response parsing errors.
- **Verification**: Created mock-based unit tests in `tests/unit/test_searxng.py` covering success and failure states. (5/5 tests passed).

### Task 2.2: DuckDuckGo Fallback Client
- **Implementation**: Created `search/ddg.py` which wraps the synchronous `duckduckgo-search` library API (`DDGS().text`) in `asyncio.to_thread` to maintain our strict **async-first rule** and prevent blocking the asyncio event loop.
- **Verification**: Output is standardized into `SearchResult` Pydantic schemas. Verified in `tests/unit/test_ddg.py` (3/3 tests passed).

### Task 2.3: URL Scraping & Content Extraction
- **Implementation**: Created `search/scraper.py` which fetches pages asynchronously via `httpx.AsyncClient` with User-Agent spoofing and strips scripts, styles, footers, and nav tags before converting standard tags to clean structured markdown text via `BeautifulSoup`.
- **Verification**: Tested title parsing, markdowns, and HTTP timeouts in `tests/unit/test_scraper.py` (4/4 tests passed).

---

## 2. Search Pipeline Logic

### Task 2.4: Deduplication Logic
- **Implementation**: Created `search/dedup.py` implementing exact-URL matching and semantic similarity-based document deduplication. Cosine similarity threshold is set to `0.95`, discarding redundant duplicate text blocks using local embeddings.
- **Verification**: Unit tested in `tests/unit/test_dedup.py` (5/5 tests passed).

### Task 2.5: Reciprocal Rank Fusion (RRF)
- **Implementation**: Created `search/fusion.py` implementing reciprocal rank fusion:
  \[RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}\]
  with a default constant \(k = 60\) and deterministic tie-breaker sorting based on score and URL.
- **Verification**: Tested in `tests/unit/test_fusion.py` (4/4 tests passed).

### Task 2.5.5: Query Diversification Utility
- **Implementation**: Created `search/diversify.py` leveraging standard/bulk tier LLM calls to generate 3-5 highly distinct, targeted search query variants for any given research sub-question.
- **Verification**: Unit tested in `tests/unit/test_diversify.py` (4/4 tests passed).

---

## 3. Multi-Agent Topology & Nodes

### Task 2.6: Supervisor Router
- **Implementation**: Created `core/router.py` implementing routing rules. Parses the `ResearchBrief` sub-questions list, allocates them to dynamic parallel researcher agent nodes, and compiles the overall execution flow.
- **Verification**: Unit tested in `tests/unit/test_router.py` (6/6 tests passed).

### Task 2.7: Parallel Researcher Agents
- **Implementation**: Created `core/nodes/research.py` implementing parallel researcher agent node execution. Under a Map-Reduce model, each researcher runs in an isolated context on its assigned sub-question: performing query diversification, parallel search, URL/semantic deduplication, and compiling a results summary.
- **Verification**: Unit tested in `tests/unit/test_research.py` (2/2 tests passed).

### Task 2.8: Context Aggregator
- **Implementation**: Created `core/nodes/aggregator.py` which merges individual researcher nodes' summarized context blocks, formats metadata citations, eliminates duplicates, and prepares the overall consolidated context.
- **Verification**: Unit tested in `tests/unit/test_aggregator.py` (2/2 tests passed).

---

## 4. Cache & Integration

### Task 2.9: Semantic Cache Layer
- **Implementation**: Created `db/cache.py` implementing a robust local semantic cache using SQLite. On lookups, if an entry has a cosine similarity score exceeding `0.90` (using local sentence-transformers embeddings), a cache hit is returned to save network latency and token costs.
- **Verification**: Unit tested in `tests/unit/test_cache.py` (4/4 tests passed).

### Task 2.10: Graph Wiring Update
- **Implementation**: Modified `core/graph.py` to integrate parallel researcher nodes and the aggregator. Configured StateGraph State model (`GraphState`) to use additive list/dict reducer attributes to safely merge state updates across map-reduce transitions.
- **Verification**: Graph compiles successfully with the complete Phase 2 layout.

### Task 2.11: Search Pipeline Integration Test
- **Implementation**: Created `tests/integration/test_search_pipeline.py` implementing an end-to-end pipeline simulation. Mocks LLM responses, scrapers, and embedding providers to achieve complete determinism and offline speed, while running a live StateGraph execution loop.
- **Verification**: 
  - Integrated **Scoping Token Tracking** into scoping nodes (`core/nodes/scoping.py`).
  - Solved pytest state-leakage by resetting module cached `_router` singletons inside `core.nodes.scoping` and `search.diversify` back to `None` in the test fixture.
  - Aligned token assertions with mathematically precise counts reflecting StateGraph sequential node aggregation.
  - Full suite: **65/65 tests passed successfully (100% pass rate)**.
