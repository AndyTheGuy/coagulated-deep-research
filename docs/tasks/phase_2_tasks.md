# Phase 2 Checklist: Search & Research

Use this task list to implement Phase 2. Mark tasks as completed `[x]` as you finish them. Do not implement code beyond Phase 2.

---

## Step 0: Phase 1 Bugfixes & Stabilization

### [ ] Task 2.0a: Critical Bugfixes
- **Description**: Fix committed secrets in Docker files, lazy-initialize LLMRouter to avoid import-time crashes, fix vector store ID collision bug, and correct the embeddings single-text dimension bug.
- **Files**:
  - `docker/docker-compose.yml`
  - `docker/searxng/settings.yml`
  - `core/nodes/scoping.py`
  - `db/vector_store.py`
  - `db/embeddings.py`

### [ ] Task 2.0b: Pydantic Constraints & Async Wrappers
- **Description**: Add status enums and validation ranges to Pydantic models. Wrap blocking synchronous embedding/vector store calls in async wrappers (using `asyncio.to_thread`) to preserve the async-first rule.
- **Files**:
  - **`core/models.py`**
  - **`db/vector_store.py`**
  - **`db/embeddings.py`**

### [ ] Task 2.0c: Phase 2 Data Schema Updates
- **Description**: Update Pydantic models in `core/models.py` with schema extensions needed for Phase 2 data flows (e.g. researcher nodes state, cached document format, query variants).
- **Files**:
  - **`core/models.py`**

---

## Step 1: Search & Scraper Clients

### [x] Task 2.1: SearXNG API Client Wrapper
- **Description**: Implement the async SearXNG client wrapper that queries the self-hosted Docker SearXNG instance and returns structured search results.
- **Files**:
  - **`search/searxng.py`**
- **Acceptance Criteria**:
  - Functions use `httpx` to perform async GET queries.
  - Correctly parses JSON responses from the `http://localhost:8080` endpoint.
  - Handles errors (network failure, timeout) by raising custom search exceptions.
- **Verification Command**:
  ```bash
  uv run pytest tests/unit/test_searxng.py
  ```

### [x] Task 2.2: DuckDuckGo Fallback Client
- **Description**: Implement a fallback search client using the `duckduckgo-search` library to run searches if SearXNG is unavailable.
- **Files**:
  - **`search/ddg.py`**
- **Acceptance Criteria**:
  - Uses the async DDG client to query results.
  - Standardizes raw outputs to match the `SearchResult` Pydantic schema.
- **Verification Command**:
  ```bash
  uv run pytest tests/unit/test_ddg.py
  ```

### [x] Task 2.3: URL Scraping & Content Extraction
- **Description**: Build a robust, async web scraper using `httpx` and `BeautifulSoup` to extract clean markdown or plain text content from URLs.
- **Files**:
  - **`search/scraper.py`**
- **Acceptance Criteria**:
  - Fetches page source asynchronously with User-Agent spoofing and standard timeouts.
  - Extracts clean content (strips script, style, nav, footer tags).
  - Uses HTML-to-markdown conversion or structured text extraction.
- **Verification Command**:
  ```bash
  uv run pytest tests/unit/test_scraper.py
  ```

---

## Step 2: Search Pipeline Logic

### [x] Task 2.4: Deduplication Logic
- **Description**: Implement URL and semantic deduplication to prevent analyzing redundant pages.
- **Files**:
  - **`search/dedup.py`**
- **Acceptance Criteria**:
  - URL-based exact matching deduplication.
  - Semantic similarity deduplication using embeddings (removes pages with >0.95 cosine similarity).
- **Verification Command**:
  ```bash
  uv run pytest tests/unit/test_dedup.py
  ```

### [x] Task 2.5: Reciprocal Rank Fusion (RRF)
- **Description**: Implement RRF to merge and rank search result sets from SearXNG and DDG.
- **Files**:
  - **`search/fusion.py`**
- **Acceptance Criteria**:
  - Implements standard RRF scoring formula: \(RRF(d) = \sum_{m \in M} \frac{1}{k + r_m(d)}\).
  - Merges rankings deterministically.
- **Verification Command**:
  ```bash
  uv run pytest tests/unit/test_fusion.py
  ```

### [x] Task 2.5.5: Query Diversification Utility
- **Description**: Implement query diversification utility to generate 3-5 query variants for a research sub-question using a BULK/STANDARD tier LLM call.
- **Files**:
  - **`search/diversify.py`**
- **Acceptance Criteria**:
  - Diversifies a sub-question into multiple query strings.
- **Verification Command**:
  ```bash
  uv run pytest tests/unit/test_diversify.py
  ```

---

## Step 3: Multi-Agent Topology & Nodes

### [x] Task 2.6: Supervisor Router
- **Description**: Implement the routing logic that parses the research brief, allocates sub-questions to researcher agents, and determines execution paths.
- **Files**:
  - **`core/router.py`**
- **Acceptance Criteria**:
  - Dynamically generates sub-tasks and routes researcher execution.
- **Verification Command**:
  ```bash
  uv run pytest tests/unit/test_router.py
  ```

### [x] Task 2.7: Parallel Researcher Agents
- **Description**: Implement the researcher agent nodes which run parallel queries, scrape relevant pages, and summarize findings in isolation.
- **Files**:
  - **`core/nodes/research.py`**
- **Acceptance Criteria**:
  - Node function executes asynchronously in parallel (using StateGraph send / Map-Reduce).
  - Employs LLM to summarize raw content relative to the specific sub-question.
- **Verification Command**:
  ```bash
  uv run pytest tests/unit/test_research.py
  ```

### [x] Task 2.8: Context Aggregator
- **Description**: Implement the context aggregator node to combine findings from all researcher agents, filter redundant details, and prepare standard context.
- **Files**:
  - **`core/nodes/aggregator.py`**
- **Acceptance Criteria**:
  - Merges summarized context blocks, formats metadata references.
- **Verification Command**:
  ```bash
  uv run pytest tests/unit/test_aggregator.py
  ```

---

## Step 4: Cache & Integration

### [x] Task 2.9: Semantic Cache Layer
- **Description**: Build a cache layer to store search results and scrape content.
- **Files**:
  - **`db/cache.py`**
- **Acceptance Criteria**:
  - Caches documents locally using SQLite/shelve.
  - Implements semantic similarity lookups for cache hits.
- **Verification Command**:
  ```bash
  uv run pytest tests/unit/test_cache.py
  ```

### [x] Task 2.10: Graph Wiring Update
- **Description**: Wire the new supervisor, parallel researchers, and context aggregator nodes into the LangGraph state machine.
- **Files**:
  - **`core/graph.py`**
- **Acceptance Criteria**:
  - Updates GraphState to support map-reduce execution paths.
  - Successfully compiles StateGraph with the complete Phase 2 layout.
- **Verification Command**:
  ```bash
  uv run python -c "from core.graph import compile_graph; compile_graph()"
  ```

### [x] Task 2.11: Search Pipeline Integration Test
- **Description**: Create an end-to-end integration test validating the entire search pipeline (query -> diversify -> parallel search -> scrape -> dedup -> fuse -> cache -> aggregate).
- **Files**:
  - **`tests/integration/test_search_pipeline.py`**
- **Verification Command**:
  ```bash
  uv run pytest tests/integration/test_search_pipeline.py
  ```
