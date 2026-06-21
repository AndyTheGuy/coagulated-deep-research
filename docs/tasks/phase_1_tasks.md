# Phase 1 Checklist: Foundation

Use this task list to implement Phase 1. Mark tasks as completed `[x]` as you finish them. Do not implement code beyond Phase 1.

---

## Step 1: Scaffolding & Docker Setup

### [x] Task 1.1: Project Scaffolding
- **Description**: Initialize the Python project and configuration files using `uv`.
- **Files**:
  - `pyproject.toml`
  - `.gitignore` (ignore `.env`, `.venv/`, `__pycache__/`, `.pytest_cache/`)
  - `.env.example`
- **Acceptance Criteria**:
  - `pyproject.toml` contains all dependencies listed in the spec.
  - Running `uv sync` installs all dependencies without conflicts.
- **Verification Command**:
  ```bash
  uv sync
  ```

### [x] Task 1.2: Dockerized Infrastructure Setup
- **Description**: Configure a single Docker compose file running SearXNG and FreeLLMAPI. SearXNG must have its JSON API enabled, rate limiters disabled for local queries, and default search engines active.
- **Files**:
  - `docker/docker-compose.yml`
  - `docker/searxng/settings.yml`
- **Acceptance Criteria**:
  - Running `docker compose up -d` starts both containers.
  - Executing a GET request to SearXNG (`http://localhost:8080/?q=test&format=json`) returns search results.
  - Executing a GET request to FreeLLMAPI (`http://localhost:8000/v1/models`) returns a model list.
- **Verification Command**:
  ```bash
  docker compose -f docker/docker-compose.yml up -d
  curl "http://localhost:8080/?q=test&format=json"
  curl "http://localhost:8000/v1/models"
  ```

### [x] Task 1.3: Application Configuration & Settings
- **Description**: Build settings parsing using Pydantic Settings and configure structured JSON logging using `structlog`.
- **Files**:
  - `config/settings.py` (inherits from `BaseSettings`)
  - `config/logging_config.py`
  - `config/__init__.py`
- **Acceptance Criteria**:
  - Reads variables from `.env` or system environment.
  - Variables parsed include Vertex credentials, FreeLLMAPI endpoints, SearXNG URL, logging level.
  - Logging output is structured JSON in non-development modes.
- **Verification Command**:
  ```bash
  uv run python -c "from config.settings import settings; print(settings.model_dump())"
  ```

---

## Step 2: Core Components Scaffolding

### [x] Task 1.4: Define Data Models
- **Description**: Create Pydantic models for all data models, messages, and state schemas. Must define the state structure that flows through the LangGraph nodes.
- **Files**:
  - `core/models.py`
- **Acceptance Criteria**:
  - Defines schemas for: `ResearchBrief`, `SubQuestion`, `SearchResult`, `VerifiedSource`, `Claim`, `Report`, `GraphState`.
  - Type-hinted validation checks out.
- **Verification Command**:
  ```bash
  uv run pytest tests/unit/test_models.py
  ```

### [x] Task 1.5: 3-Tier LLM Router
- **Description**: Implement the routing client that dispatches LLM calls to Google Vertex AI (Gemini 3.5 Flash) or FreeLLMAPI based on the task criticality tier. Must feature retry mechanisms and failover to Vertex AI if FreeLLMAPI times out or fails.
- **Files**:
  - `core/llm_router.py`
- **Acceptance Criteria**:
  - Routes `CRITICAL` calls strictly to Vertex AI.
  - Routes `STANDARD` and `BULK` to FreeLLMAPI.
  - Transparently falls back to Vertex AI if FreeLLMAPI throws an API error or times out.
  - Tracks token usage and counts inputs/outputs per provider.
- **Verification Command**:
  ```bash
  uv run pytest tests/unit/test_llm_router.py
  ```

### [x] Task 1.6: Qdrant Vector DB & Embedding Scaffolding
- **Description**: Scaffolding the `sentence-transformers` embedding service (using `all-mpnet-base-v2`) and client connections to in-memory Qdrant.
- **Files**:
  - `db/embeddings.py`
  - `db/vector_store.py`
- **Acceptance Criteria**:
  - Vector database runs in-memory without needing a local Qdrant server container.
  - Embedding class returns 768-dimension vectors for query text.
  - Support functions to write documents, perform semantic search queries, and filter collections.
- **Verification Command**:
  ```bash
  uv run pytest tests/unit/test_embeddings.py tests/unit/test_vector_store.py
  ```

---

## Step 3: Graph Scaffolding & Scoping Nodes

### [ ] Task 1.7: Scoping & Research Brief Nodes
- **Description**: Build the two scoping nodes: `clarify_with_user` (detecting topic ambiguity) and `write_research_brief` (compiling research plan, questions, constraints).
- **Files**:
  - `core/nodes/scoping.py`
- **Acceptance Criteria**:
  - `clarify_with_user` node outputs clarification questions if the user prompt is vague or missing key info.
  - `write_research_brief` outputs a structured `ResearchBrief` containing sub-questions and scope targets.
- **Verification Command**:
  ```bash
  uv run pytest tests/integration/test_scoping_flow.py
  ```

### [ ] Task 1.8: Graph Definition
- **Description**: Scaffold the LangGraph `StateGraph` definition. Connect the scoping nodes, supervisor placeholder, and terminal output.
- **Files**:
  - `core/graph.py`
- **Acceptance Criteria**:
  - Instantiates `StateGraph` using `GraphState`.
  - Compile method runs successfully.
- **Verification Command**:
  ```bash
  uv run python -c "from core.graph import compile_graph; compile_graph()"
  ```

---

## Step 4: Interface & Entry Points

### [ ] Task 1.9: CLI Entry Fallback
- **Description**: Build a terminal CLI interface to test scoping and brief compilation.
- **Files**:
  - `cli/main.py`
- **Acceptance Criteria**:
  - Runs inside CLI, prompts for search query, runs the compiled scoping portion of the graph, outputs research brief JSON.
- **Verification Command**:
  ```bash
  uv run python -m cli.main
  ```

### [ ] Task 1.10: Streamlit App Shell
- **Description**: Build the baseline Streamlit app UI layout, establishing components for user input, progress logs, and mock outputs.
- **Files**:
  - `ui/app.py`
  - `ui/state.py`
- **Acceptance Criteria**:
  - Basic page renders layout panels: input box, sidebar details, progress logger, cost estimator.
  - Real-time logging outputs to screen.
- **Verification Command**:
  ```bash
  # Run and test page render manually
  uv run streamlit run ui/app.py
  ```

---

## Step 5: Test Scaffolding

### [ ] Task 1.11: Unit and Integration Test Shells
- **Description**: Establish unit test configurations, environment fixtures, and API/search client mock handlers.
- **Files**:
  - `tests/conftest.py`
  - `tests/fixtures/sample_research_brief.json`
- **Acceptance Criteria**:
  - Fixtures provide mock responses for Vertex AI, FreeLLMAPI, and search queries.
  - Tests run deterministically without internet access (using mocks).
- **Verification Command**:
  ```bash
  uv run pytest
  ```
