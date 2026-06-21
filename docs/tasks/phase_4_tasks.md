# Phase 4 Checklist: Advanced Search & Planning

Use this task list to implement Phase 4. Mark tasks as completed `[x]` as you finish them. Do not implement code beyond Phase 4.

---

## Step 0: Model Context Protocol (MCP) Client & Servers

### [x] Task 4.0: MCP Client Integration & Hub
- **Description**: 
  - Build a generic python-native MCP client capable of connecting to, calling, and managing MCP servers.
  - Integrate standard servers: Sequential Thinking (`@modelcontextprotocol/server-sequential-thinking`), Knowledge Graph, and a headless Chrome/Puppeteer server.
- **Files**:
  - **`core/mcp_client.py`** [NEW]
  - **`config/settings.py`** (add MCP server configurations)
- **Verification Command**:
  ```bash
  uv run pytest tests/unit/test_mcp_client.py
  ```

---

## Step 1: Plan-MCTS (Plan-Space Monte Carlo Tree Search)

### [x] Task 4.1: Plan-MCTS Planning Engine
- **Description**: Implement the natural language-based Plan-Space Monte Carlo Tree Search (Plan-MCTS) engine.
  - Reason over natural language research intents (subplans) rather than DOM-level clicking.
  - Governed by the Upper Confidence Bound Applied to Trees (UCT) formula.
  - Implement node definition (state, actions, visits, total_reward), selection, expansion, simulation, and backpropagation.
  - Add dynamic plan repair rules to recover from web bugs or dead-ends.
- **Files**:
  - **`planning/mcts_engine.py`** [NEW]
- **Verification Command**:
  ```bash
  uv run pytest tests/unit/test_mcts_engine.py
  ```

### [x] Task 4.2: Puppeteer/Playwright Web Exploration Integration
- **Description**: Connect the MCTS planning engine with the headless browser MCP tool to execute dynamic search/navigation tasks.
- **Files**:
  - **`planning/browser_explorer.py`** [NEW]
- **Verification Command**:
  ```bash
  uv run pytest tests/unit/test_browser_explorer.py
  ```

---

## Step 2: Mango URL Router (Thompson Sampling)

### [x] Task 4.3: Thompson Sampling Starting-Point Selector
- **Description**: Implement the Mango starting-point optimization algorithm modeled as a Multi-Armed Bandit.
  - Prevents redundant web traversal by keeping Beta distributions for candidate URL starting points.
  - Updates rewards (alpha/beta parameters) based on the relevance and factuality density of scraped content.
- **Files**:
  - **`planning/mango_router.py`** [NEW]
- **Verification Command**:
  ```bash
  uv run pytest tests/unit/test_mango_router.py
  ```

---

## Step 3: Graph Node & Researcher Integration

### [ ] Task 4.4: Advanced Researcher Node Wiring
- **Description**: Replace the basic parallel researcher search tool with the advanced MCTS explorer and Mango URL router inside the parallel researcher node loop.
- **Files**:
  - **`core/nodes/research.py`** (integrate MCTS and Mango)
- **Verification Command**:
  ```bash
  uv run pytest tests/unit/test_research_node_advanced.py
  ```

---

## Step 4: Full System E2E & Verification

### [ ] Task 4.5: Advanced Planning E2E Integration Test
- **Description**: Write a complete end-to-end integration test validating that research tasks utilize MCP Sequential Thinking, memory graphs, and the MCTS planning engine successfully under mock environments.
- **Files**:
  - **`tests/integration/test_advanced_planning_e2e.py`** [NEW]
- **Verification Command**:
  ```bash
  uv run pytest tests/integration/test_advanced_planning_e2e.py
  ```
