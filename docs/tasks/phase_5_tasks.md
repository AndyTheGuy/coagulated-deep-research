# Phase 5 Checklist: Evaluation & Polish

Use this task list to implement Phase 5. Mark tasks as completed `[x]` as you finish them. Do not implement code beyond Phase 5.

---

## Step 0: DREAM Evaluator Node

### [x] Task 5.0: Pydantic Schema Definitions for Evaluation
- **Description**: Add Pydantic schemas representing the evaluation reports, scores, and criteria.
- **Files**:
  - **`core/models.py`** (add `DREMEvaluation`, `MetricScore`)
- **Verification Command**:
  ```bash
  uv run pytest tests/unit/test_models.py
  ```

### [x] Task 5.1: DREAM Evaluation Node Implementation
- **Description**: Build the DREAM evaluation engine inside the graph.
  - Implement **Key-Information Coverage (KIC)**: Convert key facts to binary questions via LLM, evaluate coverage (threshold ≥ 0.80).
  - Implement **Reasoning Quality (RQ)**: Cross-reference logical claims against ground truth via LLM (threshold ≥ 0.75).
  - Implement **Factuality**: Assess citation health, quote accuracy (using fuzz matching), and URL liveness (threshold ≥ 0.90).
  - Define the graph conditional routing: if any score falls below its threshold, route back to `supervisor_node` for remediating research.
- **Files**:
  - **`core/nodes/evaluator.py`** [NEW]
- **Verification Command**:
  ```bash
  uv run pytest tests/unit/test_evaluator_node.py
  ```

---

## Step 1: Real-time UI and Progress Log Stream

### [ ] Task 5.2: Structured Log Parser & Streaming Log Component
- **Description**: Connect Streamlit with `structlog` to read and stream log transitions in real-time.
  - Formatted as `[timestamp] [agent_name] [node_name] action_description`.
  - Color-code by agent role and log severity (blue=info, yellow=warning, red=error).
- **Files**:
  - **`ui/log_streamer.py`** [NEW]
- **Verification Command**:
  ```bash
  uv run pytest tests/unit/test_log_streamer.py
  ```

### [ ] Task 5.3: Streamlit Interface Construction
- **Description**: Implement a high-premium, responsive Streamlit dashboard interface based on the UI specifications.
  - **Research Input Panel**: Topic, optional constraints, target counts, "Start Research" button.
  - **Clarification Dialog**: Interactive inline input if the scoping agent flags ambiguity.
  - **Progress Log Container**: Real-time log component.
  - **Cost Estimator**: Track Vertex AI and FreeLLMAPI calls, token metrics, and cost, alongside estimated remaining cost.
  - **Report Viewer**: Final report rendered in markdown with inline clickable links, plus confidence badges.
- **Files**:
  - **`ui/app.py`** [MODIFY]
- **Verification Command**:
  - Manually launch:
    ```bash
    streamlit run ui/app.py
    ```

---

## Step 2: Cost Tracking & Performance Optimization

### [ ] Task 5.4: Live Cost Tracking Integration
- **Description**: Integrate the LLMRouter token usage statistics into the UI state and state graph.
  - Track Vertex AI prompt and completion counts, pricing them according to Gemini 3.5 Flash 2026 pricing.
  - Track FreeLLMAPI calls separately as zero-cost metrics.
- **Files**:
  - **`core/llm_router.py`** (ensure robust tracking is integrated with graph state)
- **Verification Command**:
  ```bash
  uv run pytest tests/unit/test_llm_router.py
  ```

---

## Step 3: End-to-End Evaluation & Verification

### [ ] Task 5.5: Phase 5 End-to-End Integration Test
- **Description**: Write a complete end-to-end integration test validating the DREAM evaluator conditional routing loop and Streamlit data streaming compatibility.
- **Files**:
  - **`tests/integration/test_dream_evaluation_e2e.py`** [NEW]
- **Verification Command**:
  ```bash
  uv run pytest tests/integration/test_dream_evaluation_e2e.py
  ```
