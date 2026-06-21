# Phase 3 Checklist: Verification & Report Writing

Use this task list to implement Phase 3. Mark tasks as completed `[x]` as you finish them. Do not implement code beyond Phase 3.

---

## Step 0: Prerequisites & Pydantic Models

### [x] Task 3.0: Dependency Installation & Pydantic Model Updates
- **Description**: 
  - Install the fuzzy string matching library `rapidfuzz` using `uv add rapidfuzz` to add it to `pyproject.toml`.
  - Add any new Pydantic models required by Phase 3 (e.g. `Claim`, `VerifiedSource`, `QuoteVerification`, `VerificationResult`, `ReportConfidenceScore`) to `core/models.py` before building nodes or pipelines. This guarantees complete type-safety across all subsequent modules.
- **Files**:
  - **`pyproject.toml`** (dependency installation)
  - **`core/models.py`** (schema definitions)
- **Verification Command**:
  ```bash
  uv run pytest tests/unit/test_models.py
  ```

---

## Step 1: Claim Extraction

### [x] Task 3.1: Claim Extractor
- **Description**: Implement the LLM-powered claim extractor that parses a draft report into individual factual `Claim` objects. Each claim must be tied to its source section, supporting quotes, and source URLs.
- **Files**:
  - `verification/claim_extractor.py`
- **Acceptance Criteria**:
  - Accepts a `Report` object and returns `List[Claim]`.
  - Each `Claim` has `claim_text`, `section`, `supporting_quotes`, `source_urls` populated.
  - Uses a STANDARD-tier LLM call with a structured Pydantic output parser.
- **Verification Command**:
  ```bash
  uv run pytest tests/unit/test_claim_extractor.py
  ```

---

## Step 2: Source & Quote Verification

### [x] Task 3.2: Source Checker
- **Description**: Implement async source checker that verifies each cited URL is accessible (HTTP 200) and retrieves its content (or pulls from the semantic cache built in Phase 2).
- **Files**:
  - `verification/source_checker.py`
- **Acceptance Criteria**:
  - Async HTTP HEAD/GET check against each cited URL.
  - Cache-first lookup before making a live network request.
  - Sets `VerifiedSource.accessible = False` and records the error on failure.
- **Verification Command**:
  ```bash
  uv run pytest tests/unit/test_source_checker.py
  ```

### [x] Task 3.3: Quote Verifier
- **Description**: Implement the fuzzy quote verifier that checks whether each literal quote from a claim actually appears in its cited source document.
- **Files**:
  - `verification/quote_verifier.py`
- **Acceptance Criteria**:
  - Uses `rapidfuzz` (or `difflib`) to compute match score between claim quotes and cached source content.
  - Threshold: exact match = 1.0, fuzzy ≥ 0.85 = PASS, < 0.85 = FAIL.
  - Returns a `QuoteVerification` result per quote.
- **Verification Command**:
  ```bash
  uv run pytest tests/unit/test_quote_verifier.py
  ```

---

## Step 3: Confidence Scoring & Remediation

### [ ] Task 3.4: Confidence Scorer
- **Description**: Compute a per-claim confidence score based on source accessibility, quote verification result, and cross-reference count.
- **Files**:
  - `verification/confidence_scorer.py`
- **Acceptance Criteria**:
  - `HIGH` confidence: ≥2 independent verified sources.
  - `MEDIUM` confidence: 1 verified source.
  - `FLAGGED`: 0 verified sources or quote verification failed.
  - Confidence score stored as float 0.0–1.0 on each `Claim`.
- **Verification Command**:
  ```bash
  uv run pytest tests/unit/test_confidence_scorer.py
  ```

### [ ] Task 3.5: Verification Pipeline Orchestrator
- **Description**: Wire all 4 verification stages into a single `run_verification_pipeline()` function that takes a `Report` and returns a fully verified/annotated report with confidence metadata.
- **Files**:
  - `verification/pipeline.py`
- **Acceptance Criteria**:
  - Runs stages 1–4 sequentially with structured logging at each stage.
  - Returns a report with all `Claim` objects updated with verification statuses and scores.
- **Verification Command**:
  ```bash
  uv run pytest tests/unit/test_verification_pipeline.py
  ```

---

## Step 4: Graph Nodes — Verifier & Writer

### [ ] Task 3.6: Adversarial Verifier Node
- **Description**: Implement the adversarial verifier LangGraph node. It critiques the aggregated context, identifies logical holes and unverified claims, and decides whether to route back to the supervisor for gap-filling or forward to the writer.
- **Files**:
  - `core/nodes/verifier.py`
- **Acceptance Criteria**:
  - CRITICAL-tier LLM call that produces a structured critique of the research context.
  - Routes back to the supervisor if gaps are found, otherwise proceeds to writer.
  - Integrates with the verification pipeline from Task 3.5.
- **Verification Command**:
  ```bash
  uv run pytest tests/unit/test_verifier_node.py
  ```

### [ ] Task 3.7: Report Writer Node
- **Description**: Implement the academic report writer LangGraph node that compiles verified findings into a structured markdown report with literal quote citations.
- **Files**:
  - `core/nodes/writer.py`
- **Acceptance Criteria**:
  - CRITICAL-tier LLM call producing a structured `Report` object.
  - Report includes ≥20 cited sources, every factual claim has a supporting quote.
  - Outputs well-formed markdown with section headers, citations section, and confidence metadata.
- **Verification Command**:
  ```bash
  uv run pytest tests/unit/test_writer_node.py
  ```

---

## Step 5: Graph Wiring

### [ ] Task 3.9: Graph Wiring Update
- **Description**: Wire the verifier and writer nodes into the existing LangGraph `StateGraph`. Implement conditional routing from the verifier back to the supervisor when gaps are found.
- **Files**:
  - `core/graph.py`
- **Acceptance Criteria**:
  - Graph compiles successfully with the full Phase 3 topology.
  - Conditional edges: Verifier → Supervisor (gaps) or Verifier → Writer (verified).
- **Verification Command**:
  ```bash
  uv run python -c "from core.graph import compile_graph; compile_graph()"
  ```

---

## Step 6: Integration Testing

### [ ] Task 3.10: Verification Pipeline Integration Test
- **Description**: Create an end-to-end integration test that runs a known report through the full verification pipeline and asserts correct confidence scoring behavior.
- **Files**:
  - `tests/integration/test_verification_pipeline.py`
  - `tests/fixtures/sample_report.md`
  - `tests/fixtures/sample_sources/` (cached source documents for quote verification)
- **Verification Command**:
  ```bash
  uv run pytest tests/integration/test_verification_pipeline.py
  ```

### [ ] Task 3.11: End-to-End Research + Write Graph Test
- **Description**: Integration test that runs a full mock graph end-to-end from query through scoping, research aggregation, verification, and report writing.
- **Files**:
  - `tests/integration/test_full_pipeline_e2e.py`
- **Verification Command**:
  ```bash
  uv run pytest tests/integration/test_full_pipeline_e2e.py
  ```
