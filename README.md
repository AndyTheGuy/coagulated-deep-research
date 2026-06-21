# Ultimate Deep Researcher

An autonomous, self-correcting deep research agent framework built using LangGraph, LangChain, Qdrant, and Python.

## Core Features

- **Multi-Agent Orchestration**: Powered by LangGraph StateGraph (Scoping, Supervisor Router, Parallel Researchers, Context Aggregator, Adversarial Verifier, Report Writer, Fact Checker, DREAM Evaluator).
- **Zero-Tolerance Hallucination Prevention**: 5-stage verification pipeline verifying factual claims and grounding them in literal, fuzzy-matched quotes from source documents.
- **Cost-Optimized Hybrid LLM Routing**: Dispatches critical reasoning tasks to Google Vertex AI (Gemini 3.5 Flash) and bulk/standard processing to a local FreeLLMAPI instance, complete with automatic failovers.
- **Unified Search Pipeline**: Merges self-hosted SearXNG with DuckDuckGo fallback, utilizing Reciprocal Rank Fusion (RRF), URL deduplication, and isolated URL markdown scraping.
- **DREAM Evaluation**: Grades final reports against Key-Information Coverage (KIC), Reasoning Quality (RQ), and Factuality thresholds.
- **Interactive UI**: Real-time progress log stream and live cost estimation tracked directly inside a Streamlit interface.

---

## Technical Stack

| Component | Technology |
|---|---|
| **Language** | Python 3.11+ |
| **Agent Framework** | LangGraph, LangChain |
| **LLMs** | Google Vertex AI (Gemini 3.5 Flash), FreeLLMAPI |
| **Vector DB** | Qdrant (in-memory -> local Docker) |
| **Search Engine** | SearXNG (Docker), DuckDuckGo |
| **User Interface** | Streamlit |
| **Environment** | `uv` package manager |

---

## Getting Started

### Prerequisites

- Python 3.11+
- `uv` package manager installed
- Docker & Docker Compose
- Google Cloud Application Default Credentials (ADC) configured

### Quickstart

1. Clone the repository:
   ```bash
   git clone https://github.com/AndyTheGuy/coagulated-deep-research.git
   cd coagulated-deep-research
   ```

2. Initialize dependencies:
   ```bash
   uv sync
   ```

3. Spin up Docker services (SearXNG + FreeLLMAPI):
   ```bash
   docker compose -f docker/docker-compose.yml up -d
   ```

4. Launch the Streamlit application:
   ```bash
   uv run streamlit run ui/app.py
   ```

5. Run the CLI fallback (optional):
   ```bash
   uv run python -m cli.main
   ```

---

## Development & Git Workflow

This project adheres to **Trunk-Based Development** and the **Save Point Pattern** for AI-driven code generation.

- **Main Branch**: `main` must remain stable and deployable at all times.
- **Feature Branches**: Run changes in short-lived feature branches (`feature/<description>`) and merge within 1-3 days.
- **Session Continuity**: We utilize `.agents/GEMINI.md` to load project context automatically across clean chat sessions.
- **Testing**: Run pytest suite before commits:
  ```bash
  uv run pytest
  ```
