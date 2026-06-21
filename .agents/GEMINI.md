# GEMINI.md: Ultimate Deep Researcher Context

This context is loaded automatically at the start of every session. Use it to keep track of state, goals, and coding rules across sessions.

---

## 1. Project Goal & Identity
Build an autonomous, self-correcting deep research agent framework using LangGraph and Python. 
It must perform parallel searches, verify citations with fuzzy matching, and produce academic/commercial-grade reports with zero tolerance for hallucinations.

---

## 2. Core Technology Stack
- **Agent Framework**: LangGraph (StateGraph)
- **LLM SDK**: LangChain (`langchain-google-vertexai`, `langchain-openai`)
- **LLMs**: Vertex AI (Gemini 3.5 Flash) + FreeLLMAPI (3-tier hybrid routing)
- **Search**: SearXNG (Docker, primary) + DuckDuckGo (fallback)
- **Vector DB & Embeddings**: Qdrant (in-memory) + `sentence-transformers/all-mpnet-base-v2` (768-d)
- **UI & Entry**: Streamlit + fallback CLI runner
- **Packaging**: Python 3.11+, `uv` package manager

---

## 3. Project File Layout Pointers
- **System Specification**: [docs/spec.md](file:///c:/Users/beste/Documents/antigravity/deep-research/docs/spec.md)
- **System Architectures & Blueprints**: [agentic_deep_research_master_blueprint.md](file:///c:/Users/beste/Documents/antigravity/deep-research/agentic_deep_research_master_blueprint.md)
- **Phase 2 Task Checklist**: [docs/tasks/phase_2_tasks.md](file:///c:/Users/beste/Documents/antigravity/deep-research/docs/tasks/phase_2_tasks.md)
- **Phase 3 Task Checklist (Draft)**: [docs/tasks/phase_3_tasks.md](file:///c:/Users/beste/Documents/antigravity/deep-research/docs/tasks/phase_3_tasks.md)
- **Workspace Rules (Atomic Loop + Phase Completion)**: [.agents/AGENTS.md](file:///c:/Users/beste/Documents/antigravity/deep-research/.agents/AGENTS.md)

---

## 4. Key Engineering Constraints
1. **No Over-engineering**: Keep implementation as minimal and simple as possible (Ponytail mode). Rely on the standard library where appropriate.
2. **Async-First**: All network requests (HTTP client, search, scrapers, LLM invocation) must use `async`/`await`.
3. **Strict Schemas**: All data exchanges across agent nodes must be validated using Pydantic models.
4. **Structured Logging**: Log every transition, query, scrape, and validation step via `structlog` for UI stream visibility.
5. **No Key Commits**: Use GCP Application Default Credentials (ADC) or `.env` file templates. Do not commit secrets.
6. **Git Hygiene & Workflow**: Git is a mandatory, automatic part of your workflow. Commit every successful slice/task immediately (Save Point Pattern). Never accumulate large uncommitted changes. Push all commits to GitHub (`origin main`) before ending your turn. Use conventional commit messages (`feat`, `fix`, `refactor`, `chore`, `docs`, `test`). Ensure pre-commit checks and tests pass before committing.

---

## 5. Active Development State
- **Active Phase**: Phase 2 (Search & Research) — COMPLETE ✅
- **Active Task**: Phase 2 Critical Evaluation and Phase 3 Task List Review
- **Task List**: [docs/tasks/phase_2_tasks.md](file:///c:/Users/beste/Documents/antigravity/deep-research/docs/tasks/phase_2_tasks.md)
- **Phase 1 Status**: ✅ 100% complete, bug-fixed, fully tested (25/25 passing tests), committed and pushed.
- **Phase 2 Status**: ✅ 100% complete, bug-fixed, fully tested (65/65 passing tests), committed, and pushed.
- **Phase 3 Status**: Draft task list exists at `docs/tasks/phase_3_tasks.md`. Crucial review and planning are ready for the Phase 3 conversation.
- **Workflow Reminder**: At the end of Phase 2, follow the Phase Completion Rules (critical eval + stop — do NOT start Phase 3 code changes).
- **Known Blockers**: None.
