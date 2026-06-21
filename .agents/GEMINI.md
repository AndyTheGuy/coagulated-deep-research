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
- **Phase 2 Walkthrough (Persistent Memory)**: [.agents/PHASE_2_WALKTHROUGH.md](file:///c:/Users/beste/Documents/antigravity/deep-research/.agents/PHASE_2_WALKTHROUGH.md)

---

## 4. Key Engineering Constraints
1. **No Over-engineering**: Keep implementation as minimal and simple as possible (Ponytail mode). Rely on the standard library where appropriate.
2. **Async-First**: All network requests (HTTP client, search, scrapers, LLM invocation) must use `async`/`await`.
3. **Strict Schemas**: All data exchanges across agent nodes must be validated using Pydantic models.
4. **Structured Logging**: Log every transition, query, scrape, and validation step via `structlog` for UI stream visibility.
5. **No Key Commits**: Use GCP Application Default Credentials (ADC) or `.env` file templates. Do not commit secrets.
6. **Git Hygiene & Workflow**: Git is a mandatory, automatic part of your workflow. Commit every successful slice/task immediately (Save Point Pattern). Never accumulate large uncommitted changes. Push all commits to GitHub (`origin main`) before ending your turn. Use conventional commit messages (`feat`, `fix`, `refactor`, `chore`, `docs`, `test`). Ensure pre-commit checks and tests pass before committing.
7. **Skill Suite Optimization**: For all future development, actively use the `/ponytail` skill suite to guarantee the simplest, most elegant, and standard-library-first implementations (strictly avoiding over-engineering).
8. **Bug Auditing & Quality Verification**: Regularly invoke `/find-skills` or `/local-skill-finder` to locate specialized tools to audit code for bugs, implement robust unit/integration test suites, and enforce premium software design.

---

## 5. Active Development State
- **Active Phase**: Phase 5 (Human-in-the-Loop & Streaming UI) — ACTIVE 🔍
- **Active Task**: Debug Streamlit event-loop deadlock/freeze during parallel research streaming
- **Task List**: [docs/tasks/phase_5_tasks.md](file:///c:/Users/beste/Documents/antigravity/deep-research/docs/tasks/phase_5_tasks.md)
- **Phase 1 Status**: ✅ 100% complete, bug-fixed, fully tested (25/25 passing tests), committed and pushed.
- **Phase 2 Status**: ✅ 100% complete, bug-fixed, fully tested (65/65 passing tests), committed, and pushed.
- **Phase 3 Status**: ✅ 100% complete, bug-fixed, fully tested (all passing), committed, and pushed.
- **Phase 4 Status**: ✅ 100% complete, bug-fixed, fully tested (135/135 passing tests), committed, and pushed.
- **Phase 5 Status**: 🧪 Completed features; 155/155 passing tests; actively resolving Streamlit event-loop deadlock during graph stream execution.
- **Workflow Reminder**: Follow the 8-step loop for every single task.
- **Known Blockers**: Threading conflict / event-loop mismatch in `LLMRouter` singletons when driven by Streamlit's script execution threads.

### Phase 5 UI Debug Handoff Prompt
```markdown
/goal Debug and resolve the Streamlit execution freeze in our LangGraph agentic pipeline.

Read `.agents/GEMINI.md` to load the complete project context. The compiled LangGraph StateGraph workflow runs and streams perfectly to 100% completion inside a standard python CLI process (run `uv run python scratch/run_graph_test.py` to verify). However, inside the Streamlit web interface (`ui/app.py` running on `http://localhost:8501`), when clicking "Start Deep Research", the graph executes the first node (`scoping_ambiguity_check`) and then completely freezes or terminates without proceeding to the next node (`write_research_brief`). Fix this issue by implementing thread-safe router context or running the asyncio event loop safely outside Streamlit's script-runner thread. Confirm 100% success with browser testing and ensure no regressions (155/155 passing tests).
```

