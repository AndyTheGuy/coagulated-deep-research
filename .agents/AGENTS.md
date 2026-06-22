# Workspace Rules: Atomic Task Execution Loop

You MUST strictly follow this multi-step loop for EVERY task in the task checklist (e.g. `docs/tasks/phase_2_tasks.md`):

1. **Design & Plan**: Inspect current files, explain implementation strategy, and address design details.
2. **Implement**: Write the code modifications.
3. **Audit & Review**: Subject changes to a self-critique or subagent-driven code quality audit, ensuring async-first compliance and no regressions.
4. **Test**: Write corresponding unit/integration tests with comprehensive edge-case coverage.
5. **Verify**: Run the tests. The pass rate must be 100%. If any test fails, fix it immediately.
6. **Commit**: Stage and commit the files immediately (e.g. `git add .` and `git commit -m "feat/fix: <description>"`) following the Git Save-Point Pattern.
7. **Sync Memory**: Update task checklists (`docs/tasks/phase_2_tasks.md` and the local task list artifact `task.md`) and `.agents/GEMINI.md` to reflect the completed state.
8. **Push**: Push the commit to origin main (`git push origin main`) to ensure git remote is fully up-to-date.

Do NOT combine multiple checklist tasks into a single commit or proceed to a new task until all 8 steps of the current task loop are fully finished and verified.

---

## Phase Completion Rules

These rules apply at the end of **every phase** (Phase 2, Phase 3, Phase 4, and all subsequent phases):

When ALL tasks in the active phase checklist are complete:

1. **Critical Evaluation**: Perform a thorough critical evaluation of every change made during the phase:
   - Review every file modified against the spec (`docs/spec.md`).
   - Identify any regressions, design gaps, over-engineering, or missing requirements.
   - List specific issues with file references and line numbers.

2. **Next-Phase Task Review**: Open the next phase task file (e.g. `docs/tasks/phase_N+1_tasks.md`) and evaluate it:
   - Confirm the tasks are correctly sequenced and properly scoped.
   - Identify any tasks that are missing, incorrectly specified, or too coarse-grained.
   - Propose concrete amendments with justification.
   - If no next-phase task file exists yet, create a draft one based on `docs/spec.md` before evaluating it.

3. **Write the Evaluation Report**: Write a concise, structured evaluation report to the user summarising:
   - Phase completion status (what was done, what was skipped, any known debt).
   - Phase quality assessment (bugs found, design issues, test coverage gaps).
   - Proposed amendments to the next-phase task list.

4. **STOP — Do NOT begin the next phase**: After delivering the evaluation report, stop. The next phase MUST be implemented in a fresh, clean conversation. Do not write any production code for the next phase. This rule is non-negotiable and applies to every phase.

5. **Update Memory**: Update `.agents/GEMINI.md` to reflect the current phase as complete, the next phase as upcoming, and the active task pointer as ready for the next conversation.

---

## Subagent Fleet Strategy

To maximize development throughput while maintaining absolute correctness, utilize parallelized subagents within the 8-step atomic loop:

1. **Defining the Fleet**: At the start of the session, define these specialized subagents using `define_subagent`:
   - `tester`: Specialized in writing exhaustive unit/integration tests (using `pytest` + `pytest-asyncio`). Equipped with write tools.
   - `auditor`: Specialized in async-first compliance, PEP 8, security, and dry-run code analysis. Equipped with read tools.
   - `archivist`: Specialized in updating task checklists, `task.md`, and persistent memory files. Equipped with write tools.

2. **Parallel Loop Execution**:
   - **Step 2 (Implement) & Step 4 (Test)**: While you implement production code, invoke `tester` to draft comprehensive test cases in parallel.
   - **Step 3 (Audit & Review)**: Invoke `auditor` to conduct an adversarial code quality and async-safety review.
   - **Step 7 (Sync Memory)**: Invoke `archivist` to synchronize memory checklists (`task.md`, phase checklists, `GEMINI.md`) while you stage and commit.

3. **Active Waiting & Time Optimization**:
   - Whenever a subagent is delegated a task that leaves the main agent waiting, the main agent MUST look for high-value tasks/design work to continue in parallel.
   - To strictly avoid disrupting the subagent's active environment or violating the sequential checklist constraint, the main agent should draft upcoming task implementations (e.g. MCTS engines, routing algorithms) inside the persistent `<appDataDir>\brain\<conversation-id>/scratch/` directory.
   - These isolated drafts can then be copied, customized, and integrated instantly once the active task is finished, verified, and committed, achieving maximum throughput safely.

---

## Technical & Testing Lessons (Persistent Memory)

To prevent long debugging loops, silent crashes, or false-positive green test runs in future interactive UI integrations, always abide by these rules:

1. **Beware the "Selector Automation Sandbox" (UI Automation Pitfall)**:
   - **The Lesson**: Automated subagents using browser devtools or raw browser execution (CDP/Puppeteer) are fragile at triggering React or Streamlit state reruns (e.g. setting `.value` on DOM elements does not update underlying framework component states). Subagents can easily get stuck in a "selector-not-found" or "click-did-not-trigger" loop rather than debugging actual runtime logic.
   - **The Rule**: If automated browser testing struggles with complex reactive state widgets for more than 15 minutes, stop the automation. Have the human partner manually perform the UI clicks in their browser while you monitor log outputs and inspect file syntax top-to-bottom.

2. **Streamlit Lifecycle awareness (Linear Executions & Namespaces)**:
   - **The Lesson**: Streamlit parses and executes scripts linearly from top to bottom on every user interaction. Standard Python unit tests run backend logic directly and will *never* catch `NameError` placeholder exceptions arising from Streamlit's rendering order.
   - **The Rule**: When streaming log/stats outputs to UI containers/placeholders (`st.empty()`), NEVER run the execution logic inline in the middle of the script. Set session state triggers and defer all execution blocks (`run_async_safely`) to the absolute bottom of the script, ensuring all placeholders are fully rendered and registered in the active Python namespace first. Pass placeholders as arguments to the executors to preserve dynamic layout rendering placements.

3. **Validate Event-Loop / Thread Cleanup**:
   - **The Lesson**: Storing event-loop-bound singletons in module-level global variables causes severe threading and event-loop deadlocks inside interactive, multi-threaded frameworks like Streamlit.
   - **The Rule**: Always isolate event-loop-bound caches using task-local context containers (`contextvars.ContextVar`) rather than simple global/module variables. This ensures thread-safety, loop safety, and clean test isolation.

