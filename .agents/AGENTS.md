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
