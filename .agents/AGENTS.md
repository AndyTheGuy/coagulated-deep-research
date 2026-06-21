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
