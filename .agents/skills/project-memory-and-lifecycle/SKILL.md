---
name: project-memory-and-lifecycle
description: Establish, maintain, and execute an atomic development loop and persistent memory system (using GEMINI.md and AGENTS.md) for any software project.
---

# Project Memory & Lifecycle Management

Use this skill to initialize or maintain an ultra-disciplined, zero-regression development loop and cross-session persistent memory system in any codebase.

---

## 1. Initialization (Scaffolding a New Project)

When starting a project or introducing this memory system to an existing codebase, scaffold the memory directory immediately:

1. Create a `.agents/` folder in the root of the project.
2. Create `.agents/GEMINI.md` using the template below to track project state.
3. Create `.agents/AGENTS.md` using the template below to enforce behavior constraints.
4. Create a task tracking directory (e.g., `docs/tasks/`) containing markdown checklists for project milestones.

### GEMINI.md Template
```markdown
# GEMINI.md: Project Context & Memory

This context is loaded automatically at the start of every session. Use it to keep track of state, goals, and coding rules across sessions.

---

## 1. Project Goal & Identity
[Describe the core goal and identity of the project]

---

## 2. Core Technology Stack
- **Languages & Runtimes**: [e.g., Python 3.11]
- **Frameworks**: [e.g., FastAPI, Next.js]
- **Databases & Utilities**: [e.g., SQLite, Pydantic]
- **Packaging & Package Manager**: [e.g., poetry, uv]

---

## 3. Project File Layout Pointers
- **System Specification**: [Link to spec]
- **Workspace Rules**: [.agents/AGENTS.md](file:///.agents/AGENTS.md)
- **Active Task Checklist**: [Link to current task list]

---

## 4. Key Engineering Constraints
1. **YAGNI (Keep It Simple)**: Avoid speculative generalizations or unnecessary external dependencies.
2. **Standard Library First**: Prefer native runtime features before reaching for third-party packages.
3. **Structured Logging**: Enforce structured, observable logging for all transaction layers.
4. **Git Save-Point Pattern**: Stage and commit every single granular task as soon as it passes verification.
5. **No Secret Commits**: Never commit credentials, API keys, or raw configurations.

---

## 5. Active Development State
- **Active Phase**: [e.g., Phase 1: Initialization]
- **Active Task**: [e.g., Task 1.1: Core Scaffolding]
- **Task List**: [Link to active task list]
- **Milestone Progress**:
  - Phase 1: ⏳ Not Started
  - Phase 2: ⏳ Not Started

### Next Session Handoff Prompt
\`\`\`markdown
/goal [Brief task summary]
Read \`.agents/GEMINI.md\` to load the complete project context. Pick up on Task [X.X] by implementing...
\`\`\`
```

### AGENTS.md Template
```markdown
# Workspace Rules: Atomic Task Execution Loop

You MUST strictly follow this multi-step loop for EVERY task in your active checklist:

1. **Design & Plan**: Inspect current files, explain your implementation strategy, and address design details.
2. **Implement**: Write clean, minimal code modifications.
3. **Audit & Review**: Subject changes to self-critique or code quality review (style, safety, security).
4. **Test**: Write corresponding unit/integration tests with thorough edge-case coverage.
5. **Verify**: Run the tests. Pass rate must be 100%. If any test fails, fix it immediately.
6. **Commit**: Stage and commit files immediately (\`git add .\` and \`git commit -m "feat/fix: <description>"\`).
7. **Sync Memory**: Update task checklists and \`.agents/GEMINI.md\` to reflect the completed state.
8. **Push**: Push the commit to origin main to keep the remote up-to-date.

Do NOT combine multiple checklist tasks into a single commit or proceed to a new task until all 8 steps of the current loop are fully finished and verified.

---

## Phase Completion Rules
When ALL tasks in the active phase checklist are complete:
1. **Critical Evaluation**: Audit all modified files against the specification to find regressions, debt, or bloat.
2. **Next-Phase Task Review**: Verify the sequencing and clarity of the next phase's checklist.
3. **Write the Evaluation Report**: Summarize phase status, quality assessment, and next-phase amendments for the user.
4. **STOP — Do NOT begin the next phase**: After delivering the report, stop. The next phase must be executed in a fresh, clean conversation to avoid context degradation.
5. **Update Memory**: Update \`.agents/GEMINI.md\` to mark the current phase complete and set up the handoff prompt.

---

## Persistent Technical Lessons (Avoid Repeats)
Keep a running ledger of technical traps, framework bugs, or environment issues discovered during development to prevent wasting time on duplicate debugging loops in future sessions:
- [Lesson 1]: ...
- [Lesson 2]: ...
```

---

## 2. Execution (Operating the System)

For every turn, check if the project has `.agents/GEMINI.md` and `.agents/AGENTS.md`. If it does:

1. **Load Context**: Immediately read `.agents/GEMINI.md` to grasp the active phase, technology stack, constraints, and current task.
2. **Execute Loop**: Perform the 8-step atomic loop on the designated task.
3. **Maintain Memory**: At the end of your turn, update `.agents/GEMINI.md`'s Active Development State and compile the exact Next Session Handoff Prompt.
4. **Record Lessons**: If you encounter a complex framework quirk, runtime error, or UI pitfall, immediately document it under `Persistent Technical Lessons` in `.agents/AGENTS.md` before concluding the task.

---

## 3. Benefits & Yield
* **Zero Amnesia**: Seamless handoff between agents and conversations.
* **Deterministic Execution**: The 8-step loop guarantees that every commit in the git tree represents a buildable, 100% verified, and fully tested state.
* **Self-Improving Memory**: Documenting platform lessons directly in the workspace prevents agents from repeating the same debugging loops across conversations.
