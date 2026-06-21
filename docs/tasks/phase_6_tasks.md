# Phase 6 Checklist: Self-Improvement

Use this task list to draft and implement Phase 6. Mark tasks as completed `[x]` as you finish them. Do not implement code beyond Phase 6.

---

## Step 0: Python-native Skill Registry

### [ ] Task 6.0: Python-native Skill Registry Scaffolding
- **Description**: Implement a dynamic Python-native skill registry (`skills/registry.py`) that registers, catalogs, and loads custom skill modules.
- **Acceptance Criteria**:
  - Dynamically load Python modules from the `skills/` directory.
  - Expose a decorator `@register_skill` with description, inputs, and output schemas.
  - Include basic skills for math, string transformations, and report styling.
- **Files**:
  - **`skills/registry.py`** [NEW]
  - **`skills/__init__.py`** [NEW]

---

## Step 1: Sandbox & Isolation

### [ ] Task 6.1: Explore Lane Sandbox
- **Description**: Create an isolated execution sandbox (`skills/sandbox.py`) using `multiprocessing` or clean sub-processes to execute and benchmark newly generated skills without side-effects.
- **Acceptance Criteria**:
  - Intercept print statements and system exceptions.
  - Timeout execution after 5 seconds to prevent infinite loops.
  - Restrict imports or run in a restricted environment if possible.
- **Files**:
  - **`skills/sandbox.py`** [NEW]

---

## Step 2: Automated Skill Generation Loop

### [ ] Task 6.2: Deficit Identification & Remediation
- **Description**: Connect DREAM evaluator failures with an LLM generator that drafts new custom Python-native skills to solve specific deficiencies.
- **Acceptance Criteria**:
  - Parse DREAM scores and evaluator notes to detect specific deficits (e.g. specialized mathematical calculations, units conversion, report table formatting).
  - Draft clean, PEP-8 compliant Python code for a new skill using the `@register_skill` decorator.
- **Files**:
  - **`skills/generator.py`** [NEW]

---

## Step 3: Loop Verification & Tests

### [ ] Task 6.3: Sandbox Test-Runner and Self-Improvement Integration
- **Description**: Implement a closed-loop system where generated skills are run in the sandbox against a suite of auto-generated assertion checks. If they pass, they are permanently written to `skills/` and registered.
- **Acceptance Criteria**:
  - Run dynamic pytest-like checks in the sandbox.
  - If verified, commit the new skill to the registry.
- **Files**:
  - **`skills/self_improvement.py`** [NEW]
  - **`tests/unit/test_self_improvement.py`** [NEW]
