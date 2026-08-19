---
name: grill-me
description: Relentlessly interview the user one decision at a time to stress-test a plan, architecture, experiment, or product direction before implementation. Use when the user invokes grill-me, asks to be grilled, or wants a design tree resolved through questions.
license: MIT
metadata:
  source: "https://mcpmarket.com/tools/skills/grill-me"
---

# Grill Me

Turn an incomplete plan into explicit shared understanding. Resolve the design
tree depth-first, one decision at a time. Do not begin implementation during the
interview.

## Start

1. Read the plan and inspect relevant repository files, tests, Git state, and
   documentation before asking anything the environment can answer.
2. Identify three to eight top-level decision branches and order them by
   dependency. Briefly show this map and name which branch must be resolved first.
3. Ask exactly one decision question. Include a concrete recommended answer and
   the reason it is preferred. Wait for the user's answer.

## Interview discipline

- Ask one question per turn. Do not hide multiple decisions in subclauses.
- Facts are the agent's responsibility: inspect local evidence or authoritative
  sources instead of asking the user to retrieve them.
- Decisions belong to the user. Offer a recommendation, alternatives, and the
  material tradeoff, but do not silently select one.
- Probe vague terms such as "accurate," "organic," "secure," "real data," or
  "production ready" until they have measurable definitions.
- Resolve prerequisites before dependent choices. When an answer opens a new
  branch, insert it at the correct dependency point.
- Reconcile contradictions immediately by citing the conflicting decisions and
  asking which one governs.
- After resolving a branch, restate its decision before proceeding.
- If the user says `skip`, mark only that decision deferred and explain what it
  blocks. If the user says `stop`, return a partial shared-understanding summary.
- Do not mutate code, files, Git, external systems, or datasets until the user
  confirms the interview is complete and separately asks for implementation.

## Project-specific evidence

When grilling this repository's AI code-provenance work, inspect
`README.md`, `docs/CODE_PROVENANCE_RESEARCH_DESIGN.md`, the corpus manifest
contract, and the implementation before asking. Common branches include:

- intended user and decision supported;
- admissible human, AI, hybrid, and unknown labels;
- generator, language, repository, time, and duplicate leakage controls;
- meaning and presentation of the organic-code index;
- public-reuse corpus provenance and licensing;
- calibration, OOD abstention, and acceptable error costs;
- security-tool validation and package-registry uncertainty;
- repository, commit, PR, and adaptive-review product surfaces.

Use only the branches relevant to the user's topic; this is not a mandatory
checklist.

## Completion gate

When no unresolved branch remains, summarize:

- resolved decisions;
- confirmed assumptions and measurable success criteria;
- constraints and rejected alternatives;
- named deferrals and their consequences;
- remaining risks;
- the recommended next action.

Ask the user to confirm that this is the shared understanding. Do not act on it
until they confirm.
