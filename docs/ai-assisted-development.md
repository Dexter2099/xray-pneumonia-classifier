# AI-assisted development and the engineering harness

This project was built and improved with AI-assisted coding. Codex contributed implementation, tests, integration, automated source review and verification commands. Local Qwen produced bounded drafts for image handling, the desktop window layout, the verification runner, input-inspection snapshots and readability font roles, which were inspected and changed before integration. Dexter set the interview goal, scope and acceptance scenarios. This is not a claim that Dexter manually authored or personally reviewed every line.

The reusable repository harness was added after the initial viewer implementation to codify the constraints and checks already being used. It should not be presented as a framework that existed from the first commit, a fully autonomous development system, or proof that an agent's decisions are always correct.

## What constitutes the harness

| Part | What it contributes |
|---|---|
| `AGENTS.md` | Repository context, model invariants, lifecycle rules and delivery boundaries. |
| `.agents/skills/xray-maintenance/SKILL.md` | A discoverable procedure for changing and verifying this application. |
| `docs/architecture.md` | The responsibilities and contracts the implementation should preserve. |
| `tests/fixtures/trained_baseline.json` | Original outputs captured before refactoring, tied to input/checkpoint hashes. |
| Focused tests | Observable invalid-input, numerical, lifecycle, recovery and responsiveness behavior. |
| `scripts/check.py` | A reproducible, fail-fast verification entry point with a machine-readable result. |
| GitHub Actions and PRs | The same portable checks on pushed changes plus an inspectable commit/diff history. |

Instructions guide the coding model. The tests and executable checks provide evidence about the resulting program. Neither replaces checking whether the requirements or scientific assumptions are right. `AGENTS.md` and a `SKILL.md` package follow the documented repository-instruction and reusable-skill conventions: [OpenAI guidance](https://developers.openai.com/blog/skills-agents-sdk).

## Repeatable commands

Use the project interpreter from the repository root:

```powershell
# Portable checks, suitable for CI; no trained checkpoint required.
.venv\Scripts\python.exe -m scripts.check

# Include the original checkpoint regression and actual desktop workflow.
.venv\Scripts\python.exe -m scripts.check --trained

# Include numerical-checked performance measurements and the dependency audit.
.venv\Scripts\python.exe -m scripts.check --release
```

The command stops on the first failed check, returns a non-zero status, and records the commit, working-tree state, executed checks and unrun checks in `outputs/verification/checks.json`. It does not repair code, install packages, update golden results, commit, push or merge. The release profile includes the trained checks, benchmark and dependency audit; those require the trusted local artifacts and network access for the advisory database. It is a repository verification harness around the coding agent, not an API wrapper that runs a model autonomously.

## A concrete change example

The original prediction helper reloaded the checkpoint for every image. Before changing it, the original trained outputs were captured, including a borderline prediction. The reusable service was then introduced and tested for one model load, repeated predictions, image/tensor equivalence and single/batch agreement. A benchmark compared both algorithms using the same runtime and input, and refused to report success if probabilities disagreed.

The window layout draft also required correction during review: clearing a QLabel's text after assigning its pixmap cleared the image, and the offscreen Windows renderer needed a font for readable screenshots. Source review and an actual rendered run caught problems that a plausible-looking code draft alone did not establish were absent.

For the optional input inspector, local Qwen drafted a small snapshot module. Review corrected an unsupported histogram argument combination, a sample-standard-deviation calculation labelled as population standard deviation, and an unbounded source preview. Known-pixel tests verify the corrected statistics and display reconstruction. The inference tensor and historical scientific fixtures were not changed to accommodate diagnostics.

## Interview wording

"I used AI-assisted coding to implement the project, with local Qwen helping on bounded drafts. I defined the scope and the scenarios I wanted to demonstrate. The repository now has agent instructions, a focused skill, regression fixtures and a verification command. For example, the model-reuse refactor was checked against outputs captured from the original code, and performance measurements were accepted only when the predictions agreed. I can show the changes, explain the design decisions and distinguish automated checks from what I have personally tested."

Before the interview, rehearse the native window and file dialog, inspect the relevant diffs, and practice explaining or modifying the input validation, service lifetime and worker boundary. Those are human acceptance and understanding tasks; an automated pass should not be described as completing them on Dexter's behalf.
