# CLAUDE.md

kremetart images TART snapshots into a stream of HEALPix sky maps and flags transients with a per-pixel
Kalman filter. The package installs a single command, `kremetart` (no subcommands); helper scripts live in
`scripts/`, outside the package. The per-snapshot maths lives in runtime-agnostic **stages** (`src/kremetart/imaging/`);
a **runner** drives them: Holoscan on a GPU, an in-process NumPy loop on the CPU, or a CyberEther
flowgraph (`--runner cyberether`, opt-in while the port in #47 matures). Linux, macOS (Apple Silicon) and
Windows are supported; Holoscan and CUDA are Linux-only.

Domain vocabulary is in `GLOSSARY.md`; settled decisions are in `docs/adr/`; the derivations the code cites
are in `docs/design/`.

| Rule file | Read it when editing |
|---|---|
| `.claude/rules/architecture.md` | `src/kremetart/**`: package layout, dependency direction, stages vs adapters, packaging and releases. |
| `.claude/rules/python-standards.md` | any `**/*.py`: type hints, import placement, Typer options, private names. |
| `.claude/rules/testing-and-ci.md` | `tests/**` or `.github/workflows/**`: test surfaces, GPU and parity tests, dev workflow, commits. |

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues for landmanbester/kremetart (via the `gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` and `docs/adr/` at the repo root. See `docs/agents/domain.md`.
