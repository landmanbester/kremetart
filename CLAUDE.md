# CLAUDE.md — Project Context

This project was bootstrapped with [`hip-cargo init`](https://github.com/landmanbester/hip-cargo).
It is a **hip-cargo package**: a Python CLI whose commands are decorated so that
Stimela cab definitions are generated automatically from the CLI source. Cabs
let the same commands be invoked from Stimela recipes and from `kremetart`
on the command line interchangeably.

When working in this repo, treat the patterns in the rules below as load-bearing
— they are what makes the round-trip between CLI source, generated cabs, and
container fallback work. If you find yourself wanting to deviate, stop and check
[hip-cargo's own docs](https://github.com/landmanbester/hip-cargo) first.

*Note: Detailed architecture/domain logic, Python standards, and testing/CI
rules have been modularized into the `.claude/rules/` directory for progressive
disclosure. Read the relevant file before editing the matching files.*

| Rule file | Read it when editing |
|---|---|
| `.claude/rules/architecture.md` | `src/kremetart/**` — package layout, install modes, container fallback, cab generation. |
| `.claude/rules/python-standards.md` | any `**/*.py` — type hints, lazy imports, Typer syntax, hip-cargo types. |
| `.claude/rules/testing-and-ci.md` | `tests/**` or `.github/workflows/**` — round-trip tests, dev workflow, commits. |

---

## Where to Go Deeper

- hip-cargo source & docs: <https://github.com/landmanbester/hip-cargo>
- Stimela: <https://github.com/caracal-pipeline/stimela>
- Twelve-factor principles guide most architectural decisions in this repo:
  <https://12factor.net/>
