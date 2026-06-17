# Architectural Rules & Domain Logic

Read this when editing `src/kremetart/**` files.

## 1. Package Layout

```
kremetart/
├── src/kremetart/
│   ├── __init__.py
│   ├── _container_image.py    # CONTAINER_IMAGE — single source of truth for the image tag
│   ├── cli/                   # Lightweight Typer wrappers. THIS is what generate-cabs parses.
│   │   ├── __init__.py        # Builds the Typer `app` and registers subcommands
│   │   └── onboard.py         # One file per subcommand (delete onboard once setup is done)
│   ├── core/                  # Real implementations. Heavy deps live here.
│   │   ├── __init__.py
│   │   └── onboard.py         # Mirrors cli/onboard.py — same function name, no decorators
│   └── cabs/                  # AUTO-GENERATED Stimela YAMLs. Do NOT hand-edit.
│       ├── __init__.py
│       └── onboard.yml
├── tests/
│   ├── test_install.py
│   └── test_roundtrip.py      # Guards the CLI → cab → CLI round-trip
├── Dockerfile                 # Builds the image referenced by _container_image.py
├── pyproject.toml
├── tbump.toml                 # Release tooling — updates _container_image.py + cabs
├── .pre-commit-config.yaml    # Runs generate-cabs on every commit
└── .github/workflows/
    ├── ci.yml
    ├── publish.yml             # PyPI on tag push
    ├── publish-container.yml   # ghcr.io on tag + every push to main
    └── update-cabs.yml         # Regenerates cabs on merge to main
```

### Role of each directory

| Directory | What lives there | What does NOT live there |
|---|---|---|
| `cli/` | Thin Typer wrappers with `@stimela_cab` (and optional `@stimela_output`). One file per command. **Imports from `core/` must be lazy** (inside the function body). | Heavy imports at module top. Business logic. NumPy / pandas / domain libs. |
| `core/` | The actual implementation. Type-hinted function with the same name as the CLI wrapper, but **no Typer / hip-cargo decorators**. Free to import anything. | Typer. `@stimela_cab`. UI concerns. `typer.Exit(...)`. |
| `cabs/` | Generated `<command>.yml` files. Committed to source control. Loaded by Stimela. | Anything you wrote by hand. Drift from `cli/*.py`. |

### `core/` is one module per command (enforced)

`cli/`, `core/` and `cabs/` are in **strict 1:1:1 correspondence**: for every command `<name>`
there is exactly `cli/<name>.py`, `core/<name>.py`, and `cabs/<name>.yml`, all sharing the
command's name. This correspondence *is* the round trip — do not weaken it.

- **`core/` holds commands only.** A `core/<name>.py` that has no `cli/<name>.py` mirror is a bug.
  Shared / host helper logic goes in `utils/`; Holoscan operators **and the apps that wire them**
  go in `operators/`. A `core/<cmd>.py` imports from `utils/` and `operators/` (heavy deps at module
  scope are fine in all three) — it never grows sibling `core/<cmd>_*.py` helper or app modules.
  (`core/stream_msv4.py` holds a Holoscan app *because `stream_msv4` is itself a command*, not
  because apps live in `core/`.)
- **`core.<cmd>` must mirror `cli.<cmd>`'s signature.** Its parameters equal the cli wrapper's
  parameters minus the `StimelaMeta(skip=True)` flags (`backend`, `always_pull_images`). **Never add
  a parameter to `core/<cmd>.py` that is not on the cli wrapper.** A core-only parameter is dead
  surface — unreachable from the CLI or a Stimela cab, only from a direct Python call — and almost
  always means runtime/environment branching (e.g. GPU-vs-CPU imaging) that should be **auto-detected
  internally**, with tests forcing a path by monkeypatching the detector, not by a new argument. If a
  knob genuinely needs to be user-facing, add it to `cli/<cmd>.py` (and let the cab regenerate),
  never to `core/` alone.
- **Dependencies flow one way: `cli/` → `core/` → {`utils/`, `operators/`}.** `utils/` and
  `operators/` must never import from `core/`. A helper defined in a `core/<cmd>.py` and imported by
  another module — another command, or anything in `utils/` / `operators/` — is misplaced: promote it
  to `utils/` under a public (non-underscore) name (see `python-standards.md` §6). `core/<cmd>.py`
  keeps only helpers private to that one command, and (per `python-standards.md` §2) imports its
  dependencies at module top.

`tests/test_structure.py` enforces the directory bijection and the signature mirror;
`tests/test_roundtrip.py` enforces the `cli` ↔ `cab` half. If you are extending an existing command
with substantial new machinery, that machinery lands in `utils/` / `operators/`, and `core/<cmd>.py`
orchestrates it.

### Adding a new command

1. Create `src/kremetart/cli/<name>.py` with a `@stimela_cab`-decorated
   Typer function. Lazily import the core implementation inside the function.
2. Create `src/kremetart/core/<name>.py` with the real implementation —
   same function name, no decorators, free to import heavy deps.
3. Register the new command in `src/kremetart/cli/__init__.py` (next to
   the existing `onboard` registration; mirror its pattern).
4. Commit. The pre-commit hook regenerates `src/kremetart/cabs/<name>.yml`
   automatically.

**Never** create files under `cabs/` by hand. They are derived artefacts.

## 2. Lightweight vs Full Installation

This package supports two install modes. The split is what makes the
container-fallback pattern below work.

| Mode | Command | What it pulls | When to use |
|---|---|---|---|
| **Lightweight** | `pip install kremetart` | `hip-cargo` + `typer` only | Cab consumers (Stimela), CI machines that only need to dispatch commands into containers, anyone who already has the project's container image available. |
| **Full** | `pip install kremetart[full]` | Lightweight + everything listed under `[project.optional-dependencies].full` in `pyproject.toml` | Local development; native (non-container) execution. |

The lightweight install is **always sufficient to invoke any command** because
the generated CLI wrappers fall back to running the same command inside the
project's container when native imports fail (see §3).

### When you add a heavy dep

- Add it to `[project.optional-dependencies].full` in `pyproject.toml`, **not**
  to the top-level `dependencies`. The top-level list must stay tiny so the
  lightweight install remains lightweight.
- Import it **only from inside `core/`**. Never import it from `cli/` at module
  scope.

## 3. Container Fallback & Backends

Every generated CLI wrapper in `cli/*.py` follows the same shape (this is
emitted by `hip-cargo generate-function`, but the pattern matters when you
write a new command by hand too):

```python
def my_command(...):
    if backend == "native" or backend == "auto":
        try:
            from hip_cargo.utils.runner import preflight_remote_must_exist
            preflight_remote_must_exist(my_command, dict(...))
            from kremetart.core.my_command import my_command as my_command_core
            my_command_core(...)
            return
        except ImportError:
            if backend == "native":
                raise
    # Heavy deps missing OR backend explicitly chose a container → run in container.
    from hip_cargo.utils.config import get_container_image
    from hip_cargo.utils.runner import run_in_container
    image = get_container_image("kremetart")
    run_in_container(my_command, dict(...), image=image, backend=backend, ...)
```

### How `--backend` flows

Every command auto-grows two parameters via `hip-cargo generate-function`:

| Flag | Values | Effect |
|---|---|---|
| `--backend` | `auto` (default), `native`, `apptainer`, `singularity`, `docker`, `podman` | `auto` tries native then falls back to a detected container runtime. `native` forces in-process execution and surfaces the `ImportError` if `[full]` is not installed. The explicit backends skip the native attempt entirely and dispatch into the matching runtime. |
| `--always-pull-images` | bool | Forces a fresh `pull` before each container run. |

Both flags are decorated with `StimelaMeta(skip=True)` so they appear in the
Python CLI but **not** in the generated cab YAML — Stimela manages container
execution on its own side and doesn't need them.

### Image resolution

The image tag is owned by `src/kremetart/_container_image.py`:

```python
CONTAINER_IMAGE = "ghcr.io/landmanbester/kremetart:latest"
```

Three things keep this in sync — do not bypass them:

1. **Feature branches:** Edit `_container_image.py` by hand to point at your
   branch tag (e.g. `:my-feature`). The `publish-container.yml` workflow builds
   and pushes that tag on every push of the PR.
2. **Merge to `main`:** The `update-cabs.yml` workflow resets the
   tag to `latest` and regenerates cabs in a `[skip checks]` commit.
3. **Releases:** `tbump <version>` rewrites the tag to the semantic version and
   regenerates cabs as a `before_commit` hook.

### Remote URIs (S3 / GCS / Azure)

Path-typed parameters (`File`, `Directory`, `MS`, `URI`) accept both local
paths and remote URIs (`s3://...`, `gs://...`, `az://...`). When the path is
remote:

- `_resolve_mounts` skips it (nothing to bind-mount).
- `preflight_remote_must_exist` checks existence via fsspec.
- `run_in_container` forwards the matching credentials (`AWS_*`, `~/.aws`,
  `GOOGLE_APPLICATION_CREDENTIALS`, `~/.config/gcloud`, `AZURE_*`, `~/.azure`).

Users who want native remote access install the right extra: `pip install
hip-cargo[s3]`, `[gcs]`, or `[azure]`. Without it, the wrapper's existing
`try/except ImportError` routes them into the container, which already has the
backends.

## 4. Cab Generation is Automatic

**The `src/kremetart/cabs/*.yml` files are generated artefacts. Never edit
them by hand and never run `hip-cargo generate-cabs` manually.**

Three automated paths keep them in sync with `cli/*.py`:

1. **Pre-commit hook** (`.pre-commit-config.yaml`): on every commit, runs
   `hip-cargo generate-cabs --module src/kremetart/cli/*.py --output-dir
   src/kremetart/cabs`. If it modifies files, pre-commit will "fail" the
   commit — re-run `git add -u && git commit` to include the updates.
2. **`update-cabs.yml` workflow**: on merge to `main`, resets the
   container tag to `latest` and regenerates cabs in a `[skip checks]` commit.
3. **`tbump`**: on release, rewrites the container tag to the version and
   regenerates cabs.

If you ever see a cab YAML in a diff that wasn't generated by one of these
three paths, that's a bug — revert it and edit the corresponding `cli/*.py`
instead.

> **Heads-up:** `generate-cabs` resolves the `image:` field from the *installed*
> package metadata. Activate the project venv (so `kremetart` is importable)
> before committing — otherwise the regenerated cab is written without an
> `image:` field.

### How CLI source maps to cab YAML

- `@stimela_cab(name=..., info=...)` → the cab's name and top-level info.
- `@stimela_output(...)` → entries under `outputs:` in the cab.
- Each Typer parameter → an entry under `inputs:` (dtype inferred from the type
  hint, `info` from `help=`, defaults from `= ...`).
- `Annotated[..., StimelaMeta(skip=True)]` → omitted from the cab (used for
  `--backend`, `--always-pull-images`, etc.).
- `Annotated[..., StimelaMeta(metadata={"rich_help_panel": "Inputs", "tunable":
  True})]` → flows into the cab's `metadata:` dict.
- Inline comments after `Annotated[...]` rows are preserved through the round
  trip — they show up as `# noqa: ...` or similar on the matching cab field.
