# Live web visualizer for smoovie

**Date:** 2026-06-22
**Status:** Design approved, ready for implementation plan
**Scope:** Replace `smoovie`'s PNG+ffmpeg movie rendering with a built-in live web
visualizer: a terminal Holoscan sink operator taps the named float32 HEALPix maps off the
last compute stage and hands them to a single FastAPI+uvicorn app that serves both the
three.js renderer (HTTP) and the frame stream (`/stream` WebSocket) on one port. After the
pipeline finishes, the session freezes for scrub-back inspection. Prototyped in
`~/software/scratch/kremetart-server-demo/{server_v1.py,renderer_v1.html}`; this is an
integration of that wire protocol and UX, not a redesign of it.

## 1. Motivation and context

`smoovie` today (`src/kremetart/core/smoovie.py`) is a four-operator Holoscan app:

```
reader (HealpixZarrReaderOperator) → imager (HealpixDFTOperator)
    → iwp (IWPKalmanOperator) → writer (HealpixWriterOperator)
```

`IWPKalmanOperator` — the **last compute stage** — emits exactly three named `(1, npix)`
float32 HEALPix maps per sub-integration plus `time_out`: `cube` (dirty), `filtered`, and
`znorm`. The writer regions them into a durable `(TIME, PIX)` zarr; the host then renders
the maps to PNGs (matplotlib Mollweide) and encodes three mp4s with ffmpeg.

That PNG+ffmpeg path is removed entirely. In its place a new terminal sink operator fans
out from `iwp` (alongside the retained writer) and feeds a web layer, so a user runs
`kremetart smoovie ... --port 8080`, opens one URL, and watches the live sky on an
interactive sphere — the three demo outputs mapped to the names `raw` / `smooth` / `znorm`.

## 2. Decisions (locked)

| Axis | Decision |
|---|---|
| Serve by default | **`--serve/--no-serve`, default `True`.** Headless/cab runs pass `--no-serve`. |
| Durable artifact | **Always write `<output>.zarr`** (both modes). The web sink is an *additional* consumer of `iwp`, not a replacement for the writer. |
| Sink tap point | Fan-out from `iwp` via Holoscan broadcast: `iwp → writer` **and** `iwp → web_sink`. |
| Name mapping | `cube → "raw"`, `filtered → "smooth"`, `znorm → "znorm"` (matches the demo `NAMES`). |
| Holder | **`LatestFrameHolder`** — `threading.Lock`, one latest-wins slot per name, shared `seq`, `finished` flag. The compute callback never awaits/sends → no backpressure on the GXF scheduler. |
| Server | One **FastAPI + uvicorn** app on one port: `GET /` (renderer) + `/static/*` (vendored three.js) + `WS /stream`. Runs in a daemon thread. |
| Wire protocol | **Preserved** from the demo (`geometry` once per name, then `frame` header + binary float32 pairs). Two **additive** control messages: `tracks` and `end`. |
| three.js | **Vendored** into `static/vendor/` — no CDN at runtime (may run air-gapped). |
| WS URL | Derived client-side from `window.location` (`${proto}//${location.host}/stream`). |
| Frozen inspection | On `app.run()` return → `holder.finish()` → server sends `{"type":"end"}` and closes; client freezes (PAUSED, slider parked at newest cached seq). `core.smoovie` keeps serving until Ctrl-C. Session-scoped; survives pipeline end, **not** a page reload. No disk persistence. |
| Reconnect | Client `onclose` reconnects **only if no clean `end` was received** (clean end = freeze; unexpected drop = retry). |
| Satellite overlay | Ported to the sphere: `--overlay-catalog` (+ `catalog_*`) computes tracks host-side and the server sends a `tracks` control message; the renderer draws 3D trails + current markers. |
| znorm coloring | Single sequential ramp (demo behavior) with **symmetric** per-frame `vmin/vmax` (`−max\|·\|..+max\|·\|`) so the innovation reads zero-centered — within the existing protocol, no client redesign. |
| New deps | `fastapi`, `uvicorn[standard]` added to `[full]`. `matplotlib` **stays** in `[full]` — `utils/visualisation.py` still imports it (see below). |
| Mollweide renderer | The PNG/ffmpeg helpers (`render_frames`, `_overlay_tracks`, `_encode_movie`) have been **moved** to `utils/visualisation.py`, retained dormant for a future Mollweide-rendering sub-command. **Out of scope here** — this work neither uses nor modifies them. |
| Container serve | `--serve` is a **native** (full-install) feature. Container-dispatched serving would need port-publishing in `run_in_container`; documented as out of scope. |

## 3. The sink ↔ holder ↔ server boundary

```
GXF scheduler thread                  shared object              server thread (asyncio)
─────────────────────                 ─────────────              ───────────────────────
IWPKalmanOperator
  └─ cube, filtered, znorm, time_out
        ▼
WebStreamSinkOperator.compute():
  receive 3 tensors + time            LatestFrameHolder          per-connection task:
  cp.asnumpy() if GPU-resident   ───► (threading.Lock,    ────►  poll holder (~30 ms);
  vmin/vmax per name (znorm sym)        one slot/name,            for each name whose seq
  holder.put(name, seq, t, bytes)       shared seq,               advanced, send header+
  return immediately                    finished flag)            binary; on finished,
                                                                  flush + send {"type":"end"}
```

**Never-block guarantee.** `compute()` does the device→host copy (its own work), computes
`vmin/vmax`, and `holder.put()` swaps the per-name slot under a lock held only for the
swap. Stored payloads are immutable `bytes`/`numpy`, so the server reads and sends them
after releasing the lock. The callback never touches the socket and never awaits → a slow
or disconnected browser cannot backpressure the scheduler. Latest-wins means a slow client
simply misses intermediate frames; at TART's ~1 fps a local browser receives every frame.

**Static metadata path (not through the holder).** Geometry (from `nside`/`nest`) and the
optional satellite tracks are computed host-side and passed to the server at construction.
The holder carries only live frames + the finish signal.

## 4. Module layout

Layer rules (`architecture.md` §1): the operator lives in `operators/`, host glue in
`utils/`, and the `SmooviePipeline` app stays in `core/smoovie.py` (it is the command's
wiring). `operators/web_sink.py` imports the holder from `utils/` — the same direction
`operators/iwp_kalman.py` already imports `utils/iwp.py`. Dependencies flow
`cli → core → {utils, operators}`; nothing new imports from `core/`.

### `src/kremetart/utils/healpix_viz.py` (host, no GPU, no web framework)

Pure host helpers — public names, importable by both the operator and the server:

- `class LatestFrameHolder` — `put(name, seq, t, vmin, vmax, data: bytes)`,
  `snapshot() -> dict[name, slot]`, `current_seq`, `finish()`, `finished`. Thread-safe via
  `threading.Lock`. One latest-wins slot per name.
- `geometry_message(name, nside, nest) -> dict` — `hp.boundaries`-derived corners
  (`(npix,4,3)` float32, reshaped to a flat list), matching the demo's `geometry` payload.
- `tracks_payload(tracks, names_optional) -> dict` — converts `satellite_tracks` output
  (`name → [(frame_index, ra_deg, dec_deg, flux_jy), ...]`) into
  `{"type":"tracks","sats":[{name, points:[{seq, xyz:[x,y,z], flux}]}]}` using
  `hp.ang2vec` (ICRS ra/dec → unit vector, same frame as the pixel boundaries; ordering-
  independent).

Imports `numpy`/`healpy` at module top (host deps, no GPU, no fastapi).

### `src/kremetart/utils/web_server.py` (host, FastAPI/uvicorn)

- `class FrameServer` — wraps the FastAPI app + a `uvicorn.Server` on a daemon thread.
  Constructed with `(holder, *, nside, nest, names, port, tracks=None, host="0.0.0.0")`.
  `start()` launches the thread and returns the view URL; `stop()` sets
  `server.should_exit`.
- FastAPI routes: `GET /` → `static/index.html`; `StaticFiles` mount at `/static`;
  `WS /stream` → handler that (1) sends `geometry` per name, (2) sends `tracks` if present,
  (3) loops: poll holder, send advanced frames as `frame` header + binary pairs, until
  `holder.finished` and all names flushed, then sends `{"type":"end"}` and returns (clean
  close). Per-connection `last_sent` seq tracking; uvicorn websocket configured so large
  geometry/frame messages are not truncated.

Imports `fastapi`/`uvicorn` at module top (both in `[full]`; this module is only imported
by `core.smoovie`, which already requires the full install).

### `src/kremetart/operators/web_sink.py` (GPU-adjacent operator)

```python
class WebStreamSinkOperator(Operator):
    def __init__(self, fragment, *args, holder, **kwargs): ...
    def setup(self, spec):
        spec.input("raw"); spec.input("smooth"); spec.input("znorm"); spec.input("time_out")
    def compute(self, op_input, op_output, context):
        # receive each map; cp.asnumpy() iff GPU-resident; float32 little-endian bytes
        # vmin/vmax per name (znorm: symmetric); holder.put(name, seq, t, vmin, vmax, data)
        # increment shared seq once per integration; return immediately (no emit)
```

Terminal sink (no outputs). `cupy` imported at module top like the other GPU operators;
falls back to `numpy` view if a tensor is already host-resident. The `iwp` port names map
to public names in the flow wiring (`("cube","raw")`, `("filtered","smooth")`,
`("znorm","znorm")`), not inside the operator.

### `src/kremetart/static/`

- `index.html` — the renderer, adapted from `renderer_v1.html` (see §6).
- `vendor/three.module.min.js` — vendored three.js (the demo's 0.160.x module build).

## 5. App, core, and CLI changes — `core/smoovie.py`

- **`SmooviePipeline`** gains an optional `holder=None`. `compose()` keeps
  `iwp → writer`; when `holder is not None` it also adds `WebStreamSinkOperator` and wires
  `iwp → web_sink` on `{("cube","raw"), ("filtered","smooth"), ("znorm","znorm"),
  ("time_out","time_out")}` (Holoscan auto-broadcasts the shared `iwp` outputs to both
  receivers).
- **`image_via_app(...)`** gains `holder=None`, forwarded to `SmooviePipeline`. It still
  writes the durable `output_zarr`; with a holder, frames also stream live during
  `app.run()`. Nothing renders the maps anymore, so it **no longer eager-loads them** —
  the post-run `xr.open_zarr(...).values` block is dropped and the function returns the
  `output` path. Existing host tests that stub `image_via_app` are updated for the new
  signature/return.
- **`smoovie()`** orchestration:
  1. Validate; resolve `hdf_paths`, phase direction; fail-fast if `<output>.zarr` exists
     and not `overwrite`.
  2. If `overlay_catalog`: compute `tracks` host-side (`satellite_tracks`, network) **before**
     the run, consistent with `nframes`.
  3. If `serve`: build `LatestFrameHolder(names=["raw","smooth","znorm"])`; start
     `FrameServer(holder, nside=nside, nest=nest, names=..., port=port, tracks=tracks)`;
     print the view URL; `webbrowser.open(url)` if `open_browser`.
  4. `image_via_app(..., holder=holder if serve else None)`.
  5. If `serve`: `holder.finish()` (→ server emits `end`), print "serving frozen session
     at <url> (Ctrl-C to exit)", block on an interruptible wait, then `server.stop()`.
  6. Return `output`.
- **Removed from `core/smoovie.py`**: the matplotlib/ffmpeg imports, the
  `shutil.which("ffmpeg")` check, and the render/encode orchestration (`movie_specs`,
  the `render_frames`/`_encode_movie` calls, the `diverging`/`cmap`/`fps` path). The
  helper *functions themselves* are **not deleted** — they already live in
  `utils/visualisation.py` (moved there for a future Mollweide sub-command, out of scope).
  `core/smoovie.py` simply stops importing/calling them; it does **not** import
  `utils/visualisation.py`.

### `cli/smoovie.py` and the cab

Mirror rule (`architecture.md` §1): `core.smoovie`'s params equal `cli.smoovie`'s minus the
`StimelaMeta(skip=True)` flags. Changes (both files, regenerated cab):

- **Add** `serve: bool = True`, `port: int = 8080`, `open_browser: bool = False`. These are
  **ordinary cab inputs** (not `skip=True`) — `core` needs them, so the mirror rule forbids
  skipping. Recipe authors set `serve: false` for batch.
- **Replace** the `movie: File` input and `@stimela_output(dtype="File", name="movie")`
  with `output: Directory` and `@stimela_output(dtype="Directory", name="output")` (the
  durable `<output>.zarr`).
- **Remove** render-only `fps`, `cmap`. **Keep** `overlay_catalog`,
  `catalog_elevation_deg`, `catalog_cache` (now feed the web overlay), `phase_ra_deg`,
  `phase_dec_deg`, `correct_gains`, `profile`, `iwp_sigma`, `iwp_noise`, `overwrite`,
  `nframes`, `nside`.
- The wrapper stays in hip-cargo's round-trippable shape; the cab regenerates via the
  pre-commit hook. `--backend`/`--always-pull-images` remain `skip=True`.

## 6. Renderer changes (`static/index.html`)

Adapted from `renderer_v1.html`. **Preserved unchanged**: per-name ring buffers, the
shared-seq rewind slider, the tunable cache + live MB estimate, the recolor-ms HUD,
drag-rotate / scroll-zoom, the inline colormap. Edits:

1. **Importmap** `three` → `/static/vendor/three.module.min.js` (no CDN).
2. **WS URL** derived from `window.location`:
   `const proto = location.protocol === "https:" ? "wss:" : "ws:";`
   `const WS_URL = `${proto}//${location.host}/stream`;` (replaces the hardcoded address).
3. **`{"type":"end"}` handling** — set `finished = true`, switch to PAUSED, park the slider
   at the newest cached seq for the current name (immediate scrub-back).
4. **Reconnect gate** — `ws.onclose` calls `connect()` again **only if `!finished`**; a
   clean `end` freezes (no reconnect), an unexpected drop still retries.
5. **Satellite overlay** — on `{"type":"tracks"}`, store per-sat `{seq, xyz, flux}` arrays;
   in `showSeq`, draw a `THREE.Line` trail (points with `seq ≤ current`) + a marker at the
   current seq per satellite, positioned just outside the unit sphere. Overlay is global
   (drawn over whichever name is selected), matching the current movie behavior.

## 7. Packaging

- Add `fastapi` and `uvicorn[standard]` to `[project.optional-dependencies].full`. **Keep
  `matplotlib`** — `utils/visualisation.py` (the parked Mollweide renderer) imports it.
- Vendor `three.module.min.js` under `src/kremetart/static/vendor/`.
- Confirm `uv_build` ships non-`.py` files under the package; if not, add the matching
  `[tool.uv.build-backend]` data/include configuration so `static/**` is packaged. The
  server resolves `index.html` via `importlib.resources` (not a source-tree-relative path).

## 8. Testing

- **`tests/test_structure.py`** — update the `smoovie` signature-mirror expectation for the
  new params and the `output` rename. `healpix_viz.py`, `web_server.py`, `web_sink.py` are
  not commands, so they need no `cli`/`core`/`cab` triple.
- **`tests/test_roundtrip.py`** — keep the `smoovie` round-trip green for the new signature
  (output dtype `Directory`, new bool/int inputs).
- **`tests/test_web_viz.py` (new, CPU, no GPU, no real GPU tensors):**
  - `LatestFrameHolder`: concurrent `put`/`snapshot` correctness, latest-wins per name,
    `seq` monotonicity, `finish()`/`finished` semantics.
  - `geometry_message` shape/dtype for a small `nside`; `tracks_payload` xyz matches
    `hp.ang2vec` and seq alignment.
  - `WebStreamSinkOperator.compute` writes the holder given numpy-stubbed inputs (no
    `cupy`/GXF): the operator's host path is exercised via a fake `op_input`.
  - `FrameServer` smoke test (if practical): start on an ephemeral port, connect a client,
    assert `geometry` → `frame` → `end` ordering and the no-reconnect-on-clean-end contract;
    otherwise assert the handler's message sequence against a fake websocket.
- **Existing host tests that stub `image_via_app`** run with `serve=False` (no server, no
  block) and are updated for its new `holder=` signature.

## 9. Explicitly out of scope

Container-dispatched serving (port publishing in `run_in_container`); disk persistence of
sessions (frozen inspection is session-scoped and lost on reload); multi-client fan-out
tuning beyond the simple per-connection poll; server-side history/ring buffers (the client
owns history); per-name colormap selection or a configurable LUT; authentication / TLS on
the served port; and any change to the imaging, IWP filter, or zarr-writer numerics.

## 10. Affected files (summary)

| File | Change |
|---|---|
| `src/kremetart/utils/healpix_viz.py` | **New** — `LatestFrameHolder`, `geometry_message`, `tracks_payload`. |
| `src/kremetart/utils/web_server.py` | **New** — `FrameServer` (FastAPI app + uvicorn thread, `/`, `/static`, `/stream`). |
| `src/kremetart/operators/web_sink.py` | **New** — `WebStreamSinkOperator` terminal sink → holder. |
| `src/kremetart/static/index.html` | **New** — renderer (vendored three.js, location-derived WS, `end`/reconnect gate, sat overlay). |
| `src/kremetart/static/vendor/three.module.min.js` | **New** — vendored three.js. |
| `src/kremetart/core/smoovie.py` | Fan-out sink + serve orchestration + frozen inspection; stop importing/calling the PNG/ffmpeg helpers. |
| `src/kremetart/cli/smoovie.py` | `+serve/port/open_browser`; `movie:File`→`output:Directory`; drop `fps`/`cmap`. |
| `src/kremetart/utils/visualisation.py` | **Unchanged** — parked Mollweide renderer for a future sub-command (out of scope; keeps `matplotlib` in `[full]`). |
| `src/kremetart/cabs/smoovie.yml` | Auto-regenerated (pre-commit). |
| `pyproject.toml` | `+fastapi`, `+uvicorn[standard]` in `[full]`; keep `matplotlib`; package `static/**`. |
| `tests/test_structure.py`, `tests/test_roundtrip.py` | Follow the new `smoovie` signature/output. |
| `tests/test_web_viz.py` | **New** — holder/geometry/tracks/sink/server unit tests. |
