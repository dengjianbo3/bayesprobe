# Task 4 Report: WebUI Streaming Architecture and Verification

Status: DONE_WITH_CONCERNS

Commit: `5c649e7 docs: record WebUI streaming milestone`

## Scope

Committed only:

- `docs/ARCHITECTURE.md`
- `docs/superpowers/specs/2026-07-10-webui-streaming-progress-design.md`

No implementation code or tests were modified. This report is intentionally
uncommitted for the controller.

## Commands and Output

```text
$ git rev-parse HEAD
1faa2c74155bb30170ecb9bc96394c80d9c0419e

$ pytest tests/test_question_runner.py tests/test_webui.py tests/test_public_api_and_config.py -q
........................................................................ [ 78%]
....................                                                     [100%]
92 passed in 5.40s

$ node --test tests/test_webui_stream.js
✔ consumes split NDJSON chunks and unlocks a completed stream (6.926625ms)
✔ cancels and unlocks a stream after a malformed progress event (1.089708ms)
✔ rejects incomplete streams after unlocking at EOF (1.071084ms)
✔ preserves an integrated cycle when a sanitized terminal failure arrives (0.966791ms)
✔ handleSubmit preserves integrated output after a terminal stream failure (1.020708ms)
✔ handleSubmit marks malformed stream progress as failed (0.5855ms)
tests 6
suites 0
pass 6
fail 0
cancelled 0
skipped 0
todo 0
duration_ms 94.654334

$ pytest -q
........................................................................ [ 18%]
........................................................................ [ 37%]
........................................................................ [ 56%]
............................ss.......................................... [ 75%]
........................................................................ [ 94%]
...................                                                      [100%]
377 passed, 2 skipped in 5.67s

$ kill -TERM 76654
exit=0

$ python3 -m bayesprobe.webui --host 127.0.0.1 --port 8766
server left running; PID 73391

$ curl -sS -D - -o /dev/null http://127.0.0.1:8766/
HTTP/1.0 200 OK
Server: BayesProbeWebUI/0.1 Python/3.14.3
Content-Type: text/html; charset=utf-8
Cache-Control: no-store
Content-Length: 6471

$ git diff --check
exit=0; no output

$ git grep -n -e 'test-key-webui-m0-10-20260710' -e 'delayed-fake-test-key-20260710' || true
exit=0; no output

$ git status --short
 M docs/ARCHITECTURE.md
 M docs/superpowers/specs/2026-07-10-webui-streaming-progress-design.md

$ git add docs/ARCHITECTURE.md docs/superpowers/specs/2026-07-10-webui-streaming-progress-design.md
exit=0

$ git commit -m "docs: record WebUI streaming milestone"
[main 5c649e7] docs: record WebUI streaming milestone
 2 files changed, 42 insertions(+), 3 deletions(-)
```

The Node suite is `tests/test_webui_stream.js`; this repository has no
frontend `package.json` or npm test script.

## Browser Verification

Browser tooling was available through the Codex in-app browser.

- Local WebUI: entered a non-secret test value in Chat Completions, switched
  to Deterministic, ran the A-E graph question with one cycle, and switched
  back to Chat Completions. The API-key field still held the test value.
- The deterministic run rendered `Run started`, `Belief initialized`,
  `Probes planned`, `Executing probes`, `Integrating evidence`, `Posterior
  updated`, and `Run completed`. It reported `Completed`, best answer `D`,
  posterior mass `1.000`, and an `integrated` cycle.
- Reloading the local WebUI cleared the API-key field.
- Test-only delayed-provider server: started on `127.0.0.1:8767` with an
  inline `DelayedFakeChatOpenAI` passed to
  `create_handler_class(client_factory=DelayedFakeChatOpenAI)`. At 250 ms,
  the browser reported `Running`, `Executing probes | Cycle 1`, and a disabled
  run button. After 4.5 s it reported `Completed`, an integrated cycle trace,
  the full phase list, and an enabled run button. The test-only server was
  stopped with Ctrl-C.
- Desktop layout measurement (`1280 x 720`): document width `1265`, client
  width `1265`, and progress overlap count `0`.
- Mobile layout measurement (`390 x 844`): document width `375`, client width
  `375`, and progress overlap count `0`. The temporary viewport was reset.

## Server State

- `http://127.0.0.1:8766` is running the requested BayesProbe WebUI process
  (PID `73391`) and returned `200 OK` after the documentation commit.
- The test-only delayed fake server on port `8767` has been stopped.
- No push was attempted.

## Concerns for Controller

1. Populated long trace JSON does not remain constrained inside its scrollable
   block. On desktop, `#trace-pane` was `865px` wide with `scrollWidth 5924`;
   each `pre` was also `5893px` wide despite `overflow-x: auto`. On mobile,
   `#trace-pane` was `343px` wide with the same `5924px` scroll width. The
   document itself did not horizontally overflow, but the trace containment
   acceptance condition is not satisfied.
2. The controller should perform the reserved whole-branch review, final
   browser verification, and push. Re-run the trace containment check after
   the implementation follow-up above.

## Trace Containment Fix

Status: DONE_WITH_CONCERNS

Fix commit: `b8591e5eb9bba87e423993c39c42defc944167c6`

The trace containment chain now applies `min-width: 0` and bounded width to
`#trace-pane`, each `.trace-item`, `.trace-stack`, each
`.trace-stack > section`, and each `pre`. Each `pre` also has `width: 100%`,
so long JSON keeps its horizontal overflow inside the individual trace item.
JSON formatting and the existing visual styles are unchanged.

Focused verification:

```text
$ pytest tests/test_webui.py -q
....................................                                     [100%]
36 passed in 5.36s

$ git diff --check
exit=0; no output
```

Browser measurements after the fix, with populated deterministic trace data:

- Desktop `1280 x 720`: `#trace-pane` client width `865`, scroll width `865`.
  All 14 `pre` elements were `836px` wide with `836px` parent client widths;
  long JSON retained larger internal `scrollWidth` values up to `2915px`.
- Mobile `390 x 844`: post-fix measurement could not be completed because the
  in-app browser blocked the local URL during the reload/read step under its
  URL security policy. No alternate browser surface was used to bypass that
  restriction.

No push was attempted.
