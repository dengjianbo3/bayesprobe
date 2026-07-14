# Terminal-Bench Task 8 Report

## Scope

Implemented the Harbor 0.18 BaseAgent entry point without changing the
BayesProbe kernel, benchmark configuration, or unrelated reports.

Owned files changed:

- `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/agent.py`
- `benchmarks/terminal_bench/tests/test_agent.py`
- `.superpowers/sdd/terminal-bench-task-8-report.md`

## RED Evidence

The initial focused test run failed during collection before the production
module existed:

```text
ModuleNotFoundError: No module named 'bayesprobe_terminal_bench.agent'
```

Command:

```bash
uv run --project benchmarks/terminal_bench pytest benchmarks/terminal_bench/tests/test_agent.py -q
```

A second focused RED test verified the installed Harbor 0.18 contract mismatch:
`BaseAgent.context_id` is a `UUID`, while `build_live_session` takes a string.
The test failed because the UUID was forwarded unchanged, then passed after the
agent converted only that value at the adapter boundary.

## Implementation

- `BayesProbeHarborAgent` is a real `harbor.agents.base.BaseAgent` with the
  Harbor 0.18 async `setup` and `run` signatures, `name() == "bayesprobe"`,
  and the nested package version.
- `setup` deliberately performs no environment access, installation, or
  mutation.
- `run` resolves settings through `TerminalBenchConfig.from_sources(self.extra_env)`
  without changing `os.environ`, captures Harbor's active event loop, builds one
  live session, and invokes `runner.run_question` exactly once through
  `asyncio.to_thread`.
- On a completed run only, the agent preserves mapping-valued Harbor metadata
  and adds bounded scalar run identity, stop reason, cycle, action, and model
  call counts. Nonmapping metadata is replaced safely.
- The summary receives only the agent-owned metadata. The API key is redacted
  from agent-generated text and propagated ordinary exceptions; no failure
  result, metadata update, or summary is fabricated for config, session, runner,
  or cancellation failures.
- Cancellation is allowed to propagate. The worker can finish its one already
  started call, but the cancelled coroutine does not update metadata or write a
  summary afterward.

## Verification

```text
uv run --project benchmarks/terminal_bench pytest benchmarks/terminal_bench/tests/test_agent.py -q
7 passed in 0.04s

uv run --project benchmarks/terminal_bench pytest benchmarks/terminal_bench/tests/test_agent.py benchmarks/terminal_bench/tests/test_public_reuse.py -q
13 passed in 0.05s

uv run --project benchmarks/terminal_bench pytest benchmarks/terminal_bench/tests -q
187 passed in 6.81s
```

The Harbor 0.18 import/signature check with the nested source path supplied
reported:

```text
harbor=0.18.0
bayesprobe_terminal_bench.agent:BayesProbeHarborAgent
bayesprobe
0.1.0
(self, environment: 'BaseEnvironment') -> 'None'
(self, instruction: 'str', environment: 'BaseEnvironment', context: 'AgentContext') -> 'None'
True
```

`python -m compileall` completed for `agent.py`.

## Blocker

The brief's unmodified standalone import command fails:

```bash
uv run --project benchmarks/terminal_bench python -c 'from bayesprobe_terminal_bench.agent import BayesProbeHarborAgent; print(BayesProbeHarborAgent.import_path())'
```

It exits with `ModuleNotFoundError` because `benchmarks/terminal_bench/src` is
not on the direct `uv run python` import path. Pytest succeeds because its
configuration sets `pythonpath = ["src", "scripts"]`. The equivalent import
with `PYTHONPATH=benchmarks/terminal_bench/src` succeeds and prints the expected
Harbor import path above. Fixing the direct command requires a nested-project
packaging or configuration change outside this task's owned files, so none was
made.
