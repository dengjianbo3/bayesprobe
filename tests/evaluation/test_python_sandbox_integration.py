import os
from pathlib import Path

import pytest

from bayesprobe.evaluation.python_probe import (
    DockerPythonSandbox,
    DockerPythonSandboxConfig,
    PythonExecutionRequest,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("BAYESPROBE_RUN_DOCKER_TESTS") != "1",
    reason="set BAYESPROBE_RUN_DOCKER_TESTS=1 to run Docker isolation tests",
)


def make_request(sandbox, code: str, *, execution_id: str):
    return PythonExecutionRequest(
        execution_id=execution_id,
        run_id="integration_run",
        cycle_id="cycle_1",
        probe_id="probe_1",
        code=code,
        image=sandbox.preflight(),
    )


def test_real_sandbox_executes_deterministic_math():
    sandbox = DockerPythonSandbox()
    request = make_request(
        sandbox,
        "import sympy as sp\nprint(sp.factor(84))",
        execution_id="math",
    )

    first = sandbox.execute(request)
    second = sandbox.execute(request)

    assert first.success is True
    assert first.stdout == "84\n"
    assert second.stdout == first.stdout


def test_real_sandbox_cannot_reach_network():
    sandbox = DockerPythonSandbox()
    record = sandbox.execute(
        make_request(
            sandbox,
            (
                "import socket\n"
                "try:\n"
                "    socket.create_connection(('1.1.1.1', 53), timeout=1)\n"
                "    print('network-open')\n"
                "except OSError:\n"
                "    print('network-blocked')\n"
            ),
            execution_id="network",
        )
    )

    assert record.stdout == "network-blocked\n"


def test_real_sandbox_cannot_read_host_file(tmp_path: Path):
    host_secret = tmp_path / "host-secret.txt"
    host_secret.write_text("host-secret-value", encoding="utf-8")
    sandbox = DockerPythonSandbox()
    record = sandbox.execute(
        make_request(
            sandbox,
            (
                "from pathlib import Path\n"
                f"print(Path({str(host_secret)!r}).exists())\n"
            ),
            execution_id="host-file",
        )
    )

    assert record.stdout == "False\n"


def test_real_sandbox_does_not_inherit_host_secrets(monkeypatch):
    monkeypatch.setenv("BAYESPROBE_HOST_SECRET", "must-not-leak")
    sandbox = DockerPythonSandbox()
    record = sandbox.execute(
        make_request(
            sandbox,
            "import os\nprint(os.getenv('BAYESPROBE_HOST_SECRET'))",
            execution_id="environment",
        )
    )

    assert record.stdout == "None\n"


def test_real_sandbox_timeout_removes_container():
    sandbox = DockerPythonSandbox(
        DockerPythonSandboxConfig(timeout_seconds=0.5)
    )
    record = sandbox.execute(
        make_request(
            sandbox,
            "import time\ntime.sleep(10)\nprint('late')",
            execution_id="timeout",
        )
    )

    assert record.timed_out is True
    assert record.success is False


def test_real_sandbox_caps_combined_output():
    sandbox = DockerPythonSandbox(
        DockerPythonSandboxConfig(max_output_bytes=1024)
    )
    record = sandbox.execute(
        make_request(
            sandbox,
            "print('x' * 10000)",
            execution_id="output-cap",
        )
    )

    assert record.output_truncated is True
    assert len(record.stdout.encode("utf-8")) + len(record.stderr.encode("utf-8")) <= 1024


def test_real_sandbox_enforces_process_limit():
    sandbox = DockerPythonSandbox()
    record = sandbox.execute(
        make_request(
            sandbox,
            (
                "import os, time\n"
                "children = []\n"
                "blocked = False\n"
                "try:\n"
                "    for _ in range(100):\n"
                "        pid = os.fork()\n"
                "        if pid == 0:\n"
                "            time.sleep(0.2)\n"
                "            os._exit(0)\n"
                "        children.append(pid)\n"
                "except OSError:\n"
                "    blocked = True\n"
                "for pid in children:\n"
                "    os.waitpid(pid, 0)\n"
                "print(blocked, len(children))\n"
            ),
            execution_id="pids-limit",
        )
    )

    blocked, child_count = record.stdout.strip().split()
    assert blocked == "True"
    assert int(child_count) < 64
