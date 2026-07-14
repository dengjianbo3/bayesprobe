from pathlib import Path

import pytest

from bayesprobe.cli import main
from bayesprobe.evaluation import cli as evaluation_cli


@pytest.mark.parametrize(
    ("argv", "function_name", "argument"),
    [
        (
            ["eval", "prepare", "--config", "pilot.json"],
            "prepare_capability_experiment",
            Path("pilot.json"),
        ),
        (
            ["eval", "run", "--config", "pilot.json"],
            "run_capability_experiment",
            Path("pilot.json"),
        ),
        (
            ["eval", "score", "--experiment", "restricted/experiment"],
            "score_capability_experiment",
            Path("restricted/experiment"),
        ),
        (
            ["eval", "report", "--experiment", "restricted/experiment"],
            "report_capability_experiment",
            Path("restricted/experiment"),
        ),
        (
            [
                "eval",
                "checkpoint-prepare",
                "--config",
                "pilot.json",
                "--source-experiment",
                "restricted/source",
            ],
            "prepare_paradigm_checkpoint",
            (Path("pilot.json"), Path("restricted/source")),
        ),
        (
            ["eval", "checkpoint-run", "--experiment", "restricted/checkpoint"],
            "run_paradigm_checkpoint",
            Path("restricted/checkpoint"),
        ),
        (
            ["eval", "checkpoint-score", "--experiment", "restricted/checkpoint"],
            "score_paradigm_checkpoint",
            Path("restricted/checkpoint"),
        ),
    ],
)
def test_eval_cli_dispatches_four_phase_commands(
    monkeypatch,
    capsys,
    argv,
    function_name,
    argument,
):
    calls = []

    def fake_handler(*paths):
        calls.append(paths[0] if len(paths) == 1 else paths)
        return f"completed {function_name}"

    monkeypatch.setattr(evaluation_cli, function_name, fake_handler)

    exit_code = main(argv)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert calls == [argument]
    assert f"completed {function_name}" in captured.out
    assert captured.err == ""


@pytest.mark.parametrize("command", ["prepare", "run"])
def test_eval_cli_requires_config_for_prepare_and_run(command, capsys):
    exit_code = main(["eval", command])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--config" in captured.err


@pytest.mark.parametrize("command", ["score", "report"])
def test_eval_cli_requires_experiment_for_score_and_report(command, capsys):
    exit_code = main(["eval", command])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "--experiment" in captured.err


def test_eval_cli_returns_one_with_sanitized_error(monkeypatch, capsys):
    def fail(path):
        raise RuntimeError("provider rejected sk-super-secret-value")

    monkeypatch.setattr(evaluation_cli, "run_capability_experiment", fail)

    exit_code = main(["eval", "run", "--config", "pilot.json"])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "provider rejected" in captured.err
    assert "sk-super-secret-value" not in captured.err
    assert "<redacted>" in captured.err


def test_eval_cli_does_not_expose_restricted_key_error_value(monkeypatch, capsys):
    def fail(path):
        raise KeyError("restricted-sample-id")

    monkeypatch.setattr(evaluation_cli, "score_paradigm_checkpoint", fail)

    exit_code = main(
        ["eval", "checkpoint-score", "--experiment", "restricted/checkpoint"]
    )

    captured = capsys.readouterr()
    assert exit_code == 1
    assert "restricted-sample-id" not in captured.err
    assert "invalid or incomplete checkpoint artifact" in captured.err
