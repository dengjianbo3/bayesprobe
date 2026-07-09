import json
import subprocess
import sys
import tomllib
from pathlib import Path

from bayesprobe.cli import main


FIXTURE_PATH = Path("fixtures/benchmarks/toy_belief_revision.json")


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_experiment_config(config_path: Path) -> tuple[Path, Path]:
    report_path = config_path.parent / "outputs" / "report.json"
    ledger_path = config_path.parent / "outputs" / "ledger.jsonl"
    write_json(
        config_path,
        {
            "dataset_path": str(FIXTURE_PATH.resolve()),
            "report_path": "outputs/report.json",
            "ledger_path": "outputs/ledger.jsonl",
        },
    )
    return report_path, ledger_path


def test_cli_run_writes_report_ledger_and_prints_summary(tmp_path: Path, capsys):
    config_path = tmp_path / "experiment.json"
    report_path, ledger_path = write_experiment_config(config_path)

    exit_code = main(["run", "--config", str(config_path)])

    captured = capsys.readouterr()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert captured.err == ""
    assert "BayesProbe experiment complete" in captured.out
    assert "dataset=toy_belief_revision" in captured.out
    assert "samples=3" in captured.out
    assert "final_accuracy=1.0" in captured.out
    assert "update_direction_accuracy=1.0" in captured.out
    assert f"report={report_path}" in captured.out
    assert f"ledger={ledger_path}" in captured.out
    assert report["sample_count"] == 3
    assert ledger_path.exists()


def test_cli_run_prints_artifact_summary_when_enabled(tmp_path: Path, capsys):
    config_path = tmp_path / "experiment.json"
    report_path = config_path.parent / "outputs" / "report.json"
    artifact_dir = config_path.parent / "artifacts" / "toy-run"
    write_json(
        config_path,
        {
            "dataset_path": str(FIXTURE_PATH.resolve()),
            "report_path": "outputs/report.json",
            "artifact_dir": "artifacts/toy-run",
        },
    )

    exit_code = main(["run", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == ""
    assert f"report={report_path}" in captured.out
    assert f"ledger={artifact_dir / 'ledger.jsonl'}" in captured.out
    assert f"artifact={artifact_dir}" in captured.out
    assert (artifact_dir / "manifest.json").exists()


def test_cli_run_returns_one_for_invalid_config(tmp_path: Path, capsys):
    config_path = tmp_path / "experiment.json"
    write_json(config_path, {"report_path": "outputs/report.json"})

    exit_code = main(["run", "--config", str(config_path)])

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "error: missing required experiment config field: dataset_path" in captured.err


def test_cli_run_returns_two_for_missing_required_args(capsys):
    exit_code = main(["run"])

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "usage:" in captured.err
    assert "--config" in captured.err


def test_cli_module_execution_runs_experiment(tmp_path: Path):
    config_path = tmp_path / "experiment.json"
    report_path, ledger_path = write_experiment_config(config_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "bayesprobe.cli",
            "run",
            "--config",
            str(config_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stderr == ""
    assert "BayesProbe experiment complete" in completed.stdout
    assert f"report={report_path}" in completed.stdout
    assert f"ledger={ledger_path}" in completed.stdout
    assert report_path.exists()
    assert ledger_path.exists()


def test_pyproject_registers_bayesprobe_script():
    metadata = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["scripts"]["bayesprobe"] == "bayesprobe.cli:main"
