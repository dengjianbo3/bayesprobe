import json
from pathlib import Path

from bayesprobe.evaluation.artifacts import (
    CapabilityArtifactStore,
    write_prepared_evaluation_set,
)
from bayesprobe.evaluation.contracts import ArmCaseResult
from bayesprobe.evaluation.hle import HLEDatasetAdapter, HLESelectionConfig
from bayesprobe.evaluation.python_probe import DockerPythonSandbox, PythonExecutionRecord
from bayesprobe.evaluation.runner import (
    CapabilityExperimentRunner,
    build_experiment_identity,
)
from bayesprobe.evaluation.scoring import score_and_write_experiment
from bayesprobe.provider_telemetry import (
    ProviderInvocationContext,
    ProviderInvocationRecord,
    ProviderUsage,
)


REVISION = "b" * 40


def synthetic_rows():
    rows = []
    for index in range(100):
        left = index + 1
        right = index + 2
        answer = left + right
        rows.append(
            {
                "id": f"synthetic_capability_{index:03d}",
                "category": "synthetic-arithmetic" if index < 50 else "synthetic-logic",
                "question": (
                    f"What is {left} + {right}?\n"
                    f"A. {answer - 1}\nB. {answer}\nC. {answer + 1}"
                ),
                "answer": "B",
                "answer_type": "multipleChoice",
                "image": None,
                "rationale": "Synthetic private rationale not copied.",
                "canary": "SYNTHETIC-INTEGRATION-CANARY",
            }
        )
    return rows


class CountingArm:
    def __init__(
        self,
        name,
        answer_for_index,
        *,
        provider_observer,
        python_observer_factory=None,
    ):
        self.name = name
        self.answer_for_index = answer_for_index
        self.provider_observer = provider_observer
        self.python_observer_factory = python_observer_factory
        self.calls = []

    def run_case(self, case):
        self.calls.append(case.sample_id)
        self.provider_observer.observe(
            ProviderInvocationRecord(
                task="synthetic_answer",
                adapter_kind="synthetic",
                model="synthetic-model",
                base_host=None,
                prompt_id="synthetic",
                prompt_version="v0.1",
                schema_name="MultipleChoiceAnswer",
                schema_version="v0.1",
                request_sha256="a" * 64,
                started_at="2026-07-11T00:00:00Z",
                completed_at="2026-07-11T00:00:01Z",
                latency_seconds=1.0,
                usage=ProviderUsage(
                    input_tokens=10,
                    cached_input_tokens=2,
                    reasoning_tokens=3,
                    output_tokens=5,
                    total_tokens=15,
                ),
                finish_reason="stop",
                response_id=f"{self.name}:{case.sample_id}",
                system_fingerprint="synthetic-fingerprint",
                outcome="success",
                error_category=None,
                context=ProviderInvocationContext(
                    experiment_id="synthetic-e2e",
                    arm=self.name,
                    sample_id=case.sample_id,
                ),
            )
        )
        if self.python_observer_factory is not None:
            self.python_observer_factory(case).observe(
                PythonExecutionRecord(
                    execution_id=f"python:{case.sample_id}",
                    run_id=f"run:{case.sample_id}",
                    cycle_id="cycle_1",
                    probe_id="probe_1",
                    code="print(1 + 1)",
                    code_sha256="b" * 64,
                    image_digest="sha256:" + "f" * 64,
                    started_at="2026-07-11T00:00:00Z",
                    completed_at="2026-07-11T00:00:01Z",
                    wall_seconds=1.0,
                    exit_code=0,
                    stdout="2\n",
                    stderr="",
                    output_truncated=False,
                    timed_out=False,
                    policy_violation=False,
                    repair_attempt_index=0,
                    policy_snapshot=DockerPythonSandbox().policy_snapshot(
                        image_digest="sha256:" + "f" * 64,
                    ),
                )
            )
        index = int(case.sample_id.rsplit("_", 1)[1])
        answer = self.answer_for_index(index)
        probabilities = (
            {"A": 0.05, "B": 0.9, "C": 0.05}
            if answer == "B"
            else {"A": 0.8, "B": 0.1, "C": 0.1}
        )
        return ArmCaseResult(
            sample_id=case.sample_id,
            arm=self.name,
            state="completed",
            answer_label=answer,
            probabilities=probabilities,
            answer_summary="Synthetic fixture answer.",
            process_metrics={"model_calls": 1},
        )


def test_synthetic_100_case_prepare_resume_score_report_workflow(tmp_path: Path):
    prepared = HLEDatasetAdapter().prepare_rows(
        synthetic_rows(),
        HLESelectionConfig(revision=REVISION, sample_count=100),
    )
    identity = build_experiment_identity(
        experiment_name="synthetic end-to-end capability pilot",
        code_git_sha="a" * 40,
        dataset_revision_sha=REVISION,
        selection_manifest_sha256=prepared.manifest_sha256,
        config_sha256="d" * 64,
        prompt_registry_sha256="e" * 64,
        python_image_digest="sha256:" + "f" * 64,
    )
    store = CapabilityArtifactStore(
        tmp_path / "restricted",
        identity,
        secret=b"fixed-integration-secret" * 2,
    )
    prepared_paths = write_prepared_evaluation_set(store.root, prepared)
    cases = list(prepared.runtime_cases)
    first_case = cases[0]
    provider_observer = store.provider_observer()

    direct = CountingArm(
        "direct_flash",
        lambda index: "B" if index % 2 == 0 else "A",
        provider_observer=provider_observer,
    )
    bayesprobe = CountingArm(
        "bayesprobe_python",
        lambda index: "B",
        provider_observer=provider_observer,
        python_observer_factory=lambda case: store.python_observer_for(
            "bayesprobe_python", case
        ),
    )

    # Simulate one terminal task from an interrupted earlier process.
    store.initialize_case("direct_flash", first_case.sample_id)
    store.mark_running("direct_flash", first_case.sample_id)
    store.write_terminal_result(direct.run_case(first_case))
    direct.calls.clear()

    runner = CapabilityExperimentRunner(
        identity=identity,
        cases=cases,
        arms={"direct_flash": direct, "bayesprobe_python": bayesprobe},
        artifact_store=store,
        direct_concurrency=8,
        bayesprobe_concurrency=4,
    )
    first_summary = runner.run()
    second_summary = runner.run()

    assert first_summary.task_count == 200
    assert first_summary.executed_count == 199
    assert first_summary.terminal_count == 200
    assert second_summary.executed_count == 0
    assert len(direct.calls) == 99
    assert len(bayesprobe.calls) == 100
    assert len(list(store.root.glob("arms/*/*/result.json"))) == 200
    assert prepared_paths.selection_manifest.exists()
    assert prepared_paths.gold_store.exists()
    provider_records = []
    python_records = []
    for case in cases:
        for arm in ("direct_flash", "bayesprobe_python"):
            paths = store.paths_for(arm, case.sample_id)
            records = [
                json.loads(line)
                for line in paths.provider_invocations_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line
            ]
            assert len(records) == 1
            provider_records.extend(records)
            executions = [
                json.loads(line)
                for line in paths.python_executions_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line
            ]
            assert len(executions) == (1 if arm == "bayesprobe_python" else 0)
            python_records.extend(executions)
    assert len(provider_records) == 200
    assert len({record["response_id"] for record in provider_records}) == 200
    assert len(python_records) == 100
    assert len({record["execution_id"] for record in python_records}) == 100

    score_paths = score_and_write_experiment(
        artifact_store=store,
        cases=cases,
        gold=prepared.gold_store,
        categories={entry.sample_id: entry.category for entry in prepared.manifest_entries},
        report_root=tmp_path / "reports",
        restricted_canaries=["SYNTHETIC-INTEGRATION-CANARY"],
        bootstrap_resamples=100,
    )

    summary = json.loads(score_paths.summary_json.read_text(encoding="utf-8"))
    assert summary["arms"]["bayesprobe_python"]["accuracy"] == 1.0
    assert summary["arms"]["direct_flash"]["accuracy"] == 0.5
    assert score_paths.score_marker.exists()
    assert len(list(store.root.glob("scoring_complete.json"))) == 1
    shareable_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            score_paths.summary_json,
            score_paths.summary_markdown,
            score_paths.paired_metrics,
            score_paths.provenance,
        )
    )
    assert "What is" not in shareable_text
    assert "synthetic_capability_" not in shareable_text
    assert "SYNTHETIC-INTEGRATION-CANARY" not in shareable_text
