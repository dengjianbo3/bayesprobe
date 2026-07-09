import json
import os
from pathlib import Path

import pytest

from bayesprobe.experiment_runner import (
    ExperimentRunConfig,
    run_benchmark_experiment,
)
from bayesprobe.model_gateway import StructuredModelRequest, evidence_judgment_from_mapping
from bayesprobe.openai_gateway import OpenAIModelGatewayConfig, OpenAIResponsesModelGateway


def test_openai_live_smoke_judges_evidence_when_explicitly_enabled():
    if os.environ.get("BAYESPROBE_RUN_OPENAI_LIVE") != "1":
        pytest.skip("set BAYESPROBE_RUN_OPENAI_LIVE=1 to run OpenAI live smoke")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("set OPENAI_API_KEY to run OpenAI live smoke")

    gateway = OpenAIResponsesModelGateway(
        config=OpenAIModelGatewayConfig(model="gpt-5.5", max_output_tokens=256)
    )
    payload = gateway.complete_structured(
        StructuredModelRequest(
            task="judge_evidence",
            input={
                "signal_id": "S_live_openai",
                "source_type": "live_smoke",
                "source": "pytest",
                "raw_content": "SUPPORTS: this fixture supports H1 more than H2.",
                "target_hypotheses": ["H1", "H2"],
            },
            prompt_id="evidence_judgment",
            prompt_version="v0.1",
            schema_name="EvidenceJudgment",
            schema_version="v0.1",
        )
    )

    judgment = evidence_judgment_from_mapping(payload)

    assert judgment.interpretation


def test_openai_live_benchmark_writes_model_invocation_artifacts_when_enabled(
    tmp_path: Path,
):
    if os.environ.get("BAYESPROBE_RUN_OPENAI_LIVE") != "1":
        pytest.skip("set BAYESPROBE_RUN_OPENAI_LIVE=1 to run OpenAI live smoke")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("set OPENAI_API_KEY to run OpenAI live smoke")

    dataset_path = tmp_path / "one-sample-openai.json"
    dataset_path.write_text(
        json.dumps(
            {
                "dataset_name": "openai_live_artifact_smoke",
                "samples": [
                    {
                        "sample_id": "openai_live_passive",
                        "question_or_claim": (
                            "Can OpenAI live smoke produce provenance artifacts?"
                        ),
                        "signal_shape": "passive_only",
                        "gold_best_hypothesis": "H1",
                        "passive_signals": [
                            {
                                "signal_id": "S_openai_live",
                                "source_type": "live_smoke",
                                "source": "pytest",
                                "raw_content": "SUPPORTS: this smoke fixture supports H1 more than H2.",
                                "target_hypotheses": ["H1", "H2"],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    artifact_dir = tmp_path / "artifacts" / "openai-live"

    result = run_benchmark_experiment(
        ExperimentRunConfig(
            dataset_path=dataset_path,
            report_path=tmp_path / "report.json",
            artifact_dir=artifact_dir,
            model_gateway={
                "kind": "openai",
                "model": os.environ.get("BAYESPROBE_OPENAI_MODEL", "gpt-5.5"),
                "max_output_tokens": 256,
            },
        )
    )

    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    model_invocations = json.loads(
        (artifact_dir / "model_invocations.json").read_text(encoding="utf-8")
    )
    assert result.suite_result.sample_count == 1
    assert manifest["model_invocations_path"] == str(
        artifact_dir / "model_invocations.json"
    )
    assert manifest["model_invocation_count"] >= 1
    assert model_invocations["invocations"][0]["adapter_kind"] == "openai"
