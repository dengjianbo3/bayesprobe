import os

import pytest

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
