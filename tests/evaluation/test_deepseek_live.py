import os

import pytest

from bayesprobe.evaluation.arms import DirectFlashArm
from bayesprobe.evaluation.contracts import EvaluationCase
from bayesprobe.model_gateway import (
    ModelGatewayConfig,
    ProviderRequestControls,
    build_model_gateway,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("BAYESPROBE_RUN_DEEPSEEK_LIVE") != "1"
    or not os.environ.get("DEEPSEEK_API_KEY"),
    reason=(
        "set BAYESPROBE_RUN_DEEPSEEK_LIVE=1 and DEEPSEEK_API_KEY "
        "to run the self-authored provider smoke"
    ),
)


def test_deepseek_live_answers_self_authored_non_hle_question():
    gateway = build_model_gateway(
        ModelGatewayConfig(
            kind="openai_chat_completions",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-flash",
            api_key_env="DEEPSEEK_API_KEY",
            timeout_seconds=900,
            max_output_tokens=65536,
            request_controls=ProviderRequestControls(
                temperature=0,
                top_p=1,
                thinking="enabled",
                reasoning_effort="max",
            ),
        )
    )
    case = EvaluationCase(
        sample_id="self_authored_live_smoke",
        question=(
            "A sealed box contains 3 red and 2 blue balls. One ball is drawn "
            "uniformly without replacement. What is the probability it is red?\n\n"
            "Answer Choices:\nA. 2/5\nB. 1/2\nC. 3/5\nD. 2/3"
        ),
        choices={"A": "2/5", "B": "1/2", "C": "3/5", "D": "2/3"},
    )

    result = DirectFlashArm(gateway).run_case(case)

    assert result.state == "completed"
    assert result.answer_label == "C"
