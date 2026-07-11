import pytest

from bayesprobe.evaluation.scoring import assert_shareable_payload_safe


def test_leak_scan_accepts_aggregate_and_hmac_correctness_payload():
    assert_shareable_payload_safe(
        {
            "accuracy": 0.5,
            "paired": [
                {
                    "sample_pseudonym": "f" * 64,
                    "bayesprobe_correct": True,
                    "direct_correct": False,
                }
            ],
        },
        restricted_values=["private_sample_1", "Full private question text"],
        canaries=["PRIVATE-CANARY"],
        provider_secrets=["sk-private-secret"],
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "Full private question text"},
        {"sample_id": "private_sample_1"},
        {"answer_label": "A"},
        {"nested": [{"text": "prefix PRIVATE-CANARY suffix"}]},
        {"error": "request failed for sk-private-secret"},
        {"raw_model_response": {"content": "private"}},
        {"python_code": "print('private')"},
    ],
)
def test_leak_scan_rejects_benchmark_or_secret_material(payload):
    with pytest.raises(ValueError, match="shareable artifact leak"):
        assert_shareable_payload_safe(
            payload,
            restricted_values=["private_sample_1", "Full private question text"],
            canaries=["PRIVATE-CANARY"],
            provider_secrets=["sk-private-secret"],
        )
