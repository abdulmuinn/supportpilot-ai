"""Unit tests for dashboard batch helpers."""

from supportpilot.ui.batch import (
    run_batch_inference,
)


class FakeAPIClient:
    """Minimal fake API client for batch tests."""

    def __init__(self):
        self.calls = []

    def predict_batch(
        self,
        texts,
        *,
        batch_size=32,
    ):
        texts = list(texts)

        self.calls.append(
            {
                "texts": texts,
                "batch_size": batch_size,
            }
        )

        return {
            "total": len(texts),
            "accepted": len(texts),
            "fallback": 0,
            "predictions": [
                {
                    "text": text,
                }
                for text in texts
            ],
        }


def test_single_request_batch():
    client = FakeAPIClient()

    texts = [
        "message 1",
        "message 2",
    ]

    result = run_batch_inference(
        client,
        texts,
    )

    assert result["total"] == 2
    assert len(client.calls) == 1


def test_large_batch_is_chunked():
    client = FakeAPIClient()

    texts = [
        f"message {index}"
        for index in range(250)
    ]

    result = run_batch_inference(
        client,
        texts,
    )

    assert result["total"] == 250

    assert [
        len(call["texts"])
        for call in client.calls
    ] == [
        100,
        100,
        50,
    ]


def test_prediction_order_is_preserved():
    client = FakeAPIClient()

    texts = [
        f"message {index}"
        for index in range(205)
    ]

    result = run_batch_inference(
        client,
        texts,
    )

    returned_texts = [
        prediction["text"]
        for prediction in result[
            "predictions"
        ]
    ]

    assert returned_texts == texts


def test_reject_invalid_request_size():
    client = FakeAPIClient()

    try:
        run_batch_inference(
            client,
            ["message"],
            request_size=101,
        )

    except ValueError as error:
        assert (
            str(error)
            == "request_size must be between 1 and 100."
        )

    else:
        raise AssertionError(
            "Expected ValueError."
        )