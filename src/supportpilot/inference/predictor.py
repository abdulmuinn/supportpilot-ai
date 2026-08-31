"""Prediction services for SupportPilot AI."""

from __future__ import annotations

from typing import Any, Sequence

import torch

from supportpilot.config import settings
from supportpilot.inference.confidence import (
    evaluate_confidence,
)
from supportpilot.inference.model_loader import (
    ModelBundle,
    load_model_bundle,
)


def _validate_text(
    text: str,
) -> str:
    """Validate and normalize a single input message."""

    if not isinstance(text, str):
        raise TypeError(
            "Input text must be a string."
        )

    clean_text = text.strip()

    if not clean_text:
        raise ValueError(
            "Input text must not be empty."
        )

    return clean_text


def _get_bundle(
    bundle: ModelBundle | None,
) -> ModelBundle:
    """Return an injected model bundle or load the default one."""

    if bundle is not None:
        return bundle

    return load_model_bundle()


def predict_top_k(
    text: str,
    top_k: int = 5,
    *,
    bundle: ModelBundle | None = None,
    max_length: int | None = None,
) -> dict[str, Any]:
    """Return the highest-probability intent candidates."""

    clean_text = _validate_text(
        text
    )

    if not isinstance(top_k, int):
        raise TypeError(
            "top_k must be an integer."
        )

    if top_k < 1:
        raise ValueError(
            "top_k must be at least 1."
        )

    model_bundle = _get_bundle(
        bundle
    )

    effective_top_k = min(
        top_k,
        model_bundle.num_labels,
    )

    sequence_length = (
        settings.max_length
        if max_length is None
        else max_length
    )

    encoded = model_bundle.tokenizer(
        clean_text,
        truncation=True,
        max_length=sequence_length,
        return_tensors="pt",
    )

    encoded = {
        key: value.to(
            model_bundle.device
        )
        for key, value in encoded.items()
    }

    with torch.inference_mode():

        logits = model_bundle.model(
            **encoded
        ).logits

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        top_probabilities, top_ids = (
            torch.topk(
                probabilities,
                k=effective_top_k,
                dim=-1,
            )
        )

    probabilities_list = (
        top_probabilities[0]
        .detach()
        .cpu()
        .tolist()
    )

    ids_list = (
        top_ids[0]
        .detach()
        .cpu()
        .tolist()
    )

    predictions: list[
        dict[str, Any]
    ] = []

    for rank, (
        class_id,
        probability,
    ) in enumerate(
        zip(
            ids_list,
            probabilities_list,
        ),
        start=1,
    ):

        class_id = int(
            class_id
        )

        probability = float(
            probability
        )

        predictions.append(
            {
                "rank": rank,
                "predicted_id": class_id,
                "predicted_intent":
                    model_bundle.id2label[
                        class_id
                    ],
                "confidence":
                    probability,
                "confidence_percent":
                    probability * 100,
            }
        )

    return {
        "text": clean_text,
        "top_k": effective_top_k,
        "predictions": predictions,
    }


def analyze_prediction(
    text: str,
    *,
    bundle: ModelBundle | None = None,
    max_length: int | None = None,
) -> dict[str, Any]:
    """Return Top-1, Top-2, and confidence margin."""

    result = predict_top_k(
        text=text,
        top_k=2,
        bundle=bundle,
        max_length=max_length,
    )

    top_1 = result[
        "predictions"
    ][0]

    top_2 = result[
        "predictions"
    ][1]

    margin = (
        top_1["confidence"]
        - top_2["confidence"]
    )

    return {
        "text": result["text"],

        "predicted_id":
            top_1["predicted_id"],

        "predicted_intent":
            top_1["predicted_intent"],

        "confidence":
            top_1["confidence"],

        "confidence_percent":
            top_1["confidence_percent"],

        "second_best_id":
            top_2["predicted_id"],

        "second_best_intent":
            top_2["predicted_intent"],

        "second_best_confidence":
            top_2["confidence"],

        "second_best_confidence_percent":
            top_2["confidence_percent"],

        "confidence_margin":
            margin,

        "confidence_margin_percent":
            margin * 100,
    }


def predict_with_fallback(
    text: str,
    *,
    bundle: ModelBundle | None = None,
    min_confidence: float | None = None,
    min_margin: float | None = None,
    max_length: int | None = None,
) -> dict[str, Any]:
    """Predict an intent and apply the confidence policy."""

    analysis = analyze_prediction(
        text=text,
        bundle=bundle,
        max_length=max_length,
    )

    decision = evaluate_confidence(
        predicted_intent=analysis[
            "predicted_intent"
        ],
        confidence=analysis[
            "confidence"
        ],
        margin=analysis[
            "confidence_margin"
        ],
        min_confidence=min_confidence,
        min_margin=min_margin,
    )

    return {
        **analysis,

        "final_intent":
            decision.final_intent,

        "min_confidence": (
            settings.min_confidence
            if min_confidence is None
            else min_confidence
        ),

        "min_margin": (
            settings.min_margin
            if min_margin is None
            else min_margin
        ),

        "accepted":
            decision.accepted,

        "status":
            decision.status,

        "reason":
            decision.reason,
    }


def predict_batch(
    texts: Sequence[str],
    *,
    bundle: ModelBundle | None = None,
    batch_size: int = 32,
    min_confidence: float | None = None,
    min_margin: float | None = None,
    max_length: int | None = None,
) -> list[dict[str, Any]]:
    """Run efficient batched intent classification."""

    if not isinstance(
        texts,
        (list, tuple),
    ):
        raise TypeError(
            "texts must be a list or tuple."
        )

    if not texts:
        raise ValueError(
            "texts must not be empty."
        )

    if not isinstance(
        batch_size,
        int,
    ):
        raise TypeError(
            "batch_size must be an integer."
        )

    if batch_size < 1:
        raise ValueError(
            "batch_size must be at least 1."
        )

    clean_texts = [
        _validate_text(text)
        for text in texts
    ]

    model_bundle = _get_bundle(
        bundle
    )

    sequence_length = (
        settings.max_length
        if max_length is None
        else max_length
    )

    results: list[
        dict[str, Any]
    ] = []

    for start_index in range(
        0,
        len(clean_texts),
        batch_size,
    ):

        batch_texts = clean_texts[
            start_index:
            start_index + batch_size
        ]

        encoded = (
            model_bundle.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=sequence_length,
                return_tensors="pt",
            )
        )

        encoded = {
            key: value.to(
                model_bundle.device
            )
            for key, value
            in encoded.items()
        }

        with torch.inference_mode():

            logits = model_bundle.model(
                **encoded
            ).logits

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )

            top_probabilities, top_ids = (
                torch.topk(
                    probabilities,
                    k=2,
                    dim=-1,
                )
            )

        top_probabilities = (
            top_probabilities
            .detach()
            .cpu()
            .tolist()
        )

        top_ids = (
            top_ids
            .detach()
            .cpu()
            .tolist()
        )

        for (
            message,
            probabilities_row,
            ids_row,
        ) in zip(
            batch_texts,
            top_probabilities,
            top_ids,
        ):

            top_1_id = int(
                ids_row[0]
            )

            top_2_id = int(
                ids_row[1]
            )

            confidence = float(
                probabilities_row[0]
            )

            second_confidence = float(
                probabilities_row[1]
            )

            margin = (
                confidence
                - second_confidence
            )

            predicted_intent = (
                model_bundle.id2label[
                    top_1_id
                ]
            )

            decision = (
                evaluate_confidence(
                    predicted_intent=
                        predicted_intent,
                    confidence=
                        confidence,
                    margin=
                        margin,
                    min_confidence=
                        min_confidence,
                    min_margin=
                        min_margin,
                )
            )

            results.append(
                {
                    "text":
                        message,

                    "predicted_id":
                        top_1_id,

                    "predicted_intent":
                        predicted_intent,

                    "final_intent":
                        decision.final_intent,

                    "confidence":
                        confidence,

                    "confidence_percent":
                        confidence * 100,

                    "second_best_id":
                        top_2_id,

                    "second_best_intent":
                        model_bundle.id2label[
                            top_2_id
                        ],

                    "second_best_confidence":
                        second_confidence,

                    "confidence_margin":
                        margin,

                    "confidence_margin_percent":
                        margin * 100,

                    "accepted":
                        decision.accepted,

                    "status":
                        decision.status,

                    "reason":
                        decision.reason,
                }
            )

    return results