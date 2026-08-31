"""Model loading utilities for SupportPilot AI."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from supportpilot.config import settings


@dataclass(frozen=True)
class ModelBundle:
    """Loaded model resources used by the inference layer."""

    tokenizer: Any
    model: Any
    device: torch.device

    id2label: dict[int, str]
    label2id: dict[str, int]

    num_labels: int
    source: str


def resolve_model_source(
    model_id: str | None = None,
) -> str:
    """Resolve the configured model source.

    The source may be either:
    - a local model directory
    - a Hugging Face model repository ID
    """

    source = (
        settings.model_id
        if model_id is None
        else model_id
    )

    if source is None:
        raise RuntimeError(
            "No model source configured. "
            "Set SUPPORTPILOT_MODEL_ID."
        )

    source = source.strip()

    if not source:
        raise RuntimeError(
            "No model source configured. "
            "Set SUPPORTPILOT_MODEL_ID."
        )

    return source


def _prepare_model_source(
    source: str,
    revision: str | None,
) -> tuple[str, dict[str, Any]]:
    """Prepare local or remote from_pretrained arguments."""

    kwargs: dict[str, Any] = {}

    if revision:
        kwargs["revision"] = revision

    local_path = Path(
        source
    ).expanduser()

    if local_path.exists():

        if not local_path.is_dir():
            raise ValueError(
                "Local model source must be a directory."
            )

        source = str(
            local_path.resolve()
        )

        kwargs["local_files_only"] = True

    return source, kwargs


def _build_label_mapping(
    model: Any,
) -> tuple[
    dict[int, str],
    dict[str, int],
]:
    """Read and validate label mappings from model config."""

    num_labels = int(
        model.config.num_labels
    )

    id2label = {
        int(index): str(label)
        for index, label
        in model.config.id2label.items()
    }

    label2id = {
        str(label): int(index)
        for label, index
        in model.config.label2id.items()
    }

    expected_ids = set(
        range(num_labels)
    )

    if set(id2label.keys()) != expected_ids:
        raise ValueError(
            "Model id2label mapping is incomplete."
        )

    if set(label2id.values()) != expected_ids:
        raise ValueError(
            "Model label2id mapping is incomplete."
        )

    for index, label in id2label.items():

        if label2id.get(label) != index:
            raise ValueError(
                "Model label mappings are inconsistent."
            )

    return id2label, label2id


def _resolve_device(
    device_name: str | None = None,
) -> torch.device:
    """Resolve the inference device."""

    if device_name:
        return torch.device(
            device_name
        )

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


@lru_cache(maxsize=1)
def load_model_bundle(
    model_id: str | None = None,
    model_revision: str | None = None,
    device_name: str | None = None,
) -> ModelBundle:
    """Load and cache tokenizer and classification model."""

    source = resolve_model_source(
        model_id
    )

    revision = (
        settings.model_revision
        if model_revision is None
        else model_revision
    )

    source, pretrained_kwargs = (
        _prepare_model_source(
            source=source,
            revision=revision,
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(
        source,
        **pretrained_kwargs,
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            source,
            **pretrained_kwargs,
        )
    )

    device = _resolve_device(
        device_name
    )

    model.to(
        device
    )

    model.eval()

    id2label, label2id = (
        _build_label_mapping(
            model
        )
    )

    return ModelBundle(
        tokenizer=tokenizer,
        model=model,
        device=device,
        id2label=id2label,
        label2id=label2id,
        num_labels=int(
            model.config.num_labels
        ),
        source=source,
    )