"""Utilities for recovering training arguments from experiment checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch


def _as_dict(value: Any) -> dict[str, Any]:
    """Convert checkpoint argument representations to a plain dictionary."""
    if value is None:
        return {}
    if isinstance(value, argparse.Namespace):
        return vars(value).copy()
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(
        "checkpoint args must be a dict or argparse.Namespace, "
        f"got {type(value).__name__}"
    )


def load_checkpoint_args(
    checkpoint_path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load args with legacy support and explicit manual overrides.

    Resolution order is sibling ``args.json``, embedded checkpoint args, and
    finally explicit ``overrides``. Explicit overrides always win.
    """
    checkpoint_path = Path(checkpoint_path)
    resolved: dict[str, Any] = {}

    args_json = checkpoint_path.parent.parent / "args.json"
    if args_json.is_file():
        with args_json.open(encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, dict):
            raise ValueError(f"Expected an object in {args_json}")
        resolved.update(data)

    # Required for older StyleGAN checkpoints whose args field contains a
    # Namespace. Only load trusted checkpoint files.
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
    except TypeError:  # PyTorch versions predating the weights_only argument.
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Expected checkpoint mapping in {checkpoint_path}")
    embedded = _as_dict(checkpoint.get("args"))
    for key, value in embedded.items():
        resolved.setdefault(key, value)

    if overrides:
        resolved.update({key: value for key, value in overrides.items() if value is not None})

    return resolved
