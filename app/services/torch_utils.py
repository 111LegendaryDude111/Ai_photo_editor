from __future__ import annotations

from contextlib import nullcontext
from typing import Any


def maybe_import_torch() -> Any | None:
    try:  # pragma: no cover - optional dependency
        import torch

        return torch
    except Exception:
        return None


def get_inference_context():
    torch = maybe_import_torch()
    if torch is None:
        return nullcontext()
    return torch.inference_mode()


def cuda_available() -> bool:
    torch = maybe_import_torch()
    return bool(torch and torch.cuda.is_available())


def clear_cuda_cache() -> None:
    torch = maybe_import_torch()
    if not torch or not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()


def gpu_memory_allocated_mb() -> float:
    torch = maybe_import_torch()
    if not torch or not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.memory_allocated() / (1024 * 1024))
