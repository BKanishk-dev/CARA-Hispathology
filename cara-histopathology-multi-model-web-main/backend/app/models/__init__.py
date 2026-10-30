"""Model registry: every model exposes `build_model(num_classes)` and derives BaseClassifier."""
from __future__ import annotations

import importlib
from pathlib import Path

import torch

from ..utils import MODEL_KEYS, NUM_CLASSES, BaseClassifier, count_parameters

_MODULES = {
    "resnet18": "resnet18",
    "mobilenetv3": "mobilenetv3",
    "efficientnet": "efficientnet",
    "graphsage": "graphsage",
    "gan_classifier": "gan_classifier",
    "swin_transformer": "swin_transformer",
    "capsnet": "capsnet",
    "vmamba": "vmamba",
}


def build_model(name: str, num_classes: int = NUM_CLASSES) -> BaseClassifier:
    if name not in _MODULES:
        raise KeyError(f"Unknown model '{name}'. Available: {MODEL_KEYS}")
    module = importlib.import_module(f"{__name__}.{_MODULES[name]}")
    return module.build_model(num_classes)


def save_checkpoint(model: BaseClassifier, path: Path | str, **meta) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": getattr(model, "key", "unknown"), "state_dict": model.state_dict(),
                "num_classes": NUM_CLASSES, "dummy": False, **meta}, path)


def save_dummy_checkpoint(name: str, path: Path | str) -> None:
    """Tiny placeholder .pth (no weights) so the file layout is complete in Demo Mode."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": name, "state_dict": None, "num_classes": NUM_CLASSES, "dummy": True,
                "note": "Placeholder checkpoint for Demo Mode. Run scripts/train_all.py to create real weights."},
               path)


def load_checkpoint(model: BaseClassifier, path: Path | str, device: str | torch.device = "cpu") -> bool:
    """Load weights from a .pth file. Returns True when real weights were loaded and False
    when the file is missing or is a Demo-Mode placeholder (the model keeps its initial weights)."""
    path = Path(path)
    if not path.exists():
        return False
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("state_dict") if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    if state is None or (isinstance(ckpt, dict) and ckpt.get("dummy")):
        return False
    model.load_state_dict(state)
    return True


def checkpoint_has_weights(path: Path | str) -> bool:
    """True when the file exists, is readable and holds real (non-placeholder) weights."""
    path = Path(path)
    if not path.exists():
        return False
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    if isinstance(ckpt, dict):
        if ckpt.get("dummy", False):
            return False
        return ckpt.get("state_dict", ckpt) is not None
    return True


def checkpoint_is_readable(path: Path | str) -> bool:
    try:
        torch.load(Path(path), map_location="cpu", weights_only=False)
        return True
    except Exception:
        return False


__all__ = ["build_model", "save_checkpoint", "save_dummy_checkpoint", "load_checkpoint",
           "checkpoint_has_weights", "checkpoint_is_readable", "count_parameters", "MODEL_KEYS"]
