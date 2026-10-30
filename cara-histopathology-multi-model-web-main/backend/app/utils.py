"""Shared constants, paths, preprocessing helpers and the common model interface."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

# --------------------------------------------------------------------------- #
# Dataset constants
# --------------------------------------------------------------------------- #
# The 19 tissue types of PanNuke, in the spelling used by the official types.npy
CLASS_NAMES: list[str] = [
    "Adrenal_gland", "Bile-duct", "Bladder", "Breast", "Cervix", "Colon",
    "Esophagus", "HeadNeck", "Kidney", "Liver", "Lung", "Ovarian",
    "Pancreatic", "Prostate", "Skin", "Stomach", "Testis", "Thyroid", "Uterus",
]
CLASS_DISPLAY: dict[str, str] = {
    "Adrenal_gland": "Adrenal gland",
    "Bile-duct": "Bile duct",
    "HeadNeck": "Head & Neck",
}
NUM_CLASSES = len(CLASS_NAMES)  # 19
IMG_SIZE = 256
FOLD_SIZES = {1: 2656, 2: 2523, 3: 2722}          # official PanNuke fold sizes
SPLIT_SIZES = {"train": 5688, "val": 632, "test": 1581}  # split used in the CARA report
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def class_display(name: str) -> str:
    return CLASS_DISPLAY.get(name, name.replace("_", " "))


# --------------------------------------------------------------------------- #
# Model registry metadata (the classes themselves live in app/models/*.py)
# --------------------------------------------------------------------------- #
MODEL_KEYS: list[str] = [
    "resnet18", "mobilenetv3", "efficientnet", "graphsage",
    "gan_classifier", "swin_transformer", "capsnet", "vmamba",
]
MODEL_INFO: dict[str, dict[str, str]] = {
    "resnet18": {"display": "ResNet18", "origin": "original",
                 "description": "Residual CNN with 18 layers (BasicBlock x [2,2,2,2])."},
    "mobilenetv3": {"display": "MobileNetV3-Small", "origin": "original",
                    "description": "Lightweight CNN with inverted residuals, SE and h-swish."},
    "efficientnet": {"display": "EfficientNet-B0", "origin": "original",
                     "description": "Compound-scaled MBConv network with squeeze-excitation."},
    "graphsage": {"display": "GraphSAGE", "origin": "new",
                  "description": "Image -> 8x8 patch graph, classified by 3 GraphSAGE layers."},
    "gan_classifier": {"display": "GAN Classifier", "origin": "new",
                       "description": "Generator-style encoder + discriminator classification head."},
    "swin_transformer": {"display": "Swin-Tiny", "origin": "new",
                         "description": "Shifted-window hierarchical vision transformer (Swin-T)."},
    "capsnet": {"display": "CapsNet", "origin": "new",
                "description": "Primary capsules + dynamic routing-by-agreement."},
    "vmamba": {"display": "VMamba", "origin": "new",
               "description": "Vision Mamba: 2D selective state-space (SS2D) blocks, 4-way scan."},
}

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("CARA_ROOT", BACKEND_DIR.parent)).resolve()
DATA_DIR = Path(os.environ.get("CARA_DATA_DIR", PROJECT_ROOT / "data"))
PANNUKE_DIR = Path(os.environ.get("CARA_PANNUKE_DIR", DATA_DIR / "pannuke"))
SAMPLES_DIR = Path(os.environ.get("CARA_SAMPLES_DIR", DATA_DIR / "samples"))
CHECKPOINT_DIR = Path(os.environ.get("CARA_CHECKPOINT_DIR", BACKEND_DIR / "checkpoints"))
RESULTS_DIR = Path(os.environ.get("CARA_RESULTS_DIR", BACKEND_DIR / "results"))


def ensure_dirs() -> None:
    for d in (DATA_DIR, PANNUKE_DIR, SAMPLES_DIR, CHECKPOINT_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Misc helpers
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str | None = None) -> torch.device:
    prefer = prefer or os.environ.get("CARA_DEVICE", "auto")
    if prefer == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(prefer)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    z = logits - logits.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy / torch scalars and arrays into JSON-safe objects."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def save_json(path: Path | str, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2)


def load_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Image preprocessing (shared by dataset, API and scripts)
# --------------------------------------------------------------------------- #
def pil_to_uint8_256(img: Image.Image) -> np.ndarray:
    """Centre-crop to a square, resize to 256x256 and return an (H, W, 3) uint8 array."""
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    if s != IMG_SIZE:
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    return np.asarray(img, dtype=np.uint8)


_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
_STD = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)


def uint8_to_tensor(arr: np.ndarray) -> torch.Tensor:
    """(H, W, 3) uint8 -> normalised float tensor (3, H, W)."""
    x = (arr.astype(np.float32) / 255.0 - _MEAN) / _STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


# --------------------------------------------------------------------------- #
# Shared model interface
# --------------------------------------------------------------------------- #
class BaseClassifier(nn.Module):
    """Every CARA model derives from this class.

    Contract
      * forward(x): (B, 3, 256, 256) float tensor -> (B, 19) logits (train and eval mode)
      * predict_proba(x): softmax probabilities, eval mode, no gradients
      * extra_loss(x, y, logits): optional auxiliary loss used by train_all.py
    """

    key: str = "base"

    def extra_loss(self, x: torch.Tensor, y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        return logits.new_zeros(())

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            return torch.softmax(self.forward(x), dim=-1)
        finally:
            self.train(was_training)


def init_weights(module: nn.Module) -> None:
    """Kaiming / trunc-normal init used by the from-scratch CNNs."""
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
