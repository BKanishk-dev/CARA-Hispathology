"""Datasets: real PanNuke (3 folds) and a synthetic H&E-like generator for Demo Mode.

PanNuke official layout (as downloaded from the Warwick TIA Centre)::

    data/pannuke/
        Fold 1/images/fold1/images.npy   (2656, 256, 256, 3)
        Fold 1/images/fold1/types.npy    (2656,)  tissue-type strings
        Fold 2/images/fold2/images.npy   (2523, ...)
        Fold 3/images/fold3/images.npy   (2722, ...)

Any layout is accepted as long as, for every fold N in {1,2,3}, a folder whose name
contains "fold N" (case / space / underscore insensitive) holds images.npy and types.npy.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from .utils import (CLASS_NAMES, FOLD_SIZES, IMG_SIZE, NUM_CLASSES, PANNUKE_DIR, SPLIT_SIZES,
                    load_json, uint8_to_tensor)

try:  # scipy is only needed for the synthetic generator
    from scipy.ndimage import gaussian_filter
except Exception:  # pragma: no cover
    gaussian_filter = None

SPLITS_FILE = "splits.json"


# --------------------------------------------------------------------------- #
# Locating the PanNuke folds
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return re.sub(r"[\s_\-]+", "", s.lower())


def find_fold_files(root: Path, fold: int) -> tuple[Path, Path] | None:
    """Return (images.npy, types.npy) for a fold, or None if not found."""
    root = Path(root)
    if not root.exists():
        return None
    tag = f"fold{fold}"
    for img_path in sorted(root.rglob("images.npy")):
        rel = _norm(str(img_path.relative_to(root).parent))
        if tag in rel:
            types_path = img_path.with_name("types.npy")
            if not types_path.exists():
                cands = [p for p in root.rglob("types.npy") if tag in _norm(str(p.relative_to(root).parent))]
                if not cands:
                    continue
                types_path = cands[0]
            return img_path, types_path
    return None


def pannuke_available(root: Path = PANNUKE_DIR) -> bool:
    return all(find_fold_files(root, f) is not None for f in (1, 2, 3))


def _load_types(path: Path) -> np.ndarray:
    try:
        arr = np.load(path, allow_pickle=False)
    except ValueError:  # object arrays need pickle (official files are plain unicode arrays)
        arr = np.load(path, allow_pickle=True)
    return np.asarray([str(t) for t in arr])


def load_fold_arrays(root: Path = PANNUKE_DIR) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Memory-mapped image arrays and integer labels for the three folds."""
    images, labels = [], []
    for f in (1, 2, 3):
        found = find_fold_files(root, f)
        if found is None:
            raise FileNotFoundError(
                f"PanNuke fold {f} not found under {root}. See the README section 'Real mode'.")
        img_p, typ_p = found
        imgs = np.load(img_p, mmap_mode="r")
        types = _load_types(typ_p)
        if len(imgs) != len(types):
            raise ValueError(f"Fold {f}: {len(imgs)} images but {len(types)} labels")
        if len(imgs) != FOLD_SIZES[f]:
            warnings.warn(f"Fold {f} has {len(imgs)} images (official PanNuke fold has {FOLD_SIZES[f]}).")
        unknown = sorted(set(types) - set(CLASS_NAMES))
        if unknown:
            raise ValueError(f"Fold {f}: unknown tissue types {unknown}")
        images.append(imgs)
        labels.append(np.array([CLASS_NAMES.index(t) for t in types], dtype=np.int64))
    return images, labels


def split_sizes(total: int) -> dict[str, int]:
    """Exactly 5688 / 632 / 1581 for the full 7901 images, proportional otherwise."""
    if total == sum(SPLIT_SIZES.values()):
        return dict(SPLIT_SIZES)
    full = sum(SPLIT_SIZES.values())
    n_test = int(round(total * SPLIT_SIZES["test"] / full))
    n_val = int(round(total * SPLIT_SIZES["val"] / full))
    return {"train": total - n_val - n_test, "val": n_val, "test": n_test}


def stratified_split(labels: np.ndarray, sizes: dict[str, int], seed: int = 42) -> dict[str, np.ndarray]:
    """Stratified (by tissue type) random split with exact split sizes."""
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(labels))
    rest, test = train_test_split(idx, test_size=sizes["test"], stratify=labels, random_state=seed)
    train, val = train_test_split(rest, test_size=sizes["val"], stratify=labels[rest], random_state=seed)
    return {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}


# --------------------------------------------------------------------------- #
# Real dataset
# --------------------------------------------------------------------------- #
class PanNukeDataset(Dataset):
    """Reads the three PanNuke folds as one concatenated array and serves one split."""

    def __init__(self, split: str, root: Path = PANNUKE_DIR, train_augment: bool | None = None):
        splits_path = Path(root) / SPLITS_FILE
        if not splits_path.exists():
            raise FileNotFoundError(f"{splits_path} missing - run `python scripts/prepare_pannuke.py` first.")
        self.split = split
        self.augment = (split == "train") if train_augment is None else train_augment
        self.images, labels = load_fold_arrays(root)
        self.offsets = np.cumsum([0] + [len(x) for x in self.images])
        self.all_labels = np.concatenate(labels)
        info = load_json(splits_path)
        self.indices = np.asarray(info["splits"][split], dtype=np.int64)
        self.labels = self.all_labels[self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def raw_uint8(self, i: int) -> np.ndarray:
        g = int(self.indices[i])
        fold = int(np.searchsorted(self.offsets, g, side="right") - 1)
        arr = np.asarray(self.images[fold][g - self.offsets[fold]])
        return np.clip(arr, 0, 255).astype(np.uint8)          # PanNuke stores float64 in [0, 255]

    def __getitem__(self, i: int):
        arr = self.raw_uint8(i)
        if self.augment:
            arr = _augment(arr, np.random.default_rng())
        return uint8_to_tensor(arr), int(self.labels[i])


def _augment(arr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        arr = arr[:, ::-1]
    if rng.random() < 0.5:
        arr = arr[::-1]
    arr = np.rot90(arr, k=int(rng.integers(0, 4)))
    # light colour jitter (brightness / per-channel gain) to mimic stain variation
    gain = 1.0 + rng.uniform(-0.08, 0.08, size=3)
    out = np.clip(arr.astype(np.float32) * gain * (1.0 + rng.uniform(-0.1, 0.1)), 0, 255)
    return np.ascontiguousarray(out.astype(np.uint8))


# --------------------------------------------------------------------------- #
# Synthetic dataset (Demo Mode)
# --------------------------------------------------------------------------- #
def _blur(a: np.ndarray, sigma) -> np.ndarray:
    if gaussian_filter is None:
        raise RuntimeError("scipy is required for the synthetic dataset (pip install scipy).")
    return gaussian_filter(a, sigma=sigma, mode="reflect")


def synth_image(label: int, rng: np.random.Generator, size: int = IMG_SIZE,
                degrade: bool = True) -> np.ndarray:
    """Class-conditional synthetic H&E-like patch (uint8, size x size x 3).

    Each tissue class gets its own eosin tint, nuclear density, nuclear size/elongation and
    fibre texture, so that the images are separable but still noisy.  A fraction of the
    images is blurred and/or has a faded stain to give the quality score something to detect.
    """
    ang = 2 * np.pi * label / NUM_CLASSES
    tint = np.array([np.cos(ang), np.sin(ang), np.cos(ang + 2.1)]) * 16.0
    base = np.clip(np.array([226.0, 150.0, 190.0]) + tint, 0, 255)

    # low-frequency cytoplasm texture
    tex = _blur(rng.normal(size=(size, size)), 14)
    tex = tex / (np.abs(tex).max() + 1e-6)
    img = base[None, None, :] * (1.0 + 0.10 * tex[..., None])

    # fibres for some classes
    if label % 4 == 0:
        fib = _blur(rng.normal(size=(size, size)), (1.2, 9.0))
        fib = fib / (np.abs(fib).max() + 1e-6)
        img = img * (1.0 + 0.07 * fib[..., None])

    # nuclei: class-specific density, size and elongation
    density = 55 + 22 * ((label * 7) % 11)
    radius = 2.6 + 0.7 * (label % 5)
    elong = 1.0 + 0.6 * (label % 3)
    pts = np.zeros((size, size))
    ys = rng.integers(0, size, density)
    xs = rng.integers(0, size, density)
    pts[ys, xs] = 1.0
    nuc = _blur(pts, (radius * elong, radius))
    nuc = np.clip(nuc / (nuc.max() + 1e-9) * 3.2, 0, 1)
    nuc = nuc ** 0.8
    nuc_color = np.clip(np.array([72.0, 52.0, 128.0]) + tint * 0.5, 0, 255)
    img = img * (1 - nuc[..., None]) + nuc_color[None, None, :] * nuc[..., None]

    img += rng.normal(0, 4.0, size=img.shape)

    if degrade:
        if rng.random() < 0.25:                                # out-of-focus
            img = _blur(img, (rng.uniform(1.5, 3.5), rng.uniform(1.5, 3.5), 0))
        if rng.random() < 0.20:                                # faded / washed-out stain
            fade = rng.uniform(0.25, 0.6)
            img = img * (1 - fade) + 250.0 * fade
    return np.clip(img, 0, 255).astype(np.uint8)


class SyntheticPanNuke(Dataset):
    """Deterministic synthetic stand-in for PanNuke with the same split sizes."""

    def __init__(self, split: str, n: int | None = None, seed: int = 42, train_augment: bool = False):
        self.split = split
        n = n if n is not None else SPLIT_SIZES[split]
        offset = {"train": 0, "val": 1, "test": 2}[split]
        self.seed = seed * 7919 + offset * 104729
        rng = np.random.default_rng(self.seed)
        # near-uniform class prior with mild imbalance, like PanNuke
        prior = rng.dirichlet(np.full(NUM_CLASSES, 12.0))
        self.labels = rng.choice(NUM_CLASSES, size=n, p=prior).astype(np.int64)
        self.augment = train_augment
        self._cache: dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.labels)

    def raw_uint8(self, i: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed + 1_000_003 * (i + 1))
        return synth_image(int(self.labels[i]), rng)

    def __getitem__(self, i: int):
        arr = self.raw_uint8(i)
        if self.augment:
            arr = _augment(arr, np.random.default_rng())
        return uint8_to_tensor(arr), int(self.labels[i])


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def get_dataset(split: str, mode: str = "auto", root: Path = PANNUKE_DIR, **kw) -> Dataset:
    """mode: 'real' | 'demo' | 'auto' (real when the three folds + splits.json exist)."""
    if mode == "auto":
        mode = "real" if (pannuke_available(root) and (Path(root) / SPLITS_FILE).exists()) else "demo"
    if mode == "real":
        return PanNukeDataset(split, root, **kw)
    return SyntheticPanNuke(split, **kw)
