"""Image quality score Q(x) in [0, 1] combining sharpness (blur) and H&E staining quality.

Q = w_blur * blur_score + w_stain * stain_score        (default 0.5 / 0.5)

blur_score   : variance of the Laplacian of the grayscale image, squashed to [0, 1].
stain_score  : mean of three sub-scores computed on tissue pixels in optical-density space
               (Ruifrok & Johnston H&E colour deconvolution):
                 * presence  - both haematoxylin and eosin are present
                 * intensity - stain strength is neither washed-out nor over-saturated
                 * contrast  - luminance contrast inside the tissue
"""
from __future__ import annotations

import numpy as np
from PIL import Image

# Reference H&E stain vectors (rows: haematoxylin, eosin, residual), unit-normalised.
_HE = np.array([[0.65, 0.70, 0.29],
                [0.07, 0.99, 0.11],
                [0.27, 0.57, 0.78]], dtype=np.float64)
_HE = _HE / np.linalg.norm(_HE, axis=1, keepdims=True)
_HE_INV = np.linalg.inv(_HE)

LAPLACIAN_REF = 100.0     # Laplacian variance at which sharpness reaches ~63 %
OD_TARGET = 0.60          # ideal optical-density magnitude of tissue
OD_SIGMA = 0.45
TISSUE_OD_MIN = 0.15      # pixels below this optical density are background


def _to_uint8_rgb(img) -> np.ndarray:
    if isinstance(img, Image.Image):
        return np.asarray(img.convert("RGB"), dtype=np.uint8)
    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def laplacian_variance(gray: np.ndarray) -> float:
    g = gray.astype(np.float64)
    lap = (-4.0 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(lap.var())


def blur_score(rgb: np.ndarray) -> tuple[float, float]:
    gray = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    var = laplacian_variance(gray)
    return float(1.0 - np.exp(-var / LAPLACIAN_REF)), var


def stain_score(rgb: np.ndarray) -> tuple[float, dict]:
    od = -np.log10((rgb.astype(np.float64) + 1.0) / 256.0)          # (H, W, 3)
    od_mag = np.linalg.norm(od, axis=-1)
    mask = od_mag > TISSUE_OD_MIN
    tissue_fraction = float(mask.mean())
    if mask.sum() < 50:                                              # almost blank image
        return 0.0, {"tissue_fraction": tissue_fraction, "presence": 0.0,
                     "intensity": 0.0, "contrast": 0.0}

    od_t = od[mask]
    conc = od_t @ _HE_INV                                            # [H, E, residual]
    h_mean = float(np.clip(conc[:, 0], 0, None).mean())
    e_mean = float(np.clip(conc[:, 1], 0, None).mean())
    presence = min(1.0, h_mean / 0.10) * min(1.0, e_mean / 0.10)

    mean_mag = float(od_mag[mask].mean())
    intensity = float(np.exp(-((mean_mag - OD_TARGET) / OD_SIGMA) ** 2))

    lum = (rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114)[mask] / 255.0
    contrast = float(min(1.0, lum.std() / 0.15))

    score = (presence + intensity + contrast) / 3.0
    return float(score), {"tissue_fraction": tissue_fraction, "presence": float(presence),
                          "intensity": intensity, "contrast": contrast,
                          "hematoxylin": h_mean, "eosin": e_mean}


def compute_quality(img, w_blur: float = 0.5, w_stain: float = 0.5) -> dict:
    """Return {'quality', 'blur', 'stain', 'details'} for a PIL image or uint8 array."""
    rgb = _to_uint8_rgb(img)
    b, lap_var = blur_score(rgb)
    s, det = stain_score(rgb)
    total = w_blur + w_stain
    q = (w_blur * b + w_stain * s) / total
    det = dict(det)
    det["laplacian_variance"] = lap_var
    return {"quality": float(q), "blur": float(b), "stain": float(s), "details": det}


def quality_scalar(img) -> float:
    return compute_quality(img)["quality"]
