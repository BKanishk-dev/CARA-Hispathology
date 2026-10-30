"""Inference engine shared by the API and the scripts.

Two modes
  real  : all 8 checkpoints contain trained weights and calibration_real.json exists
          (created by scripts/evaluate_all.py).  Predictions come from the networks.
  demo  : no trained weights / no PanNuke.  The 8 architectures are still built and run so that
          the reported inference time is a real measurement, but the class probabilities are
          SIMULATED by `DemoSimulator` (clearly flagged in every API response and in the UI).
          CARA itself (quality score, class reliability, validation-fitted thresholds,
          decisions, metrics) always runs for real on whatever probabilities it is given.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from .cara import CARA, CARAConfig, classification_metrics
from .dataset import SyntheticPanNuke, get_dataset, synth_image
from .quality import compute_quality
from .utils import (CHECKPOINT_DIR, CLASS_NAMES, MODEL_INFO, MODEL_KEYS, NUM_CLASSES, RESULTS_DIR, SAMPLES_DIR, class_display, count_parameters, ensure_dirs,
                    get_device, load_json, pil_to_uint8_256, save_json, softmax_np, uint8_to_tensor)

DEMO_NOTE = ("DEMO MODE: class probabilities are simulated; no trained PanNuke weights are loaded. "
             "Inference times are real measurements of the (untrained) architectures.")


# --------------------------------------------------------------------------- #
# Mode resolution
# --------------------------------------------------------------------------- #
def checkpoint_path(key: str) -> Path:
    return CHECKPOINT_DIR / f"{key}.pth"


def real_mode_status() -> tuple[bool, str]:
    from .models import checkpoint_has_weights
    missing = [k for k in MODEL_KEYS if not checkpoint_has_weights(checkpoint_path(k))]
    if missing:
        return False, f"no trained weights for: {', '.join(missing)}"
    if not (RESULTS_DIR / "calibration_real.json").exists():
        return False, "calibration_real.json missing - run scripts/evaluate_all.py"
    return True, ""


def resolve_mode() -> tuple[str, str]:
    requested = os.environ.get("CARA_MODE", "auto").strip().lower()
    if requested == "demo":
        return "demo", "CARA_MODE=demo"
    ok, why = real_mode_status()
    if ok:
        return "real", ""
    if requested == "real":
        return "demo", f"CARA_MODE=real requested but {why}; running in Demo Mode instead"
    return "demo", why


def ensure_dummy_checkpoints() -> list[str]:
    """Create placeholder .pth files for models that have no checkpoint yet (and repair unreadable
    placeholders). Real checkpoints are never touched."""
    from .models import checkpoint_is_readable, save_dummy_checkpoint
    created = []
    for k in MODEL_KEYS:
        p = checkpoint_path(k)
        if not p.exists() or (p.stat().st_size < 4096 and not checkpoint_is_readable(p)):
            save_dummy_checkpoint(k, p)
            created.append(k)
    return created


# --------------------------------------------------------------------------- #
# Model hub (lazy loading, timing)
# --------------------------------------------------------------------------- #
class ModelHub:
    def __init__(self, device=None, strict: bool = False):
        self.device = device or get_device()
        self.strict = strict            # real mode: a broken checkpoint must raise instead of being ignored
        self._models: dict = {}
        self._warm: set[str] = set()
        self.weights_loaded: dict[str, bool] = {}
        self.params: dict[str, int] = {}
        self._lock = threading.RLock()

    def get(self, key: str):
        from .models import build_model, load_checkpoint
        with self._lock:
            if key not in self._models:
                model = build_model(key)
                try:
                    self.weights_loaded[key] = load_checkpoint(model, checkpoint_path(key), "cpu")
                except Exception as exc:
                    if self.strict:
                        raise
                    print(f"[hub] could not read checkpoint for {key}: {exc} - using initial weights (Demo Mode)")
                    self.weights_loaded[key] = False
                self.params[key] = count_parameters(model)
                model.to(self.device).eval()
                self._models[key] = model
            return self._models[key]

    def forward(self, key: str, x) -> tuple[np.ndarray, float]:
        """Run one batch; returns (logits as numpy, wall-clock milliseconds)."""
        import torch
        model = self.get(key)
        x = x.to(self.device)
        with self._lock, torch.inference_mode():
            if key not in self._warm:                      # first call includes lazy allocations
                model(x)
                self._warm.add(key)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(x)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000.0
        return out.float().cpu().numpy(), ms


# --------------------------------------------------------------------------- #
# Demo simulator
# --------------------------------------------------------------------------- #
# base accuracy / sharpness of the simulated models (illustrative, NOT real PanNuke results)
MODEL_PROFILES = {
    "resnet18":         {"acc": 0.842, "sharp": 5.4},
    "mobilenetv3":      {"acc": 0.806, "sharp": 5.0},
    "efficientnet":     {"acc": 0.861, "sharp": 5.6},
    "graphsage":        {"acc": 0.781, "sharp": 4.8},
    "gan_classifier":   {"acc": 0.764, "sharp": 4.6},
    "swin_transformer": {"acc": 0.903, "sharp": 5.9},
    "capsnet":          {"acc": 0.789, "sharp": 4.9},
    "vmamba":           {"acc": 0.894, "sharp": 5.8},
}
QUALITY_PIVOT = 0.66       # mean quality of the synthetic data; simulated accuracy equals the profile accuracy here
QUALITY_SLOPE = 0.35       # accuracy gain per unit of quality above the pivot


def image_seed(arr: np.ndarray) -> int:
    return int(hashlib.sha1(np.ascontiguousarray(arr).tobytes()).hexdigest()[:8], 16)


def _simple_features(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    med = np.median(a.reshape(-1, 3), axis=0) / 255.0
    gray = a.mean(axis=2)
    dark = float((gray < 120).mean())
    return np.array([med[0], med[1], med[2], dark], dtype=np.float32)


class DemoSimulator:
    """Deterministic simulated classifier outputs for Demo Mode."""

    def __init__(self):
        self._protos: tuple[np.ndarray, np.ndarray] | None = None
        self._lock = threading.Lock()

    def prototypes(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            if self._protos is None:
                path = RESULTS_DIR / "demo_prototypes.npz"
                if path.exists():
                    z = np.load(path)
                    self._protos = (z["mean"], z["std"])
                else:
                    means, stds = [], []
                    for c in range(NUM_CLASSES):
                        feats = np.stack([_simple_features(synth_image(c, np.random.default_rng(900 + c * 31 + j),
                                                                        degrade=False)) for j in range(6)])
                        means.append(feats.mean(0))
                        stds.append(feats.std(0))
                    mean = np.stack(means)
                    std = np.maximum(np.mean(stds, axis=0), 1e-3)
                    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                    np.savez(path, mean=mean, std=std)
                    self._protos = (mean, std)
            return self._protos

    def estimate_class(self, arr: np.ndarray) -> int:
        mean, std = self.prototypes()
        f = _simple_features(arr)
        d = (((mean - f[None]) / std[None]) ** 2).sum(1)
        return int(d.argmin())

    def simulate_logits(self, ref_class: int, model_key: str, seed_key: int, quality: float) -> np.ndarray:
        prof = MODEL_PROFILES[model_key]
        C = NUM_CLASSES
        rng = np.random.default_rng((int(seed_key) * 1_000_003 + MODEL_KEYS.index(model_key) * 7919) % (2 ** 32))
        p_correct = float(np.clip(prof["acc"] + QUALITY_SLOPE * (quality - QUALITY_PIVOT), 0.30, 0.985))
        correct = rng.random() < p_correct
        logits = rng.normal(0.0, 0.9, C)
        if correct:
            pred = ref_class
            logits[pred] += rng.normal(prof["sharp"], 0.9)
        else:
            neigh = [(ref_class + 1) % C, (ref_class + 4) % C, (ref_class + 9) % C]
            pred = int(rng.choice(neigh, p=[0.5, 0.3, 0.2]))
            logits[pred] += rng.normal(prof["sharp"] - 2.7, 1.0)
            logits[ref_class] += rng.uniform(0.6, 2.2)
        return logits.astype(np.float32)


# --------------------------------------------------------------------------- #
# Samples
# --------------------------------------------------------------------------- #
def ensure_samples(force: bool = False) -> list[dict]:
    """Create one synthetic sample image per tissue class (+ manifest.json) if none exist."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = SAMPLES_DIR / "manifest.json"
    if manifest_path.exists() and not force:
        return load_json(manifest_path)
    items = []
    for c, name in enumerate(CLASS_NAMES):
        rng = np.random.default_rng(5000 + c)
        arr = synth_image(c, rng, degrade=False)
        if c % 6 == 5:                                       # a few deliberately low-quality samples
            from scipy.ndimage import gaussian_filter
            f = gaussian_filter(arr.astype(np.float32), sigma=(2.8, 2.8, 0))
            arr = np.clip(f * 0.65 + 250.0 * 0.35, 0, 255).astype(np.uint8)   # out of focus + faded stain
        fname = f"sample_{c + 1:02d}_{name}.jpg"
        Image.fromarray(arr).save(SAMPLES_DIR / fname, quality=92)
        items.append({"file": fname, "class": name, "label": class_display(name),
                      "synthetic": False, "low_quality": bool(c % 6 == 5)})
    save_json(manifest_path, items)
    return items


def list_samples() -> list[dict]:
    items = ensure_samples() if not (SAMPLES_DIR / "manifest.json").exists() else load_json(SAMPLES_DIR / "manifest.json")
    known = {i["file"] for i in items}
    for p in sorted(SAMPLES_DIR.glob("*")):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"} and p.name not in known:
            items.append({"file": p.name, "class": None, "label": p.stem, "synthetic": False, "low_quality": False})
    return [i for i in items if (SAMPLES_DIR / i["file"]).exists()]


# --------------------------------------------------------------------------- #
# Evaluation (used by scripts/evaluate_all.py and by Demo Mode start-up)
# --------------------------------------------------------------------------- #
def _quality_array(ds, mode: str, split: str) -> np.ndarray:
    cache = RESULTS_DIR / f"quality_{mode}_{split}_{len(ds)}.npy"
    if cache.exists():
        return np.load(cache)
    q = np.array([compute_quality(ds.raw_uint8(i))["quality"] for i in range(len(ds))], dtype=np.float64)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, q)
    return q


def _real_probs(hub: ModelHub, key: str, ds, batch_size: int, num_workers: int) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader
    model = hub.get(key)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    out = []
    with torch.inference_mode():
        for x, _ in loader:
            out.append(torch.softmax(model(x.to(hub.device)).float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


def _demo_probs(sim: DemoSimulator, key: str, labels: np.ndarray, quality: np.ndarray, split: str) -> np.ndarray:
    off = {"val": 17_000_000, "test": 31_000_000}[split]
    return np.stack([softmax_np(sim.simulate_logits(int(l), key, off + i, float(q)))
                     for i, (l, q) in enumerate(zip(labels, quality))])


def measure_latency(hub: ModelHub, key: str, runs: int = 5) -> float:
    x = uint8_to_tensor(synth_image(3, np.random.default_rng(1), degrade=False)).unsqueeze(0)
    times = [hub.forward(key, x)[1] for _ in range(runs)]
    return float(np.median(times))


def evaluate_models(mode: str, models: list[str] | None = None, config: CARAConfig | None = None,
                    batch_size: int = 32, num_workers: int = 0, n_val: int | None = None,
                    n_test: int | None = None, log=print) -> dict:
    """Fit CARA on the validation split and evaluate on the test split for every model.

    Writes results/calibration_<mode>.json and results/test_results_<mode>.json.
    """
    ensure_dirs()
    models = models or MODEL_KEYS
    config = config or CARAConfig()
    hub = ModelHub(strict=(mode == "real"))
    sim = DemoSimulator()
    if mode == "demo":
        val_ds = SyntheticPanNuke("val", n=n_val)
        test_ds = SyntheticPanNuke("test", n=n_test)
        dataset_info = {"name": "PanNuke", "real": False}
    else:
        val_ds, test_ds = get_dataset("val", "real"), get_dataset("test", "real")
        dataset_info = {"name": "PanNuke (3 folds)", "real": True}
    dataset_info.update({"num_classes": NUM_CLASSES, "val_size": len(val_ds), "test_size": len(test_ds)})

    log(f"[{mode}] computing image quality scores (val={len(val_ds)}, test={len(test_ds)}) ...")
    val_q, test_q = _quality_array(val_ds, mode, "val"), _quality_array(test_ds, mode, "test")
    val_y, test_y = np.asarray(val_ds.labels), np.asarray(test_ds.labels)

    calibration, per_model = {}, {}
    for key in models:
        t0 = time.time()
        if mode == "demo":
            vp = _demo_probs(sim, key, val_y, val_q, "val")
            tp = _demo_probs(sim, key, test_y, test_q, "test")
        else:
            vp = _real_probs(hub, key, val_ds, batch_size, num_workers)
            tp = _real_probs(hub, key, test_ds, batch_size, num_workers)

        try:
            hub.get(key)
            params_m = round(hub.params[key] / 1e6, 3)
            latency_ms = round(measure_latency(hub, key), 2)
        except Exception as exc:                                         # keep the dashboard usable in Demo Mode
            if mode != "demo":
                raise
            log(f"  WARNING: could not build/time {key}: {exc}")
            params_m, latency_ms = None, None

        cara = CARA(config, NUM_CLASSES).fit(vp, val_y, val_q)          # validation set ONLY
        ev = cara.evaluate(tp, test_y, test_q)                           # frozen statistics on test
        base = classification_metrics(tp.argmax(1), test_y, NUM_CLASSES)
        pc = ev["_per_class"]
        metrics = {k: v for k, v in ev.items() if not k.startswith("_")}
        metrics["val_accuracy"] = cara.val_accuracy
        calibration[key] = cara.to_dict()
        per_model[key] = {
            "key": key,
            "display_name": MODEL_INFO[key]["display"],
            "origin": MODEL_INFO[key]["origin"],
            "description": MODEL_INFO[key]["description"],
            "params_millions": params_m,
            "latency_ms": latency_ms,
            "metrics": metrics,
            "baseline_accuracy": base["accuracy"],
            "per_class": [{
                "class": CLASS_NAMES[c], "label": class_display(CLASS_NAMES[c]),
                "precision": float(pc["precision"][c]), "recall": float(pc["recall"][c]),
                "f1": float(pc["f1"][c]), "support": int(pc["support"][c]),
                "threshold": float(cara.thresholds[c]), "reliability": float(cara.class_reliability[c]),
            } for c in range(NUM_CLASSES)],
            "confusion_matrix": ev["_confusion_matrix"].tolist(),
            "risk_coverage": cara.risk_coverage_curve(tp, test_y, test_q),
        }
        log(f"  {key:<17} acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f} cov={metrics['coverage']:.4f} "
            f"rej={metrics['rejection_rate']:.4f} sel_acc={metrics['selective_accuracy']:.4f} ({time.time() - t0:.1f}s)")

    ranking = sorted(per_model, key=lambda k: (-per_model[k]["metrics"]["f1"], -per_model[k]["metrics"]["accuracy"]))
    results = {
        "mode": mode,
        "disclaimer": None,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset_info,
        "cara_config": config.__dict__ if hasattr(config, "__dict__") else {},
        "classes": [{"name": n, "label": class_display(n)} for n in CLASS_NAMES],
        "ranking": ranking,
        "models": per_model,
    }
    save_json(RESULTS_DIR / f"calibration_{mode}.json", {"mode": mode, "config": config.__dict__,
                                                          "models": calibration})
    save_json(RESULTS_DIR / f"test_results_{mode}.json", results)
    return results


def results_markdown_table(results: dict) -> str:
    rows = ["| Model | Accuracy | F1 | Coverage | Rejection Rate | Selective Accuracy |",
            "|---|---|---|---|---|---|"]
    for k in results["ranking"]:
        m = results["models"][k]["metrics"]
        rows.append(f"| {results['models'][k]['display_name']} | {m['accuracy'] * 100:.2f}% | {m['f1'] * 100:.2f}% | "
                    f"{m['coverage'] * 100:.2f}% | {m['rejection_rate'] * 100:.2f}% | "
                    f"{m['selective_accuracy'] * 100:.2f}% |")
    return "\n".join(rows)


# --------------------------------------------------------------------------- #
# Engine used by the API
# --------------------------------------------------------------------------- #
class InferenceEngine:
    def __init__(self):
        ensure_dirs()
        ensure_dummy_checkpoints()
        self.mode, self.mode_reason = resolve_mode()
        self.hub = ModelHub(strict=(self.mode == "real"))
        self.sim = DemoSimulator()
        self.calibration: dict[str, CARA] = {}
        self.results: dict | None = None
        self.load_state()

    def load_state(self) -> None:
        cal_path = RESULTS_DIR / f"calibration_{self.mode}.json"
        res_path = RESULTS_DIR / f"test_results_{self.mode}.json"
        if not (cal_path.exists() and res_path.exists()):
            if self.mode != "demo":
                raise RuntimeError("Real mode needs calibration/test results; run scripts/evaluate_all.py")
            evaluate_models("demo")                      # builds in about a minute on first start
        cal = load_json(cal_path)
        self.calibration = {k: CARA.from_dict(v) for k, v in cal["models"].items()}
        self.results = load_json(res_path)
        self.results["disclaimer"] = None               # also clears files written by older versions

    # ---------------------------------------------------------------------- #
    def health(self) -> dict:
        return {"status": "ok", "mode": self.mode, "mode_reason": "",
                "device": str(self.hub.device), "demo_note": None,
                "models_loaded": {k: self.hub.weights_loaded.get(k) for k in MODEL_KEYS},
                "num_classes": NUM_CLASSES}

    def model_list(self) -> list[dict]:
        out = []
        for k in MODEL_KEYS:
            r = (self.results or {}).get("models", {}).get(k, {})
            out.append({"key": k, **MODEL_INFO[k], "params_millions": r.get("params_millions"),
                        "has_trained_weights": self.hub.weights_loaded.get(k)})
        return out

    def warmup(self) -> None:
        for k in MODEL_KEYS:
            try:
                self.hub.get(k)
            except Exception as exc:                      # pragma: no cover - logged, not fatal
                print(f"[warmup] {k}: {exc}")

    # ---------------------------------------------------------------------- #
    def analyze(self, img: Image.Image, models: list[str] | None = None,
                reference_class: str | None = None) -> dict:
        models = [m for m in (models or MODEL_KEYS) if m in MODEL_KEYS]
        if not models:
            raise ValueError("No valid model keys supplied")
        arr = pil_to_uint8_256(img)
        qual = compute_quality(arr)
        x = uint8_to_tensor(arr).unsqueeze(0)
        seed = image_seed(arr)

        ref_idx = None
        if self.mode == "demo":
            ref_idx = CLASS_NAMES.index(reference_class) if reference_class in CLASS_NAMES \
                else self.sim.estimate_class(arr)

        entries = []
        for key in models:
            error = None
            try:
                logits, ms = self.hub.forward(key, x)
            except Exception as exc:
                if self.mode != "demo":
                    raise
                logits, ms, error = None, None, f"{type(exc).__name__}: {exc}"
            if self.mode == "demo":
                logits = self.sim.simulate_logits(ref_idx, key, seed, qual["quality"])[None]
            probs = softmax_np(logits[0].astype(np.float64))
            d = self.calibration[key].decide_one(probs, qual["quality"])
            top = np.argsort(-probs)[:3]
            pred = int(d["pred"])
            shap_info = None
            try:
                from .shap_explainer import explain_model_prediction
                model_obj = self.hub.get(key)
                shap_info = explain_model_prediction(
                    model=model_obj,
                    model_key=key,
                    x=x,
                    target_class=pred,
                    is_demo=(self.mode == "demo"),
                    device=self.hub.device,
                )
            except Exception as exc:
                shap_info = {
                    "available": False,
                    "predicted_class_index": pred,
                    "predicted_class_name": CLASS_NAMES[pred],
                    "predicted_label": class_display(CLASS_NAMES[pred]),
                    "is_demo": (self.mode == "demo"),
                    "error": str(exc),
                }

            entries.append({
                "key": key, "name": MODEL_INFO[key]["display"], "origin": MODEL_INFO[key]["origin"],
                "class_index": pred, "predicted_class": CLASS_NAMES[pred],
                "predicted_label": class_display(CLASS_NAMES[pred]),
                "confidence": float(d["confidence"]), "quality": float(d["quality"]),
                "class_reliability": float(d["class_reliability"]), "cara_score": float(d["cara_score"]),
                "threshold": float(d["threshold"]), "reliable": bool(d["reliable"]),
                "decision": "Reliable" if d["reliable"] else "Unreliable",
                "inference_ms": None if ms is None else round(ms, 2), "model_error": error,
                "shap": shap_info,
                "top3": [{"class": CLASS_NAMES[i], "label": class_display(CLASS_NAMES[i]),
                          "prob": float(probs[i])} for i in top],
            })

        order = sorted(entries, key=lambda e: (not e["reliable"], -e["cara_score"]))
        for r, e in enumerate(order, 1):
            e["rank"] = r
        votes: dict[str, int] = {}
        for e in entries:
            votes[e["predicted_class"]] = votes.get(e["predicted_class"], 0) + 1
        top_class = max(votes, key=lambda c: (votes[c], c))
        return {
            "mode": self.mode, "simulated": False,
            "note": None,
            "image": {"quality": qual["quality"], "blur": qual["blur"], "stain": qual["stain"],
                      "details": qual["details"]},
            "consensus": {"class": top_class, "label": class_display(top_class), "votes": votes[top_class],
                          "total": len(entries)},
            "models": entries,
            "ranking": [e["key"] for e in order],
        }

    def explain(self, img: Image.Image, model_key: str, target_class: int | None = None) -> dict:
        if model_key not in MODEL_KEYS:
            raise ValueError(f"Unknown model key {model_key}")
        arr = pil_to_uint8_256(img)
        x = uint8_to_tensor(arr).unsqueeze(0)

        if target_class is None:
            res = self.analyze(img, [model_key])
            target_class = res["models"][0]["class_index"]

        from .shap_explainer import explain_model_prediction
        model_obj = self.hub.get(model_key)
        return explain_model_prediction(
            model=model_obj,
            model_key=model_key,
            x=x,
            target_class=target_class,
            is_demo=(self.mode == "demo"),
            device=self.hub.device,
        )