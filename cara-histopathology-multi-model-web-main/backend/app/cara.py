"""CARA - Confidence-Aware Reliability Assessment.

Pipeline
  1. C(x)  softmax confidence of the predicted class
  2. Q(x)  image quality score (blur + staining), see quality.py
  3. S_c   class-wise reliability = smoothed validation precision of class c
  4. R(x)  = alpha * C + beta * Q + gamma * S_pred          (weights normalised to sum to 1)
  5. tau_c class-specific adaptive thresholds, derived ONLY from the validation set
  6. decision: Reliable if R(x) >= tau_pred else Unreliable (flag for expert review)

Nothing in this module ever looks at test data when fitting: `fit` takes validation
arrays only, and `evaluate` / `decide` merely apply the frozen statistics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class CARAConfig:
    alpha: float = 0.5            # weight of softmax confidence
    beta: float = 0.2             # weight of image quality
    gamma: float = 0.3            # weight of class reliability
    smoothing: float = 2.0        # pseudo-count pulling S_c towards the global accuracy
    min_class_samples: int = 5    # below this, a class falls back to the global threshold
    min_keep_fraction: float = 0.3  # a class threshold must keep at least this share of its val samples
    fallback_quantile: float = 0.25  # fallback: keep 75 % of correctly classified val samples
    target_error_reduction: float = 0.5  # target selective acc = val_acc + this * (1 - val_acc)

    def normalised(self) -> tuple[float, float, float]:
        s = self.alpha + self.beta + self.gamma
        return self.alpha / s, self.beta / s, self.gamma / s


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def confusion_matrix(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (labels.astype(int), preds.astype(int)), 1)
    return cm


def classification_metrics(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> dict:
    """Accuracy and macro / weighted precision, recall, F1 (plus per-class values)."""
    cm = confusion_matrix(preds, labels, num_classes)
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    present = support > 0
    w = support / max(support.sum(), 1.0)
    return {
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "precision": float(precision[present].mean()) if present.any() else 0.0,
        "recall": float(recall[present].mean()) if present.any() else 0.0,
        "f1": float(f1[present].mean()) if present.any() else 0.0,
        "weighted_f1": float((f1 * w).sum()),
        "per_class": {"precision": precision, "recall": recall, "f1": f1, "support": support},
        "confusion_matrix": cm,
    }


# --------------------------------------------------------------------------- #
# CARA
# --------------------------------------------------------------------------- #
class CARA:
    def __init__(self, config: CARAConfig | None = None, num_classes: int = 19):
        self.config = config or CARAConfig()
        self.num_classes = num_classes
        self.class_reliability = np.full(num_classes, 0.5)
        self.thresholds = np.full(num_classes, 0.5)
        self.global_threshold = 0.5
        self.val_accuracy = 0.0
        self.target_selective_accuracy = 0.0
        self.fitted = False

    # ----------------------------- scores ---------------------------------- #
    def scores(self, probs: np.ndarray, quality: np.ndarray) -> dict:
        probs = np.atleast_2d(np.asarray(probs, dtype=np.float64))
        quality = np.atleast_1d(np.asarray(quality, dtype=np.float64))
        pred = probs.argmax(axis=1)
        conf = probs.max(axis=1)
        s = self.class_reliability[pred]
        a, b, g = self.config.normalised()
        r = a * conf + b * quality + g * s
        return {"pred": pred, "confidence": conf, "quality": quality,
                "class_reliability": s, "cara_score": r}

    # ------------------------------- fit ----------------------------------- #
    def fit(self, val_probs: np.ndarray, val_labels: np.ndarray, val_quality: np.ndarray) -> "CARA":
        """Compute class reliability S_c and adaptive thresholds tau_c from validation data only."""
        cfg = self.config
        C = self.num_classes
        val_probs = np.asarray(val_probs, dtype=np.float64)
        val_labels = np.asarray(val_labels).astype(int)
        pred = val_probs.argmax(axis=1)
        correct = pred == val_labels
        self.val_accuracy = float(correct.mean())

        # class-wise reliability: smoothed precision of "predicted as c"
        rel = np.zeros(C)
        for c in range(C):
            m = pred == c
            n_c, tp_c = int(m.sum()), int((m & correct).sum())
            rel[c] = (tp_c + cfg.smoothing * self.val_accuracy) / (n_c + cfg.smoothing)
        self.class_reliability = rel

        r = self.scores(val_probs, val_quality)["cara_score"]
        self.target_selective_accuracy = float(min(
            0.995, self.val_accuracy + cfg.target_error_reduction * (1.0 - self.val_accuracy)))

        self.global_threshold = self._threshold_for(r, correct, cfg.min_keep_fraction)
        thr = np.full(C, self.global_threshold)
        for c in range(C):
            m = pred == c
            if m.sum() >= cfg.min_class_samples:
                thr[c] = self._threshold_for(r[m], correct[m], cfg.min_keep_fraction)
        self.thresholds = thr
        self.fitted = True
        return self

    def _threshold_for(self, r: np.ndarray, correct: np.ndarray, min_keep: float) -> float:
        """Lowest threshold whose kept subset reaches the target precision while keeping
        at least `min_keep` of the samples; otherwise fall back to a quantile of the
        scores of correctly classified samples."""
        n = len(r)
        order = np.argsort(-r)
        r_sorted, c_sorted = r[order], correct[order].astype(np.float64)
        prec = np.cumsum(c_sorted) / np.arange(1, n + 1)
        min_n = max(1, int(np.ceil(min_keep * n)))
        ok = np.where((prec >= self.target_selective_accuracy) & (np.arange(1, n + 1) >= min_n))[0]
        if len(ok):
            k = ok.max()                                    # largest kept set that still meets target
            nxt = r_sorted[k + 1] if k + 1 < n else r_sorted[k] - 1e-6
            return float(0.5 * (r_sorted[k] + nxt)) if k + 1 < n else float(r_sorted[k] - 1e-9)
        if correct.any():
            return float(np.quantile(r[correct], self.config.fallback_quantile))
        return float(np.quantile(r, 0.5))

    # ------------------------------ decide --------------------------------- #
    def decide(self, probs: np.ndarray, quality: np.ndarray) -> dict:
        out = self.scores(probs, quality)
        tau = self.thresholds[out["pred"]]
        out["threshold"] = tau
        out["reliable"] = out["cara_score"] >= tau
        return out

    def decide_one(self, prob: np.ndarray, quality: float) -> dict:
        d = self.decide(np.asarray(prob)[None, :], np.array([quality]))
        return {k: (v[0].item() if hasattr(v[0], "item") else v[0]) for k, v in d.items()}

    # ----------------------------- evaluate -------------------------------- #
    def evaluate(self, probs: np.ndarray, labels: np.ndarray, quality: np.ndarray) -> dict:
        """Accuracy / P / R / F1 plus Coverage, Rejection Rate and Selective Accuracy."""
        labels = np.asarray(labels).astype(int)
        d = self.decide(probs, quality)
        m = classification_metrics(d["pred"], labels, self.num_classes)
        reliable = d["reliable"]
        correct = d["pred"] == labels
        coverage = float(reliable.mean())
        selective_acc = float(correct[reliable].mean()) if reliable.any() else 0.0
        errors = ~correct
        out = {
            "accuracy": m["accuracy"], "precision": m["precision"],
            "recall": m["recall"], "f1": m["f1"], "weighted_f1": m["weighted_f1"],
            "coverage": coverage, "rejection_rate": 1.0 - coverage,
            "selective_accuracy": selective_acc,
            "error_flag_rate": float((~reliable & errors).sum() / max(errors.sum(), 1)),
            "num_samples": int(len(labels)),
            "mean_confidence": float(d["confidence"].mean()),
            "mean_quality": float(d["quality"].mean()),
            "mean_cara_score": float(d["cara_score"].mean()),
        }
        # selective metrics restricted to the accepted subset
        if reliable.any():
            sm = classification_metrics(d["pred"][reliable], labels[reliable], self.num_classes)
            out["selective_f1"] = sm["f1"]
        else:
            out["selective_f1"] = 0.0
        out["_per_class"] = m["per_class"]
        out["_confusion_matrix"] = m["confusion_matrix"]
        out["_decision"] = d
        return out

    def risk_coverage_curve(self, probs: np.ndarray, labels: np.ndarray, quality: np.ndarray,
                            points: int = 25) -> list[dict]:
        """Selective accuracy as the CARA score cut-off is raised (global, for plotting)."""
        d = self.scores(probs, quality)
        correct = d["pred"] == np.asarray(labels).astype(int)
        order = np.argsort(-d["cara_score"])
        n = len(order)
        curve = []
        for frac in np.linspace(0.1, 1.0, points):
            k = max(1, int(round(frac * n)))
            curve.append({"coverage": float(k / n), "accuracy": float(correct[order[:k]].mean())})
        return curve

    # ---------------------------- (de)serialise ---------------------------- #
    def to_dict(self) -> dict:
        return {
            "config": asdict(self.config),
            "num_classes": self.num_classes,
            "class_reliability": self.class_reliability.tolist(),
            "thresholds": self.thresholds.tolist(),
            "global_threshold": float(self.global_threshold),
            "val_accuracy": float(self.val_accuracy),
            "target_selective_accuracy": float(self.target_selective_accuracy),
            "fitted_on": "validation",
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CARA":
        obj = cls(CARAConfig(**d["config"]), d["num_classes"])
        obj.class_reliability = np.asarray(d["class_reliability"], dtype=np.float64)
        obj.thresholds = np.asarray(d["thresholds"], dtype=np.float64)
        obj.global_threshold = float(d["global_threshold"])
        obj.val_accuracy = float(d.get("val_accuracy", 0.0))
        obj.target_selective_accuracy = float(d.get("target_selective_accuracy", 0.0))
        obj.fitted = True
        return obj
