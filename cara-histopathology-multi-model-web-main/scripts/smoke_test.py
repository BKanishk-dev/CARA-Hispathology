#!/usr/bin/env python
"""Quick self-check of the whole backend (needs PyTorch, no data):

    python scripts/smoke_test.py                  # all 8 models
    python scripts/smoke_test.py --models resnet18 vmamba

For every model it checks
  * input (B, 3, 256, 256) -> logits (B, 19) in train AND eval mode, finite values
  * predict_proba rows sum to 1
  * one optimisation step runs (backward + optimiser) - GAN model via fit_batch
  * a checkpoint round-trip through save_checkpoint / load_checkpoint reproduces the logits
and then exercises the CARA pipeline (fit on validation only, evaluate, serialise) and the split arithmetic.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import torch  # noqa: E402

from app.cara import CARA, CARAConfig  # noqa: E402
from app.dataset import split_sizes  # noqa: E402
from app.models import build_model, load_checkpoint, save_checkpoint  # noqa: E402
from app.utils import MODEL_KEYS, NUM_CLASSES, count_parameters, set_seed  # noqa: E402


def check_model(key: str) -> None:
    set_seed(0)
    model = build_model(key)
    n_params = count_parameters(model) / 1e6
    x = torch.randn(2, 3, 256, 256)
    y = torch.tensor([1, 7])

    t0 = time.time()
    model.train()
    out = model(x)
    assert out.shape == (2, NUM_CLASSES), f"train-mode shape {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "non-finite logits in train mode"

    model.eval()
    with torch.no_grad():
        out_eval = model(x)
    assert out_eval.shape == (2, NUM_CLASSES), f"eval-mode shape {tuple(out_eval.shape)}"
    assert torch.isfinite(out_eval).all(), "non-finite logits in eval mode"

    p = model.predict_proba(x)
    assert torch.allclose(p.sum(1), torch.ones(2), atol=1e-4), "softmax rows do not sum to 1"

    # one training step
    model.train()
    if key == "gan_classifier":
        opt_g = torch.optim.AdamW(model.g_parameters(), 1e-4)
        opt_d = torch.optim.AdamW(model.d_parameters(), 1e-4)
        logits, loss = model.fit_batch(x, y, opt_g, opt_d)
        assert logits.shape == (2, NUM_CLASSES) and np.isfinite(loss)
    else:
        opt = torch.optim.AdamW(model.parameters(), 1e-4)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y) + model.extra_loss(x, y, logits)
        opt.zero_grad()
        loss.backward()
        assert all(torch.isfinite(q.grad).all() for q in model.parameters() if q.grad is not None), "bad gradients"
        opt.step()

    # checkpoint round-trip
    model.eval()
    with torch.no_grad():
        ref = model(x)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / f"{key}.pth"
        save_checkpoint(model, path)
        clone = build_model(key)
        assert load_checkpoint(clone, path), "load_checkpoint returned False for a real checkpoint"
        clone.eval()
        with torch.no_grad():
            got = clone(x)
        assert torch.allclose(ref, got, atol=1e-4, rtol=1e-3), "checkpoint round-trip changed the logits"
    print(f"  OK  {key:<17} {n_params:6.2f} M params   {time.time() - t0:5.1f}s")


def check_cara() -> None:
    rng = np.random.default_rng(0)
    n, c = 800, NUM_CLASSES
    labels = rng.integers(0, c, n)
    probs = rng.dirichlet(np.full(c, 0.3), n)
    hit = rng.random(n) < 0.8
    probs[hit, :] *= 0.2
    probs[hit, labels[hit]] += 0.8
    probs /= probs.sum(1, keepdims=True)
    q = np.clip(rng.normal(0.65, 0.2, n), 0, 1)
    cara = CARA(CARAConfig(), c).fit(probs[:400], labels[:400], q[:400])          # validation only
    ev = cara.evaluate(probs[400:], labels[400:], q[400:])
    assert 0.0 <= ev["coverage"] <= 1.0 and abs(ev["coverage"] + ev["rejection_rate"] - 1) < 1e-9
    assert ev["selective_accuracy"] >= ev["accuracy"] - 1e-9, "selective accuracy should not be below accuracy"
    again = CARA.from_dict(cara.to_dict())
    assert np.allclose(again.thresholds, cara.thresholds)
    print(f"  OK  CARA             acc={ev['accuracy']:.3f} sel_acc={ev['selective_accuracy']:.3f} cov={ev['coverage']:.3f}")


def check_splits() -> None:
    s = split_sizes(2656 + 2523 + 2722)
    assert s == {"train": 5688, "val": 632, "test": 1581}, s
    print("  OK  split sizes      5688 / 632 / 1581")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=MODEL_KEYS, choices=MODEL_KEYS)
    args = ap.parse_args()
    failed = []
    print("Checking models ...")
    for k in args.models:
        try:
            check_model(k)
        except Exception:
            failed.append(k)
            print(f"  FAIL {k}")
            traceback.print_exc()
    for fn in (check_cara, check_splits):
        try:
            fn()
        except Exception:
            failed.append(fn.__name__)
            traceback.print_exc()
    print("\nAll checks passed." if not failed else f"\nFAILED: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
