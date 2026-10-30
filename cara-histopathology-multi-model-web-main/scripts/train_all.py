#!/usr/bin/env python
"""Train all (or selected) models and save checkpoints to backend/checkpoints/<model>.pth.

    python scripts/train_all.py                          # real PanNuke data, all 8 models
    python scripts/train_all.py --models resnet18 swin_transformer --epochs 30
    python scripts/train_all.py --demo --epochs 1 --max-train 256 --max-val 128   # smoke test (synthetic data)
    python scripts/train_all.py --dummy-checkpoints      # only write placeholder .pth files

The checkpoint with the best validation accuracy is kept.  Only the training split is used for
gradient updates; the validation split is used for model selection (and later for CARA calibration).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from app.dataset import get_dataset  # noqa: E402
from app.inference import checkpoint_path, ensure_dummy_checkpoints  # noqa: E402
from app.models import build_model, save_checkpoint  # noqa: E402
from app.utils import MODEL_KEYS, count_parameters, get_device, set_seed  # noqa: E402

# per-model defaults: (learning rate, weight decay, optimiser)
HPARAMS = {
    "resnet18": (1e-3, 1e-4, "adamw"),
    "mobilenetv3": (2e-3, 1e-4, "adamw"),
    "efficientnet": (1e-3, 1e-4, "adamw"),
    "graphsage": (1e-3, 1e-4, "adamw"),
    "gan_classifier": (5e-4, 1e-4, "adamw"),
    "swin_transformer": (3e-4, 5e-2, "adamw"),
    "capsnet": (1e-3, 1e-5, "adamw"),
    "vmamba": (5e-4, 5e-2, "adamw"),
}


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    ce, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        ce += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return ce / max(n, 1), correct / max(n, 1)


def train_one(key: str, train_ds, val_ds, args, device) -> dict:
    set_seed(args.seed)
    model = build_model(key).to(device)
    print(f"\n=== {key}  ({count_parameters(model) / 1e6:.2f} M parameters) ===")
    lr, wd, _ = HPARAMS[key]
    lr = args.lr or lr
    is_gan = key == "gan_classifier"
    if is_gan:
        opt_g = torch.optim.AdamW(model.g_parameters(), lr=lr, weight_decay=wd, betas=(0.5, 0.999))
        opt_d = torch.optim.AdamW(model.d_parameters(), lr=lr, weight_decay=wd, betas=(0.5, 0.999))
        opts = [opt_g, opt_d]
    else:
        opts = [torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)]
    steps = max(1, args.epochs * max(1, len(train_ds) // args.batch_size))
    scheds = [torch.optim.lr_scheduler.OneCycleLR(o, max_lr=lr, total_steps=steps, pct_start=0.1) for o in opts]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    # mixed precision is skipped for models with exp()/routing/adversarial maths that prefer fp32
    use_amp = device.type == "cuda" and not args.no_amp and key not in ("gan_classifier", "vmamba", "capsnet")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_acc, best_epoch, t_start = -1.0, -1, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, run_correct, seen, t0 = 0.0, 0, 0, time.time()
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if is_gan:
                logits, loss_val = model.fit_batch(x, y, opts[0], opts[1])
                run_loss += loss_val * len(y)
            else:
                opts[0].zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    logits = model(x)
                    loss = nn.functional.cross_entropy(logits.float(), y, label_smoothing=0.05)
                    loss = loss + model.extra_loss(x, y, logits.float())
                scaler.scale(loss).backward()
                scaler.unscale_(opts[0])
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opts[0])
                scaler.update()
                run_loss += loss.item() * len(y)
            for s in scheds:
                s.step()
            run_correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
        val_loss, val_acc = evaluate(model, val_loader, device)
        flag = ""
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            save_checkpoint(model.cpu(), checkpoint_path(key), val_accuracy=val_acc, epoch=epoch)
            model.to(device)
            flag = "  * saved"
        print(f"epoch {epoch:>3}/{args.epochs}  loss {run_loss / max(seen, 1):.4f}  "
              f"train_acc {run_correct / max(seen, 1):.4f}  val_loss {val_loss:.4f}  val_acc {val_acc:.4f}  "
              f"({time.time() - t0:.0f}s){flag}")
    return {"model": key, "best_val_acc": best_acc, "best_epoch": best_epoch, "seconds": time.time() - t_start}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=MODEL_KEYS, choices=MODEL_KEYS)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--demo", action="store_true", help="train on the synthetic dataset instead of PanNuke")
    ap.add_argument("--max-train", type=int, default=None, help="limit training images (smoke tests)")
    ap.add_argument("--max-val", type=int, default=None)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--dummy-checkpoints", action="store_true", help="only create placeholder .pth files")
    args = ap.parse_args()

    if args.dummy_checkpoints:
        print("created placeholders for:", ensure_dummy_checkpoints() or "nothing (all exist)")
        return 0

    device = get_device()
    mode = "demo" if args.demo else "real"
    train_ds, val_ds = get_dataset("train", mode), get_dataset("val", mode)
    if args.max_train:
        train_ds = Subset(train_ds, np.arange(min(args.max_train, len(train_ds))))
    if args.max_val:
        val_ds = Subset(val_ds, np.arange(min(args.max_val, len(val_ds))))
    print(f"device={device}  mode={mode}  train={len(train_ds)}  val={len(val_ds)}")

    summary = [train_one(k, train_ds, val_ds, args, device) for k in args.models]
    print("\nSummary")
    for s in summary:
        print(f"  {s['model']:<17} best val acc {s['best_val_acc']:.4f} (epoch {s['best_epoch']}, {s['seconds']:.0f}s)")
    print("\nNext: python scripts/evaluate_all.py   (fits CARA on the validation split, evaluates on test)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
