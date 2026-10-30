# Checkpoints

One `<model>.pth` per model: `resnet18`, `mobilenetv3`, `efficientnet`, `graphsage`, `gan_classifier`,
`swin_transformer`, `capsnet`, `vmamba`.

Missing files are created automatically as tiny placeholders when the backend starts (or with
`python scripts/train_all.py --dummy-checkpoints`). Placeholders carry no weights and keep the app in Demo Mode.
`python scripts/train_all.py` overwrites them with real checkpoints.
