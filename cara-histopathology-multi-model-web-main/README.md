# CARA – Confidence-Aware Histopathology Image Analysis (8 models + web app)

Upload a histopathology image and see side-by-side predictions and **reliability scores** from eight deep-learning
models. Every prediction is passed through the **CARA** framework (confidence + image quality + class-wise
reliability, with class-specific thresholds fitted on the validation set only) and is marked **Reliable** or
**Unreliable** (flag for expert review).

| | |
|---|---|
| Models | ResNet18 · MobileNetV3-Small · EfficientNet-B0 · GraphSAGE · GAN classifier · Swin-Tiny · CapsNet · VMamba |
| Data | PanNuke, 3 folds, 19 tissue classes (5,688 train / 632 val / 1,581 test) |
| Backend | FastAPI + PyTorch |
| Frontend | React + Vite + Tailwind CSS + Recharts |

> **Read this first – two modes**
>
> * **Real mode** – all 8 checkpoints contain trained weights and `calibration_real.json` exists. Predictions come
>   from the networks; the dashboard shows real PanNuke test-set results.
> * **Demo Mode** (default on a fresh checkout) – no PanNuke and no trained weights are needed. The 8 architectures
>   are still built and executed, so the **inference times are real measurements**, but the **class probabilities are
>   simulated** (`backend/app/inference.py → DemoSimulator`) and the dataset is a synthetic H&E-like generator.
>   The CARA pipeline itself (quality score, class reliability, validation-fitted thresholds, decisions, metrics)
>   runs for real on those probabilities. The UI shows a banner in this mode and **the numbers are not PanNuke
>   results.**

---

## 1. Quick start

### Option A – Docker

```bash
cp .env.example .env          # optional
docker-compose up --build     # (or: docker compose up --build)
```

* Web UI: <http://localhost:3000>  (nginx, proxies `/api` to the backend)
* API docs: <http://localhost:8000/docs>

The first start takes about a minute: the backend builds the 8 models and pre-computes the Demo-Mode dashboard.
`CARA_PRELOAD=1` keeps the models warm after that.

### Option B – run locally

Backend (Python 3.10+):

```bash
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload                            # http://localhost:8000
```

Frontend (Node 18+), in a second terminal:

```bash
cd frontend
npm install
npm run dev                                              # http://localhost:5173
```

The Vite dev server proxies `/api` and `/samples` to `http://localhost:8000`
(override with `VITE_API_TARGET=http://host:port npm run dev`).

---

## 2. Using the web app

* **Analyze** – drag & drop an image (or click a sample), press **Compare All Models**. You get one card per model
  (predicted class, softmax confidence, image quality, class reliability, CARA score with its class threshold,
  Reliable/Unreliable decision, inference time), a sortable ranking table and a confidence-vs-CARA chart.
  **Export CSV / PDF** is available once results are shown.
* **Dashboard** – pre-computed test-set results: accuracy, precision, recall, F1, coverage, rejection rate,
  selective accuracy, parameters, latency, risk–coverage curves, per-class thresholds and confusion matrices.
* **Gallery** – one sample image per tissue class; click to analyze. Put your own images in `data/samples/`.
* **Dark / light mode** toggle in the header (remembered in the browser).

Images are centre-cropped to a square and resized to 256 × 256 before inference.

---

## 3. Using real PanNuke data

### 3.1 Place the three folds

Download PanNuke (<https://warwick.ac.uk/fac/cross_fac/tia/data/pannuke>) and put the three folds under
`data/pannuke/`. The official archive layout works as-is:

```
data/pannuke/
├── Fold 1/images/fold1/images.npy     (2,656 images)   + types.npy
├── Fold 2/images/fold2/images.npy     (2,523 images)   + types.npy
└── Fold 3/images/fold3/images.npy     (2,722 images)   + types.npy
```

`masks.npy` is not needed. Any folder layout is accepted as long as, for each fold N, a directory whose name contains
`fold N` (any case, with or without space/underscore) holds `images.npy` and `types.npy`.
With Docker the folder is mounted into the container automatically (`./data/pannuke`).

### 3.2 Create the split

```bash
python scripts/prepare_pannuke.py
```

Writes `data/pannuke/splits.json`: **5,688 train / 632 validation / 1,581 test** (7,901 images), stratified by
tissue type with seed 42. If your copy has a different number of images the split is scaled proportionally.

### 3.0 Self-check (recommended first step)

```bash
python scripts/smoke_test.py      # every model: (B,3,256,256) -> (B,19) in train/eval mode, one training step, checkpoint round-trip; CARA; split sizes
```

### 3.3 Train, calibrate, evaluate

```bash
python scripts/train_all.py                                   # all 8 models, 30 epochs, best val-acc checkpoint kept
python scripts/train_all.py --models resnet18 swin_transformer --epochs 40 --batch-size 32
python scripts/evaluate_all.py --update-readme                # fits CARA on val, evaluates on test, refreshes table below
```

Checkpoints are written to `backend/checkpoints/<model>.pth`. When all eight contain weights and
`evaluate_all.py` has produced `backend/results/calibration_real.json`, the API switches to **real mode**
automatically (`CARA_MODE=auto`). Force a mode with `CARA_MODE=demo|real`.

Smoke test of the training code on synthetic data (no PanNuke needed):

```bash
python scripts/train_all.py --demo --epochs 1 --max-train 128 --max-val 64 --models resnet18 capsnet
```

> That command writes real (but meaningless) weights for the models you pass. Delete those `.pth` files, or run
> `python scripts/train_all.py --dummy-checkpoints` after removing them, to return to placeholder files.

### 3.4 Checkpoints

`backend/checkpoints/` ships with one tiny **placeholder** `.pth` per model (`{"dummy": True, "state_dict": None}`);
placeholders are ignored when loading and keep the app in Demo Mode. If PyTorch cannot read a placeholder, or one is
missing, the backend re-creates it with `torch.save` on start-up (`python scripts/train_all.py --dummy-checkpoints`
does the same). Real checkpoints written by `train_all.py` are plain dictionaries
`{"model", "state_dict", "num_classes", ...}` and load with `app.models.load_checkpoint(model, path)`.

---

## 4. The CARA framework

For an image *x* and a model with softmax probabilities *p*:

1. **C(x)** – softmax confidence `max(p)` of the predicted class *c*.
2. **Q(x)** – image quality in [0, 1] (`backend/app/quality.py`): `0.5·sharpness + 0.5·staining`.
   Sharpness is the Laplacian variance squashed to [0, 1]; staining averages haematoxylin/eosin *presence*,
   optical-density *intensity* and luminance *contrast* on tissue pixels (Ruifrok–Johnston colour deconvolution).
3. **S_c** – class-wise reliability from the **validation set only**: smoothed precision of "predicted as *c*".
4. **R(x) = α·C + β·Q + γ·S_c** (defaults α = 0.5, β = 0.2, γ = 0.3, normalised to sum to 1).
5. **Class-specific adaptive thresholds τ_c**, again from the **validation set only**: the lowest threshold whose kept
   validation predictions of class *c* reach a target selective accuracy (validation accuracy plus half of the
   remaining error), keeping at least 30 % of that class; sparse classes fall back to a global threshold, and
   classes that cannot reach the target fall back to a quantile of the correctly classified scores.
6. **Decision** – *Reliable* if `R(x) ≥ τ_c`, otherwise *Unreliable* (flag for expert review).

Metrics (`CARA.evaluate`): accuracy, macro precision/recall/F1, **coverage** (share of predictions accepted),
**rejection rate** (1 − coverage) and **selective accuracy** (accuracy on accepted predictions). The test split is
never used to fit anything – `CARA.fit` only receives validation arrays.

---

## 5. The eight models

All models derive from `BaseClassifier` (`backend/app/utils.py`), take `(B, 3, 256, 256)` and return logits
`(B, 19)` in both train and eval mode, expose `predict_proba`, and are created through
`app.models.build_model(name)`. All are implemented from scratch in PyTorch (no pretrained weights, no `torchvision`).

| Key | Model | Notes |
|---|---|---|
| `resnet18` | ResNet18 | BasicBlock × [2,2,2,2] |
| `mobilenetv3` | MobileNetV3-Small | inverted residuals, SE, h-swish |
| `efficientnet` | EfficientNet-B0 | MBConv + SE + SiLU, stochastic depth |
| `graphsage` | GraphSAGE | 8×8 patch graph (8-neighbour grid), patch-CNN node features, 3 SAGE layers, mean+max read-out |
| `gan_classifier` | GAN classifier | generator-style encoder–decoder as feature extractor; discriminator with adversarial and classification heads; `fit_batch` trains with CE + L1 reconstruction + hinge GAN loss |
| `swin_transformer` | Swin-Tiny | patch 4, dim 96, depths 2-2-6-2, heads 3-6-12-24, window 8, shifted windows |
| `capsnet` | CapsNet | conv stem → 16 primary capsule types (8-D, 1,024 capsules) → 19 class capsules (16-D), 3 routing iterations; margin loss via `extra_loss` |
| `vmamba` | VMamba ("lite") | VSS blocks with SS2D 4-direction selective scan, written in pure PyTorch (chunked closed-form scan); 8×8 patch embedding, dims 64-128-256-384, depths 2-2-4-2 |

Parameter counts and measured latency are shown in the dashboard.

---

## 6. Comparison of the 8 models (test split)

<!-- RESULTS_TABLE_START -->
**Demo Mode numbers (simulated probabilities on a synthetic dataset – NOT PanNuke results):**

| Model | Accuracy | F1 | Coverage | Rejection Rate | Selective Accuracy |
|---|---|---|---|---|---|
| Swin-Tiny | 91.90% | 91.70% | 90.20% | 9.80% | 95.93% |
| VMamba | 91.21% | 91.10% | 89.94% | 10.06% | 95.57% |
| EfficientNet-B0 | 86.84% | 86.53% | 86.08% | 13.92% | 93.02% |
| ResNet18 | 85.90% | 85.50% | 84.00% | 16.00% | 93.07% |
| MobileNetV3-Small | 83.87% | 83.42% | 81.40% | 18.60% | 91.45% |
| GraphSAGE | 82.73% | 82.49% | 81.02% | 18.98% | 90.48% |
| CapsNet | 82.54% | 82.28% | 79.89% | 20.11% | 92.24% |
| GAN Classifier | 80.52% | 79.83% | 77.10% | 22.90% | 92.12% |
<!-- RESULTS_TABLE_END -->

The table above comes from `python scripts/evaluate_all.py --demo --update-readme`: the real CARA code run on
**simulated** model outputs for a synthetic dataset. It demonstrates the pipeline (selective accuracy exceeds plain
accuracy at 77–90 % coverage) and says nothing about how the architectures perform on PanNuke. After training on
PanNuke, run `python scripts/evaluate_all.py --update-readme` to replace it with real numbers.

---

## 7. REST API

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | mode, device, whether trained weights are loaded |
| GET | `/api/models` | model metadata |
| GET | `/api/classes` | the 19 tissue classes |
| POST | `/api/compare` | multipart `file` (+ optional `reference_class`, Demo Mode only) → results of all 8 models including SHAP attributions |
| POST | `/api/predict/{model_key}` | same for a single model |
| POST | `/api/explain/{model_key}` | generate SHAP attribution for specified model on uploaded image |
| GET | `/api/results` | pre-computed test-set results (dashboard) |
| GET | `/api/samples`, `/samples/<file>` | gallery manifest and images |

Environment variables are listed in `.env.example`.

---

## 8. SHAP Explainable AI (XAI)

The Analyze page provides SHAP (SHapley Additive exPlanations) visual attributions for each model:

### 8.1 Method & Implementation
* **Explainer**: `shap.GradientExplainer` (Expected Gradients / Integrated Gradients over a background distribution).
* **Target**: Attributions are computed directly for the model's **actually predicted class** $k$ by wrapping the network with a single-target gradient projection (`SingleClassWrapper`). This accelerates gradient backward propagation by up to 19× compared to computing all-class gradients.
* **Background Set**: A small, documented set of representative baseline histopathology patches from `data/samples/` (with fallback to neutral baseline) is used to compute expected gradient expectations.
* **Attribution Map**: The 3-channel input attribution is aggregated into a 2D spatial heatmap with 99th-percentile scaling. Regions are encoded into an RGBA overlay:
  * **Red (#ef4444)**: Positive attribution — image morphology driving the model towards predicting the indicated class.
  * **Blue (#3b82f6)**: Negative attribution — image morphology penalizing or opposing the prediction.
  * **Transparent**: Low influence / background tissue.
* **Configuration**:
  * `CARA_SHAP_NSAMPLES`: Integration samples per explanation (default `5`, capped at `2` for complex recurrent architectures like VMamba on CPU).
  * `CARA_SHAP_TIMEOUT`: Per-model explanation timeout in seconds (default `6.0s`). If gradient computation exceeds this budget or encounters an architecture error, the model gracefully reports an unavailable status without breaking the comparison.

### 8.2 Demo Mode vs. Trained-Model Explanations
* **Demo Mode**: Because shipped weights are placeholders and class probabilities are simulated, SHAP attributions are explicitly labeled as **Illustrative Sensitivity Maps**. They depict raw mathematical sensitivity of the network architecture to the input pixels and must **not** be interpreted as explanations of the simulated prediction or clinical decisions.
* **Real Mode**: Once trained checkpoints are loaded, SHAP attributions reflect genuine learned feature representations of tissue morphology driving the network's output.

### 8.3 Separation from CARA Scoring
CARA reliability scoring ($R(x) = \alpha C + \beta Q + \gamma S$) and SHAP feature attributions are intentionally decoupled:
* CARA answers: *"Can this model's prediction be trusted based on calibration, image sharpness, staining, and validation reliability?"*
* SHAP answers: *"Which spatial pixels in this specific patch had the highest gradient influence on the model's logits?"*
SHAP attributions indicate model-internal feature importance, **not** clinical causation or ground-truth diagnostic proof.

---

## 9. Project structure

```
cara-histopathology-multi-model-web/
├── backend/
│   ├── app/
│   │   ├── main.py            FastAPI app
│   │   ├── models/            the 8 models (+ registry / checkpoint helpers in __init__.py)
│   │   ├── cara.py            CARA pipeline and metrics
│   │   ├── shap_explainer.py  SHAP GradientExplainer & RGBA attribution renderer
│   │   ├── quality.py         blur + staining quality score
│   │   ├── dataset.py         PanNuke loader + synthetic generator
│   │   ├── inference.py       engine, Demo-Mode simulator, evaluation
│   │   └── utils.py           constants, paths, BaseClassifier
│   ├── checkpoints/           .pth files (placeholders until you train)
│   ├── results/               calibration + test results + cached quality scores
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/                  React + Vite + Tailwind + Recharts
├── data/{pannuke,samples}/
├── scripts/{prepare_pannuke,train_all,evaluate_all}.py
├── docker-compose.yml
└── .env.example
```

## 10. Limitations

* Research prototype – **not a medical device**; do not use for diagnosis.
* In Demo Mode predictions are simulated (see the banner and the first section).
* SHAP attributions reflect statistical sensitivity of the neural network and do not establish medical causation.
* Hyper-parameters in `train_all.py` are sensible defaults, not tuned results; expect to tune epochs and learning
  rates per model. VMamba here is a compact pure-PyTorch variant, not the official CUDA-kernel implementation.
