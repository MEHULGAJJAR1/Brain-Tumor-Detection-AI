# NeuroVista MRI Workbench

A local-first, full-stack research workbench for uploading and reviewing a single brain MRI image. It provides a responsive browser UI, a Flask JSON API, persistent SQLite scan history, validated image storage, and an optional CPU-compatible ResNet50 training/inference path for three research labels: **glioma**, **meningioma**, and **pituitary**.

> **Important: not for clinical use.** This project is educational/research software, not a medical device. It cannot diagnose a brain tumor, establish that a tumor is present or absent, or guide treatment. Outputs require qualified clinical review. Do not upload identifiable patient data.

## Project status and supplied materials

**Intended users:** students and research developers prototyping an image-classification workflow—not patients making care decisions or clinicians relying on it.

The supplied reference README described a fine-tuned ResNet50 and reported approximately 99.3% accuracy. The attached files did **not** include the claimed `.pt` weights, training notebooks, or Figshare dataset. The attached legacy `requirements.txt` also specified Python 3.5 with TensorFlow 1.5/Keras 1.1, which conflicts with the more detailed PyTorch/ResNet50 project description; this implementation follows the latter and uses a modern Flask/PyTorch stack. Accordingly:

- The application works without a model: image validation, local persistence, history, preview, search, deletion, health/status APIs, and documentation are available.
- The default UI explicitly shows **Model not configured**. Uploading a scan does not generate or fabricate a prediction.
- Actual classification becomes available only after you install compatible optional dependencies and place a compatible, trained checkpoint at the configured `MODEL_PATH`.
- The reported accuracy is **unverified** and is not represented as a tested result. Training/evaluation were not run because the dataset and trained artifact were not supplied.

## Problem statement and objectives

Provide a usable, transparent full-stack starting point for exploring MRI-image classification without misrepresenting an unavailable AI model as functional. The project focuses on safe image ingestion, persistent local records, clear model readiness, reproducible training utilities, and strong medical-use limitations.

## Features

### Frontend
- Responsive dashboard for desktop, tablet, and mobile.
- Drag-and-drop or file-picker upload for JPEG, PNG, and WebP images (single-frame, up to 12 MB by default).
- Client preview, upload progress state, actionable validation messages, recent scan history, filename/status filters, image detail dialog, and delete confirmation.
- Model readiness and setup status; an explicit notice when no model is configured.
- No external fonts, scripts, CSS, or image dependencies.

### Backend and data
- Flask application with JSON endpoints, request validation, structured errors, logging, and same-origin API access. No permissive cross-origin policy is enabled because the UI and API are served from the same origin.
- SQLite-backed persistent records and local image storage; schema initialized automatically.
- Files are assigned server-generated IDs; original filenames are sanitized. Upload metadata is stripped by re-encoding to PNG before storage.
- Scan list, detail, image-preview, analyze, summary, health, and delete endpoints.
- Security response headers and file-size/image-pixel limits.
- A scan can be deleted from both the database and local image directory.

### Optional ML workflow
- ResNet50 with the reference 2048 → 512 → 3 classifier head.
- Figshare `.mat` preparation script and training/evaluation CLI.
- Train-time augmentation, head warm-up, full-network fine-tuning, validation checkpoint selection, and test-set classification report.
- CPU inference; GPU training when a compatible PyTorch/CUDA setup is installed.
- Patient-group split support through a user-provided `groups.csv`.

## Technology stack

| Layer | Technology |
|---|---|
| Web UI | HTML5, CSS3, vanilla JavaScript |
| HTTP backend | Python 3.10+, Flask |
| Persistence | SQLite (`sqlite3`, standard library) |
| Image validation | Pillow |
| Optional ML | PyTorch, torchvision, NumPy, SciPy, scikit-learn |
| Tests | Python `unittest` + Flask test client |

## Architecture and folder structure

```text
neurovista-mri-workbench/
├── app.py                     # Flask app factory, routes, API and startup
├── config.py                  # Environment-based configuration
├── backend/
│   ├── db.py                  # SQLite connection and schema initialization
│   ├── storage.py             # Image validation and metadata-stripped storage
│   └── ml/
│       ├── model.py           # ResNet50 model definition (optional dependencies)
│       └── inference.py       # Lazy model loading and CPU inference
├── database/
│   ├── schema.sql             # Persistent scan schema and indexes
│   └── README.md              # Data and backup notes
├── frontend/
│   ├── templates/index.html   # Single-page research workspace
│   └── static/                # Local CSS, JS, and SVG branding
├── scripts/
│   ├── prepare_figshare.py    # .mat → ImageFolder PNG conversion
│   ├── train.py               # ResNet50 training and test evaluation
│   └── predict.py             # CLI inference
├── models/README.md           # Required checkpoint format; no weights included
├── tests/test_api.py          # Upload, API, model-unavailable, persistence tests
├── docs/requirements-checklist.md
├── requirements.txt           # Lightweight web app dependencies
├── requirements-ml.txt        # Optional deep-learning dependencies
├── .env.example               # Configuration template
├── run.sh                     # Unix/macOS launch helper
└── run.ps1                    # Windows PowerShell launch helper
```

## Prerequisites

- Python **3.10–3.13** recommended.
- A modern browser.
- Optional model training/inference: a PyTorch/torchvision-compatible platform; a CPU is supported. CUDA is optional.
- Optional training data: Figshare Jun Cheng brain tumor dataset (not included in this archive).

## Installation and startup

### Linux / macOS

From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
python app.py
```

Or use the helper after installing dependencies:

```bash
chmod +x run.sh
./run.sh
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
python app.py
```

Alternatively, run `.\run.ps1` from PowerShell. If script execution is restricted, use the direct `python app.py` command after activating the environment.

Open **http://127.0.0.1:5000**. The default host is loopback-only. Do not bind the app to a public interface without adding authentication, access control, TLS, safe storage, and deployment hardening.

## Environment configuration

Copy `.env.example` to `.env`. The application loads this file on startup. All settings are optional; defaults are shown below.

| Variable | Default | Purpose |
|---|---|---|
| `HOST` | `127.0.0.1` | Flask bind host; keep loopback for local use. |
| `PORT` | `5000` | HTTP port. |
| `MAX_UPLOAD_MB` | `12` | Maximum incoming image file size. |
| `MAX_IMAGE_PIXELS` | `40000000` | Maximum decoded pixel count. |
| `DATABASE_PATH` | `instance/neurovista.sqlite3` | Persistent SQLite database. |
| `UPLOAD_FOLDER` | `instance/uploads` | Normalized scan image storage. |
| `MODEL_PATH` | `models/brain_tumor_resnet50.pt` | Optional trained checkpoint location. |
| `SECRET_KEY` | Local placeholder | Not used for authentication; change before adding session-based features. |

Relative paths resolve from the project root. The database and uploads are local, unencrypted files; back them up together and do not store real patient data in an unsecured environment.

## Database setup and persistence

The schema is in `database/schema.sql` and is created automatically when the app starts. It stores the sanitized original filename, image dimensions/size, SHA-256, timestamps, status, and actual model outputs when inference is available. Image files are stored under `instance/uploads/` as normalized PNGs. No scan seed data is inserted.

Delete records in the UI to remove both the database row and stored image. For complete erasure, stop the app and remove the configured database and upload directory. For backup/restore, copy both together.

## Model and dataset setup

### 1. Download the dataset (optional)

The reference project identifies the Jun Cheng brain tumor dataset on Figshare (DOI: [10.6084/m9.figshare.1512427](https://doi.org/10.6084/m9.figshare.1512427)). Dataset files are not included in this project archive. Download the dataset from its source and extract the `.mat` files under `data/raw/`.

The original setup notes describe four archives; actual filenames may vary by Figshare release. If you use those named archives, the extraction pattern is:

```bash
mkdir -p data/raw
unzip brainTumorDataPublic_1-766.zip    -d data/raw
unzip brainTumorDataPublic_767-1532.zip -d data/raw
unzip brainTumorDataPublic_1533-2298.zip -d data/raw
unzip brainTumorDataPublic_2299-3064.zip -d data/raw
```

Verify dataset terms and cite the dataset's accompanying publication if you use it. The conversion utility expects MATLAB files containing `cjdata.image` and `cjdata.label`, with source mapping 1=meningioma, 2=glioma, and 3=pituitary.

### 2. Install optional ML packages

```bash
pip install -r requirements-ml.txt
```

For CUDA, first choose/install a PyTorch wheel appropriate to your operating system and CUDA version from the official PyTorch installation selector, then install the remaining dependencies. The base Flask app does not require PyTorch. A default training run requests ImageNet-pretrained ResNet50 weights, which may download on first use; pass `--no-pretrained` to avoid that download.

### 3. Convert `.mat` files

```bash
python scripts/prepare_figshare.py --input-dir data/raw --output-dir data/processed
```

The output is:

```text
data/processed/
├── glioma/
├── meningioma/
├── pituitary/
└── manifest.csv
```

The script reports skipped files. Review its output and class counts before training. It converts image intensities to 8-bit using percentile scaling; tumor masks are not treated as image-level diagnostic labels.

### 4. Train and evaluate

```bash
python scripts/train.py \
  --data-dir data/processed \
  --output models/brain_tumor_resnet50.pt \
  --metrics-output models/metrics.json \
  --epochs 25 \
  --batch-size 32 \
  --device auto
```

The script saves the best validation-loss checkpoint and test-set metrics. It uses a stratified **image-level** 70/15/15 split by default; MRI slices from the same patient may appear in more than one split, which can inflate metrics. For a patient-independent evaluation, provide a CSV with `filename,patient_id` columns and use `--groups-csv path/to/groups.csv`. Filenames may be relative to the processed dataset root (e.g. `glioma/example.png`) or a unique basename. The script fails if it cannot build all-class patient-level partitions. Do not infer or fabricate patient IDs.

Training augmentation, split design, and metrics are research choices, not clinically validated. A test score on this dataset does not establish performance on a different scanner, population, sequence, institution, or patient cohort.

### 5. Run inference

After training, restart the app (or refresh model status) with `MODEL_PATH` pointing to the checkpoint. If using the default path, the UI will pick it up when refreshed. The app loads checkpoints lazily and runs CPU inference.

CLI example:

```bash
python scripts/predict.py --image path/to/mri-image.jpg
```

The classifier is a forced three-way classifier: it has **no “no tumor” class** and is not designed to screen arbitrary images or full studies. It accepts only one 2D JPEG, PNG, or WebP image—not DICOM, NIfTI, multi-slice studies, or archives.

## API endpoints

All API routes are same-origin under `/api`. JSON errors use `{ "error": { "code": "...", "message": "..." } }`.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Service/database health and model-ready boolean. |
| `GET` | `/api/model/status` | Model readiness and configuration hint. |
| `GET` | `/api/summary` | Actual scan/status counters. |
| `GET` | `/api/scans?limit=50` | Recent scans, up to 100. |
| `POST` | `/api/scans` | Multipart form upload using field `file`; persists a scan, does not itself claim a prediction. |
| `GET` | `/api/scans/<uuid>` | Scan record. |
| `GET` | `/api/scans/<uuid>/image` | Stored PNG preview. |
| `POST` | `/api/scans/<uuid>/analyze` | Run inference if a compatible model is ready; otherwise returns `503 MODEL_NOT_READY`. |
| `DELETE` | `/api/scans/<uuid>` | Delete scan record and local image. |

Upload example:

```bash
curl -X POST http://127.0.0.1:5000/api/scans \
  -F 'file=@path/to/mri-image.png'
```

Analysis example (replace the ID with the `scan.id` returned by the upload):

```bash
curl -X POST http://127.0.0.1:5000/api/scans/YOUR-SCAN-UUID/analyze
```

`MODEL_NOT_READY` is an expected response until valid weights are configured. The web UI surfaces that state without displaying a fabricated result.

## Testing

Install base dependencies, then from the project root run:

```bash
python -m unittest discover -s tests -v
```

Tests cover health/model status, valid and invalid uploads, persistent metadata, image preview, deletion, bounded scan listing, and refusal to predict without weights. Training and model-inference tests require the optional ML dependencies and a compatible trained checkpoint; training additionally requires the external dataset. No model accuracy is claimed as tested.

## Common issues

- **`No module named flask`**: activate the virtual environment and run `pip install -r requirements.txt`.
- **Model not configured**: expected until weights are trained or a compatible checkpoint is provided. See `models/README.md`.
- **ML dependencies missing**: install the platform-appropriate PyTorch/torchvision pair and `requirements-ml.txt`.
- **Checkpoint invalid**: use a checkpoint produced by this project's `scripts/train.py`; it must include the expected architecture and class metadata. Do not relabel unknown weights.
- **Port 5000 is busy**: set `PORT=5001` in `.env` and restart.
- **Upload rejected**: check extension and actual file type, single-frame image format, 12 MB default limit, and configured pixel limit.
- **SQLite locked**: close other processes using the database and avoid sharing one SQLite file over a network filesystem.
- **Dataset conversion skips files**: confirm the `.mat` files contain `cjdata.image` and `cjdata.label`; inspect the reported paths/errors.

## Deployment notes

This project is designed for localhost research and education. It has no user accounts, authorization, audit workflow, encryption-at-rest, retention scheduler, or clinical validation. Do not publish it or use it with protected health information. A real deployment would require a security review, identity and access controls, encrypted transport and storage, data-retention policy, monitoring, and applicable regulatory/privacy review. Adding those controls does not by itself make model output clinically valid.

## Known limitations and future work

- No pretrained model, training data, or patient grouping metadata is bundled.
- Training, evaluation, and predictions were not run in this deliverable environment; reported accuracy from the reference README is unverified.
- The 3-class model cannot detect “no tumor,” distinguish other pathologies, or evaluate a complete MRI exam.
- Confidence values are softmax scores, not calibrated clinical probabilities.
- Only single-frame JPEG/PNG/WebP images are accepted. DICOM/NIfTI and multi-series workflows are out of scope.
- No authentication or multi-user access control; local files are not encrypted at rest.
- The current automated tests validate backend behavior; browser E2E automation and clinical validation are not included.

For the implemented/tested requirement checklist, see [`docs/requirements-checklist.md`](docs/requirements-checklist.md). Dataset citation details are in the original Figshare record and Cheng et al. publications.
