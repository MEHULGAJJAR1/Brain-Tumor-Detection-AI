# Requirements checklist

This checklist is intentionally conservative: “implemented” describes code present in this deliverable; “tested” describes checks actually run in this environment. The supplied files did not contain model weights or the source dataset.

| Requirement | Status | Evidence / notes |
|---|---|---|
| Responsive frontend and in-app navigation | Implemented | Flask-served HTML/CSS/JS; desktop, tablet, mobile layouts; navigation to review, history, and model sections. |
| Image upload validation and preview | Implemented | JPEG/PNG/WebP, size and pixel limits, Pillow decode check, metadata-stripped PNG storage. |
| Persistent scan records | Implemented | SQLite schema, local uploaded-image directory, list/detail/delete APIs. |
| Model status and honest missing-model behavior | Implemented | Missing/invalid checkpoints are explicit; inference is not simulated. |
| ResNet50 training and inference code | Implemented, not run | Optional PyTorch workflow in `scripts/train.py`, `scripts/predict.py`, and `backend/ml/`. No weights/dataset supplied. |
| Figshare `.mat` preparation utility | Implemented, not run | `scripts/prepare_figshare.py`; requires the dataset and optional ML dependencies. |
| API validation and structured errors | Implemented | JSON error responses, upload limits, safe UUID paths, same-origin design, security headers. |
| Unit/integration tests | Implemented and run | `python -m unittest discover -s tests -v` (see final response for results). |
| Browser end-to-end tests | Not automated | HTML and local CSS/JS routes returned HTTP 200; interactive browser workflows were not automated (no Playwright/Selenium dependency included). |
| Training/evaluation metrics | Not run | Requires the Figshare dataset, PyTorch/torchvision, and compute; no accuracy claim is verified. |
| Authentication / multi-user authorization | Not implemented | Intended for localhost research/education use; do not expose publicly without authentication, authorization, and deployment hardening. |
| Clinical validation / regulatory compliance | Not implemented | Not a medical device; model output cannot support diagnosis or treatment. |
