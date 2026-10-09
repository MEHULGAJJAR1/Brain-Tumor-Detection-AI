"""Flask application for the NeuroVista MRI research workbench.

This is a local research/education tool, not a medical device or diagnostic service.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import HTTPException

from backend.db import get_db, init_app as init_database
from backend.ml.inference import ModelService, ModelUnavailable
from backend.storage import UploadError, save_image_upload
from config import Config


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _error(code: str, message: str, details: dict | None = None):
    body = {"error": {"code": code, "message": message}}
    if details:
        body["error"]["details"] = details
    return jsonify(body)


def _is_api_request() -> bool:
    return request.path.startswith("/api/")


def _valid_scan_id(scan_id: str) -> bool:
    try:
        UUID(scan_id)
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def _row_to_scan(row) -> dict:
    probabilities = None
    if row["probabilities_json"]:
        try:
            probabilities = json.loads(row["probabilities_json"])
        except (json.JSONDecodeError, TypeError):
            probabilities = None
    return {
        "id": row["id"],
        "filename": row["original_filename"],
        "mime_type": row["mime_type"],
        "size_bytes": row["size_bytes"],
        "width": row["width"],
        "height": row["height"],
        "sha256": row["sha256"],
        "created_at": row["created_at"],
        "analyzed_at": row["analyzed_at"],
        "status": row["status"],
        "prediction_class": row["prediction_class"],
        "confidence": row["confidence"],
        "probabilities": probabilities,
        "image_url": f"/api/scans/{row['id']}/image",
    }


def _fetch_scan(scan_id: str):
    if not _valid_scan_id(scan_id):
        return None
    return get_db().execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(ROOT / "frontend" / "templates"),
        static_folder=str(ROOT / "frontend" / "static"),
        static_url_path="/static",
    )
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)

    Path(app.config["DATABASE_PATH"]).parent.mkdir(parents=True, exist_ok=True)
    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    app.extensions["model_service"] = ModelService(app.config["MODEL_PATH"])
    init_database(app)

    @app.after_request
    def security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; "
            "script-src 'self'; connect-src 'self'; object-src 'none'; "
            "base-uri 'self'; form-action 'self'",
        )
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.errorhandler(HTTPException)
    def handle_http_error(exc: HTTPException):
        if _is_api_request():
            code = exc.name.upper().replace(" ", "_")
            response = _error(code, exc.description)
            response.status_code = exc.code or 500
            return response
        return exc

    @app.errorhandler(Exception)
    def handle_unexpected_error(exc: Exception):
        app.logger.exception("Unhandled application error")
        if _is_api_request():
            return _error("INTERNAL_ERROR", "The request could not be completed."), 500
        return "The request could not be completed. Return to the workbench and try again.", 500

    @app.get("/")
    def index():
        return render_template("index.html", max_upload_mb=app.config["MAX_UPLOAD_MB"])

    @app.get("/api/health")
    def health():
        get_db().execute("SELECT 1").fetchone()
        model_status = app.extensions["model_service"].status()
        return jsonify(
            {
                "ok": True,
                "service": "neurovista-mri-workbench",
                "database": "connected",
                "model_ready": model_status["ready"],
            }
        )

    @app.get("/api/model/status")
    def model_status():
        return jsonify(app.extensions["model_service"].status())

    @app.get("/api/summary")
    def summary():
        connection = get_db()
        total = connection.execute("SELECT COUNT(*) AS count FROM scans").fetchone()["count"]
        completed = connection.execute("SELECT COUNT(*) AS count FROM scans WHERE status = 'completed'").fetchone()["count"]
        awaiting = connection.execute("SELECT COUNT(*) AS count FROM scans WHERE status = 'awaiting_model'").fetchone()["count"]
        failed = connection.execute("SELECT COUNT(*) AS count FROM scans WHERE status = 'inference_failed'").fetchone()["count"]
        return jsonify({"total": total, "completed": completed, "awaiting_model": awaiting, "inference_failed": failed})

    @app.get("/api/scans")
    def list_scans():
        try:
            limit = min(max(int(request.args.get("limit", "50")), 1), 100)
        except ValueError:
            return _error("INVALID_LIMIT", "limit must be a whole number between 1 and 100."), 400
        rows = get_db().execute(
            "SELECT * FROM scans ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return jsonify({"scans": [_row_to_scan(row) for row in rows]})

    @app.post("/api/scans")
    def upload_scan():
        file_storage = request.files.get("file")
        try:
            uploaded = save_image_upload(
                file_storage,
                app.config["UPLOAD_FOLDER"],
                app.config["MAX_UPLOAD_BYTES"],
                app.config["MAX_IMAGE_PIXELS"],
            )
        except UploadError as exc:
            return _error(exc.code, str(exc)), exc.status_code

        try:
            get_db().execute(
                """INSERT INTO scans
                   (id, original_filename, stored_filename, mime_type, size_bytes,
                    width, height, sha256, created_at, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'awaiting_model')""",
                (
                    uploaded["id"],
                    uploaded["original_filename"],
                    uploaded["stored_filename"],
                    uploaded["mime_type"],
                    uploaded["size_bytes"],
                    uploaded["width"],
                    uploaded["height"],
                    uploaded["sha256"],
                    _utc_now(),
                ),
            )
            get_db().commit()
        except Exception:
            uploaded["path"].unlink(missing_ok=True)
            raise

        row = _fetch_scan(uploaded["id"])
        return jsonify(
            {
                "scan": _row_to_scan(row),
                "message": "Image saved locally. No medical interpretation has been made.",
            }
        ), 201

    @app.get("/api/scans/<scan_id>")
    def get_scan(scan_id: str):
        row = _fetch_scan(scan_id)
        if row is None:
            return _error("SCAN_NOT_FOUND", "No scan was found for that ID."), 404
        return jsonify({"scan": _row_to_scan(row)})

    @app.get("/api/scans/<scan_id>/image")
    def scan_image(scan_id: str):
        row = _fetch_scan(scan_id)
        if row is None:
            return _error("SCAN_NOT_FOUND", "No scan was found for that ID."), 404
        image_path = Path(app.config["UPLOAD_FOLDER"]) / row["stored_filename"]
        if not image_path.is_file():
            return _error("IMAGE_MISSING", "The stored image file is unavailable."), 410
        return send_file(image_path, mimetype="image/png", conditional=True, max_age=0)

    @app.post("/api/scans/<scan_id>/analyze")
    def analyze_scan(scan_id: str):
        row = _fetch_scan(scan_id)
        if row is None:
            return _error("SCAN_NOT_FOUND", "No scan was found for that ID."), 404
        image_path = Path(app.config["UPLOAD_FOLDER"]) / row["stored_filename"]
        if not image_path.is_file():
            return _error("IMAGE_MISSING", "The stored image file is unavailable."), 410

        service: ModelService = app.extensions["model_service"]
        if not service.status()["ready"]:
            return _error(
                "MODEL_NOT_READY",
                "No prediction was generated because a compatible model checkpoint is not ready.",
                {"model": service.status()},
            ), 503

        try:
            prediction = service.predict(image_path)
        except ModelUnavailable as exc:
            return _error("MODEL_NOT_READY", "No prediction was generated because the model is not ready.", {"model": exc.status}), 503
        except Exception:
            app.logger.exception("Inference failed for scan %s", scan_id)
            get_db().execute(
                "UPDATE scans SET status = 'inference_failed', analyzed_at = NULL, prediction_class = NULL, confidence = NULL, probabilities_json = NULL WHERE id = ?",
                (scan_id,),
            )
            get_db().commit()
            return _error("INFERENCE_FAILED", "Inference failed. The scan remains saved; check the model setup and server log."), 500

        analyzed_at = _utc_now()
        get_db().execute(
            """UPDATE scans
               SET status = 'completed', prediction_class = ?, confidence = ?,
                   probabilities_json = ?, analyzed_at = ?
               WHERE id = ?""",
            (
                prediction["prediction_class"],
                prediction["confidence"],
                json.dumps(prediction["probabilities"], sort_keys=True),
                analyzed_at,
                scan_id,
            ),
        )
        get_db().commit()
        updated = _fetch_scan(scan_id)
        return jsonify(
            {
                "scan": _row_to_scan(updated),
                "disclaimer": "Research model output only; not a diagnosis or a substitute for review by a qualified clinician.",
            }
        )

    @app.delete("/api/scans/<scan_id>")
    def delete_scan(scan_id: str):
        row = _fetch_scan(scan_id)
        if row is None:
            return _error("SCAN_NOT_FOUND", "No scan was found for that ID."), 404
        image_path = Path(app.config["UPLOAD_FOLDER"]) / row["stored_filename"]
        get_db().execute("DELETE FROM scans WHERE id = ?", (scan_id,))
        get_db().commit()
        try:
            image_path.unlink(missing_ok=True)
        except OSError:
            app.logger.warning("Unable to remove stored image for deleted scan %s", scan_id)
        return jsonify({"deleted": True, "id": scan_id})

    return app


app = create_app()


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0").lower() in {"1", "true", "yes"}
    app.run(host=host, port=port, debug=debug)
