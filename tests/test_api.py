"""Integration tests for upload persistence and the local Flask API."""
from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app import create_app


def image_bytes(color=(110, 120, 130), fmt="PNG"):
    stream = io.BytesIO()
    Image.new("RGB", (48, 40), color).save(stream, format=fmt)
    return stream.getvalue()


class WorkbenchApiTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        root = Path(self.tempdir.name)
        self.upload_dir = root / "uploads"
        self.app = create_app({
            "TESTING": True,
            "DATABASE_PATH": str(root / "test.sqlite3"),
            "UPLOAD_FOLDER": str(self.upload_dir),
            "MODEL_PATH": str(root / "missing-model.pt"),
            "MAX_CONTENT_LENGTH": 1024 * 1024,
            "MAX_UPLOAD_MB": 1,
            "MAX_UPLOAD_BYTES": 1024 * 1024,
            "MAX_IMAGE_PIXELS": 1_000_000,
        })
        self.client = self.app.test_client()

    def tearDown(self):
        self.tempdir.cleanup()

    def test_health_and_model_status_are_explicit_without_weights(self):
        health = self.client.get("/api/health")
        self.assertEqual(health.status_code, 200)
        self.assertTrue(health.json["ok"])
        self.assertFalse(health.json["model_ready"])
        model = self.client.get("/api/model/status")
        self.assertFalse(model.json["ready"])
        self.assertEqual(model.json["state"], "missing_artifact")

    def test_upload_persists_image_record_and_serves_preview(self):
        response = self.client.post("/api/scans", data={
            "file": (io.BytesIO(image_bytes()), "scan-example.png"),
        }, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 201, response.get_data(as_text=True))
        scan = response.json["scan"]
        self.assertEqual(scan["filename"], "scan-example.png")
        self.assertEqual((scan["width"], scan["height"]), (48, 40))
        self.assertEqual(scan["status"], "awaiting_model")
        self.assertIsNone(scan["prediction_class"])
        self.assertTrue((self.upload_dir / f"{scan['id'].replace('-', '')}.png").is_file())

        detail = self.client.get(f"/api/scans/{scan['id']}")
        self.assertEqual(detail.status_code, 200)
        image = self.client.get(scan["image_url"])
        self.assertEqual(image.status_code, 200)
        self.assertEqual(image.mimetype, "image/png")
        self.assertIn("X-Content-Type-Options", image.headers)
        image.close()

        listing = self.client.get("/api/scans")
        self.assertEqual(len(listing.json["scans"]), 1)
        summary = self.client.get("/api/summary")
        self.assertEqual(summary.json["total"], 1)
        self.assertEqual(summary.json["awaiting_model"], 1)

    def test_missing_model_never_returns_a_fake_prediction(self):
        upload = self.client.post("/api/scans", data={
            "file": (io.BytesIO(image_bytes()), "research.png"),
        }, content_type="multipart/form-data")
        scan_id = upload.json["scan"]["id"]
        analysis = self.client.post(f"/api/scans/{scan_id}/analyze")
        self.assertEqual(analysis.status_code, 503)
        self.assertEqual(analysis.json["error"]["code"], "MODEL_NOT_READY")
        detail = self.client.get(f"/api/scans/{scan_id}").json["scan"]
        self.assertEqual(detail["status"], "awaiting_model")
        self.assertIsNone(detail["prediction_class"])

    def test_rejects_invalid_and_unsupported_uploads(self):
        invalid = self.client.post("/api/scans", data={
            "file": (io.BytesIO(b"this is not an image"), "fake.png"),
        }, content_type="multipart/form-data")
        self.assertEqual(invalid.status_code, 400)
        self.assertIn("error", invalid.json)
        unsupported = self.client.post("/api/scans", data={
            "file": (io.BytesIO(b"pdf"), "scan.pdf"),
        }, content_type="multipart/form-data")
        self.assertEqual(unsupported.status_code, 415)
        missing = self.client.post("/api/scans", data={}, content_type="multipart/form-data")
        self.assertEqual(missing.status_code, 400)
        self.assertEqual(self.client.get("/api/scans").json["scans"], [])

    def test_scan_records_persist_after_app_recreation(self):
        upload = self.client.post("/api/scans", data={
            "file": (io.BytesIO(image_bytes()), "persistent.png"),
        }, content_type="multipart/form-data")
        self.assertEqual(upload.status_code, 201)
        scan_id = upload.json["scan"]["id"]
        second_app = create_app({
            "TESTING": True,
            "DATABASE_PATH": self.app.config["DATABASE_PATH"],
            "UPLOAD_FOLDER": self.app.config["UPLOAD_FOLDER"],
            "MODEL_PATH": self.app.config["MODEL_PATH"],
            "MAX_CONTENT_LENGTH": self.app.config["MAX_CONTENT_LENGTH"],
            "MAX_UPLOAD_BYTES": self.app.config["MAX_UPLOAD_BYTES"],
            "MAX_UPLOAD_MB": self.app.config["MAX_UPLOAD_MB"],
            "MAX_IMAGE_PIXELS": self.app.config["MAX_IMAGE_PIXELS"],
        })
        with second_app.test_client() as second_client:
            listing = second_client.get("/api/scans")
            self.assertEqual(listing.status_code, 200)
            self.assertEqual(listing.json["scans"][0]["id"], scan_id)
            self.assertEqual(listing.json["scans"][0]["filename"], "persistent.png")

    def test_delete_removes_database_record_and_image(self):
        upload = self.client.post("/api/scans", data={
            "file": (io.BytesIO(image_bytes(fmt="JPEG")), "to-delete.jpg"),
        }, content_type="multipart/form-data")
        scan = upload.json["scan"]
        image_path = self.upload_dir / f"{scan['id'].replace('-', '')}.png"
        self.assertTrue(image_path.exists())
        deleted = self.client.delete(f"/api/scans/{scan['id']}")
        self.assertEqual(deleted.status_code, 200)
        self.assertFalse(image_path.exists())
        self.assertEqual(self.client.get(f"/api/scans/{scan['id']}").status_code, 404)
        self.assertEqual(self.client.get("/api/summary").json["total"], 0)

    def test_scan_ids_are_validated_and_limit_is_bounded(self):
        self.assertEqual(self.client.get("/api/scans/not-an-id").status_code, 404)
        self.assertEqual(self.client.get("/api/scans?limit=nope").status_code, 400)
        self.assertEqual(self.client.get("/api/scans?limit=999").status_code, 200)

    def test_homepage_and_local_assets_render(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"Brain MRI workbench", page.data)
        self.assertIn(b"data-max-upload-mb=\"1\"", page.data)
        css = self.client.get("/static/css/app.css")
        self.assertEqual(css.status_code, 200)
        self.assertIn(b"--navy", css.data)
        css.close()
        javascript = self.client.get("/static/js/app.js")
        self.assertEqual(javascript.status_code, 200)
        self.assertIn(b"/api/scans", javascript.data)
        javascript.close()

    def test_upload_size_and_pixel_limits_are_enforced(self):
        self.app.config["MAX_UPLOAD_BYTES"] = 64
        too_large = self.client.post("/api/scans", data={
            "file": (io.BytesIO(b"x" * 65), "large.png"),
        }, content_type="multipart/form-data")
        self.assertEqual(too_large.status_code, 413)
        self.assertEqual(too_large.json["error"]["code"], "FILE_TOO_LARGE")

        self.app.config["MAX_UPLOAD_BYTES"] = 1024 * 1024
        stream = io.BytesIO()
        Image.new("RGB", (1200, 1000), (0, 0, 0)).save(stream, format="PNG")
        pixel_response = self.client.post("/api/scans", data={
            "file": (io.BytesIO(stream.getvalue()), "huge.png"),
        }, content_type="multipart/form-data")
        self.assertEqual(pixel_response.status_code, 413)
        self.assertEqual(pixel_response.json["error"]["code"], "IMAGE_TOO_LARGE")
        self.assertEqual(self.client.get("/api/scans").json["scans"], [])


if __name__ == "__main__":
    unittest.main()
