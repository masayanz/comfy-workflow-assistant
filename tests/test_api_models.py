import io
import os
import unittest
from unittest.mock import AsyncMock, patch
from pathlib import Path
from tempfile import TemporaryDirectory
import time

from PIL import Image

from fastapi.testclient import TestClient

from app.main import app


class ModelsApiTests(unittest.TestCase):
    def test_index_serves_builder_page(self):
        response = TestClient(app).get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="build-button"', response.text)
        self.assertIn('id="run-button"', response.text)

    def test_models_endpoint_uses_parent_comfy_api_inventory(self):
        models = [{"name": "base.safetensors", "comfy_name": "base.safetensors", "type": "checkpoint", "size": 10,
                   "path": "comfy-api:checkpoint:base.safetensors", "modified": None, "family": "sdxl", "source": "comfy_api"}]
        with patch("app.main.client") as client_factory, patch("app.main.get_settings", return_value={}), \
             patch("app.main._models_cache_at", 0.0), patch("app.main._models_cache_url", ""):
            client_factory.return_value.available_models = AsyncMock(return_value=models)
            client_factory.return_value.base_url = "https://parent-comfy"
            client = TestClient(app)
            response = client.get("/api/models")
            second = client.get("/api/comfy/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()[0]["family"], "sdxl")
        self.assertEqual(second.status_code, 200)
        client_factory.return_value.available_models.assert_awaited_once()

    def test_model_profile_endpoints_return_defaults_and_capabilities(self):
        client = TestClient(app)
        profiles = client.get("/api/model-profiles")
        self.assertEqual(profiles.status_code, 200)
        self.assertEqual({profile["id"] for profile in profiles.json()}, {"sd15", "sdxl", "flux"})
        response = client.get("/api/model-profiles/sd15")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["defaults"]["width"], 512)
        self.assertEqual(response.json()["defaults"]["cfg"], 7.0)
        self.assertTrue(response.json()["capabilities"]["txt2img"])
        self.assertFalse(response.json()["capabilities"]["img2img"])
        self.assertTrue(client.get("/api/model-profiles/sdxl").json()["capabilities"]["img2img"])

    def test_model_profile_endpoint_returns_404_for_unknown_id(self):
        response = TestClient(app).get("/api/model-profiles/unknown")
        self.assertEqual(response.status_code, 404)

    def test_flux_profile_endpoint_exposes_split_assets_and_guidance(self):
        response = TestClient(app).get("/api/model-profiles/flux")
        self.assertEqual(response.status_code, 200)
        profile = response.json()
        self.assertTrue(profile["enabled"])
        self.assertEqual(profile["architecture"], "flux_split")
        self.assertEqual(profile["default_guidance"], 3.5)
        self.assertFalse(profile["ui"]["show_negative_prompt"])
        self.assertEqual(len(profile["ui"]["model_components"]), 4)

    def test_manual_flux_classification_uses_disabled_profile(self):
        model = {"name": "unknown.safetensors", "comfy_name": "unknown.safetensors", "type": "checkpoint", "family": "unknown"}
        with patch("app.main.resolve_model", new=AsyncMock(return_value=model)), \
             patch("app.main.get_settings", return_value={"model_families": {}}), \
             patch("app.main.save_settings", side_effect=lambda value: value):
            response = TestClient(app).post("/api/models/classify", json={"model": model["comfy_name"], "family": "flux"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["family"], "flux")

    def test_manual_lora_classification_is_saved_with_source(self):
        model = {"name": "style.safetensors", "comfy_name": "style.safetensors", "type": "lora", "family": "unknown"}
        with patch("app.main.scan_current_models", new=AsyncMock(return_value=[model])), \
             patch("app.main.get_settings", return_value={}), \
             patch("app.main.save_settings", side_effect=lambda value: value) as save:
            response = TestClient(app).post("/api/models/classify", json={
                "model": model["comfy_name"], "asset_type": "lora", "family": "sdxl", "variant": "pony",
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["classification"]["source"], "manual")
        self.assertEqual(response.json()["classification"]["variant"], "pony")
        self.assertEqual(save.call_args.args[0]["model_classifications"]["lora:style.safetensors"]["family"], "sdxl")

    def test_image_endpoint_rejects_path_traversal(self):
        response = TestClient(app).get("/api/comfy/image", params={"filename": "image.png", "subfolder": "../private"})
        self.assertEqual(response.status_code, 400)

    @staticmethod
    def png_bytes(size=(12, 8)):
        output = io.BytesIO()
        Image.new("RGB", size, color=(20, 60, 90)).save(output, format="PNG")
        return output.getvalue()

    def test_image_upload_validates_and_uses_generated_comfy_filename(self):
        from app.main import _uploaded_images
        with patch("app.main.client") as client_factory, patch("app.main.cleanup_local_uploads", return_value=0):
            client_factory.return_value.status = AsyncMock(return_value={"online": True})
            async def upload(filename, content, content_type, subfolder):
                self.assertRegex(filename, r"^cwa_[0-9a-f]{32}\.png$")
                self.assertEqual(content_type, "image/png")
                self.assertEqual(subfolder, "")
                return {"name": filename, "subfolder": "", "type": "input"}
            client_factory.return_value.upload_image = AsyncMock(side_effect=upload)
            response = TestClient(app).post("/api/uploads/image", files={
                "image": ("../../user-photo.png", self.png_bytes(), "image/png"),
            })
        self.assertEqual(response.status_code, 200)
        uploaded = response.json()
        self.assertEqual((uploaded["width"], uploaded["height"]), (12, 8))
        self.assertEqual(uploaded["name"], "user-photo.png")
        self.assertRegex(uploaded["comfy_name"], r"^cwa_[0-9a-f]{32}\.png$")
        self.assertNotIn("created_at", uploaded)
        self.assertIn(uploaded["upload_id"], _uploaded_images)
        _uploaded_images.pop(uploaded["upload_id"], None)
        client_factory.return_value.upload_image.assert_awaited_once()

    def test_image_upload_rejects_extension_mismatch_and_oversized_content(self):
        png = self.png_bytes()
        mismatch = TestClient(app).post("/api/uploads/image", files={"image": ("photo.jpg", png, "image/jpeg")})
        self.assertEqual(mismatch.status_code, 415)
        with patch("app.main.MAX_IMAGE_UPLOAD_BYTES", 16):
            oversized = TestClient(app).post("/api/uploads/image", files={"image": ("photo.png", png, "image/png")})
        self.assertEqual(oversized.status_code, 413)

    def test_cleanup_only_removes_expired_generated_files_from_comfy_input_root(self):
        from app.main import UPLOAD_TTL_SECONDS, cleanup_local_uploads
        with TemporaryDirectory() as directory:
            upload_dir = Path(directory, "input")
            upload_dir.mkdir(parents=True)
            stale = upload_dir / ("cwa_" + "a" * 32 + ".png")
            personal = upload_dir / "my-photo.png"
            stale.write_bytes(b"app")
            personal.write_bytes(b"user")
            now = time.time()
            os.utime(stale, (now - UPLOAD_TTL_SECONDS - 10, now - UPLOAD_TTL_SECONDS - 10))
            with patch("app.main.current_root", return_value=directory), patch("app.main.get_settings", return_value={"comfy_url": "http://127.0.0.1:8188"}):
                removed = cleanup_local_uploads(now=now)
            self.assertEqual(removed, 1)
            self.assertFalse(stale.exists())
            self.assertTrue(personal.exists())

    def test_image_upload_rejects_unreadable_image_content(self):
        response = TestClient(app).post("/api/uploads/image", files={
            "image": ("photo.png", b"not an image", "image/png"),
        })
        self.assertEqual(response.status_code, 415)

    def test_image_upload_reports_comfy_invalid_response_as_gateway_error(self):
        with patch("app.main.client") as client_factory, patch("app.main.cleanup_local_uploads", return_value=0):
            client_factory.return_value.status = AsyncMock(return_value={"online": True})
            client_factory.return_value.upload_image = AsyncMock(return_value={
                "name": "cwa_" + "a" * 32 + ".png", "subfolder": "../outside", "type": "input",
            })
            response = TestClient(app).post("/api/uploads/image", files={
                "image": ("photo.png", self.png_bytes(), "image/png"),
            })
        self.assertEqual(response.status_code, 502)
        self.assertIn("安全でない", response.json()["detail"])

    def test_remote_comfy_upload_cleanup_does_not_touch_workspace_or_local_files(self):
        from app.main import cleanup_local_uploads
        with patch("app.main.current_root", return_value="C:/ComfyUI"), patch("app.main.get_settings", return_value={"comfy_url": "https://comfy.example"}):
            self.assertEqual(cleanup_local_uploads(), 0)


if __name__ == "__main__":
    unittest.main()
