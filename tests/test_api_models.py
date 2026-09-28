import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app


class ModelsApiTests(unittest.TestCase):
    def test_index_serves_builder_page(self):
        response = TestClient(app).get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn('id="build-button"', response.text)
        self.assertIn('id="run-button"', response.text)

    def test_models_endpoint_returns_scanned_models(self):
        models = [{"name": "base.safetensors", "comfy_name": "base.safetensors", "type": "checkpoint", "size": 10,
                   "path": "C:/models/base.safetensors", "modified": "2026-01-01T00:00:00+09:00", "family": "sdxl"}]
        with patch("app.main.scan_models", return_value=models), patch("app.main.current_root", return_value="C:/ComfyUI"):
            response = TestClient(app).get("/api/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()[0]["family"], "sdxl")

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

    def test_model_profile_endpoint_returns_404_for_unknown_id(self):
        response = TestClient(app).get("/api/model-profiles/unknown")
        self.assertEqual(response.status_code, 404)

    def test_manual_flux_classification_uses_disabled_profile(self):
        model = {"name": "unknown.safetensors", "comfy_name": "unknown.safetensors", "type": "checkpoint", "family": "unknown"}
        with patch("app.main.resolve_model", new=AsyncMock(return_value=model)), \
             patch("app.main.get_settings", return_value={"model_families": {}}), \
             patch("app.main.save_settings", side_effect=lambda value: value):
            response = TestClient(app).post("/api/models/classify", json={"model": model["comfy_name"], "family": "flux"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["family"], "flux")

    def test_image_endpoint_rejects_path_traversal(self):
        response = TestClient(app).get("/api/comfy/image", params={"filename": "image.png", "subfolder": "../private"})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
