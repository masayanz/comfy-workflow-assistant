import unittest
from unittest.mock import patch

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

    def test_image_endpoint_rejects_path_traversal(self):
        response = TestClient(app).get("/api/comfy/image", params={"filename": "image.png", "subfolder": "../private"})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
