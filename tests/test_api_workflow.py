import unittest
from unittest.mock import AsyncMock, patch
from pathlib import Path
from tempfile import TemporaryDirectory
import json

from fastapi.testclient import TestClient

from app.main import app


class WorkflowApiTests(unittest.TestCase):
    def test_build_endpoint_returns_sdxl_api_workflow(self):
        model = {"name": "base.safetensors", "comfy_name": "base.safetensors", "type": "checkpoint", "family": "sdxl"}
        with patch("app.main.resolve_model", new=AsyncMock(return_value=model)):
            response = TestClient(app).post("/api/workflow/build", json={"model": "base.safetensors", "prompt": "portrait"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["workflow"]["1"]["inputs"]["ckpt_name"], "base.safetensors")
        self.assertEqual(response.json()["ui_workflow"]["version"], 0.4)

    def test_ui_workflow_save_uses_distinct_filename_and_download(self):
        model = {"name": "base.safetensors", "comfy_name": "base.safetensors", "type": "checkpoint", "family": "sdxl"}
        with TemporaryDirectory() as directory, \
             patch("app.main.OUTPUT_DIR", Path(directory)), \
             patch("app.main.resolve_model", new=AsyncMock(return_value=model)):
            response = TestClient(app).post("/api/workflow/save-ui", json={
                "model": "base.safetensors", "prompt": "portrait", "negative_prompt": "blur", "seed": 42,
            })
            self.assertEqual(response.status_code, 200)
            result = response.json()
            self.assertTrue(result["filename"].endswith(".workflow.json"))
            saved = json.loads((Path(directory) / result["filename"]).read_text(encoding="utf-8"))
            self.assertEqual(saved["nodes"][1]["widgets_values"], ["portrait"])
            downloaded = TestClient(app).get(result["download_url"])
            self.assertEqual(downloaded.status_code, 200)
            self.assertEqual(downloaded.json()["nodes"][4]["widgets_values"][0], 42)

    def test_ui_workflow_save_rejects_sd15_without_affecting_api_prompt(self):
        model = {"name": "v1-5.safetensors", "comfy_name": "v1-5.safetensors", "type": "checkpoint", "family": "sd15"}
        with patch("app.main.resolve_model", new=AsyncMock(return_value=model)):
            response = TestClient(app).post("/api/workflow/save-ui", json={"model": "v1-5.safetensors", "prompt": "portrait"})
        self.assertEqual(response.status_code, 400)

    def test_build_endpoint_rejects_non_sdxl_model(self):
        model = {"name": "flux.safetensors", "comfy_name": "flux.safetensors", "type": "checkpoint", "family": "flux"}
        with patch("app.main.resolve_model", new=AsyncMock(return_value=model)):
            response = TestClient(app).post("/api/workflow/build", json={"model": "flux.safetensors", "prompt": "x"})
        self.assertEqual(response.status_code, 400)

    def test_build_endpoint_rejects_missing_lora(self):
        model = {"name": "base.safetensors", "comfy_name": "base.safetensors", "type": "checkpoint", "family": "sdxl"}
        with patch("app.main.resolve_model", new=AsyncMock(return_value=model)), patch("app.main._models_cache", []), patch("app.main.scan_models", return_value=[]), patch("app.main.client") as client_factory:
            client_factory.return_value.available_models = AsyncMock(return_value=[])
            response = TestClient(app).post("/api/workflow/build", json={"model": "base.safetensors", "prompt": "x", "lora": "missing.safetensors"})
        self.assertEqual(response.status_code, 400)

    def test_run_endpoint_queues_workflow_without_claiming_completion(self):
        from app.schemas.workflow import WorkflowBuildRequest
        model = {"name": "base.safetensors", "comfy_name": "base.safetensors", "type": "checkpoint", "family": "sdxl"}
        with patch("app.main.resolve_model", new=AsyncMock(return_value=model)), patch("app.main.client") as client_factory:
            client_factory.return_value.status = AsyncMock(return_value={"online": True, "url": "https://comfy"})
            client_factory.return_value.queue = AsyncMock(return_value="prompt-1")
            client_factory.return_value.available_models = AsyncMock(return_value=[])
            response = TestClient(app).post("/api/workflow/run", json={"model": "base.safetensors", "prompt": "test", "seed": 42})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "QUEUED")
        self.assertEqual(response.json()["prompt_id"], "prompt-1")
        client_factory.return_value.queue.assert_awaited_once()

    def test_workflow_status_returns_comfyui_image_url(self):
        with patch("app.main.client") as client_factory:
            client_factory.return_value.prompt_status = AsyncMock(return_value={
                "status": "COMPLETED", "prompt_id": "prompt-1",
                "images": [{"filename": "result.png", "subfolder": "", "type": "output"}],
            })
            response = TestClient(app).get("/api/workflow/status/prompt-1")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "COMPLETED")
        self.assertIn("/api/comfy/image?", response.json()["images"][0]["url"])

    def test_workflow_status_explains_checkpoint_failure(self):
        with patch("app.main.client") as client_factory:
            client_factory.return_value.prompt_status = AsyncMock(return_value={
                "status": "ERROR", "prompt_id": "bad", "errors": [["execution_error", {"node_type": "CheckpointLoaderSimple"}]],
            })
            response = TestClient(app).get("/api/workflow/status/bad")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Checkpointが見つかりません", response.json()["message"])


if __name__ == "__main__":
    unittest.main()
