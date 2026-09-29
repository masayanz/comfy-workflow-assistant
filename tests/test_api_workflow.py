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

    def test_sd15_ui_workflow_save_creates_workflow_file(self):
        model = {"name": "v1-5.safetensors", "comfy_name": "v1-5.safetensors", "type": "checkpoint", "family": "sd15"}
        with patch("app.main.resolve_model", new=AsyncMock(return_value=model)), patch("app.main.validate_lora", new=AsyncMock()):
            response = TestClient(app).post("/api/workflow/save-ui", json={"model": "v1-5.safetensors", "prompt": "portrait"})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["filename"].endswith(".workflow.json"))

    def test_sd15_build_returns_both_api_and_ui_workflows(self):
        model = {"name": "v1-5.safetensors", "comfy_name": "v1-5.safetensors", "type": "checkpoint", "family": "sd15"}
        with patch("app.main.resolve_model", new=AsyncMock(return_value=model)), patch("app.main.validate_lora", new=AsyncMock()):
            response = TestClient(app).post("/api/workflow/build", json={"model": "v1-5.safetensors", "prompt": "portrait"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["workflow"]["4"]["inputs"]["width"], 512)
        self.assertEqual(response.json()["ui_workflow"]["version"], 0.4)

    def test_build_endpoint_rejects_non_sdxl_model(self):
        model = {"name": "flux.safetensors", "comfy_name": "flux.safetensors", "type": "checkpoint", "family": "flux"}
        with patch("app.main.resolve_model", new=AsyncMock(return_value=model)):
            response = TestClient(app).post("/api/workflow/build", json={"model": "flux.safetensors", "prompt": "x"})
        self.assertEqual(response.status_code, 400)

    def test_flux_build_generates_unconfigured_preview_without_claiming_assets_ready(self):
        with patch("app.main.scan_current_models", new=AsyncMock(return_value=[])):
            response = TestClient(app).post("/api/workflow/build", json={"profile_id": "flux", "prompt": "a mountain lake"})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertFalse(result["ready"])
        self.assertEqual(len(result["missing_assets"]), 4)
        self.assertEqual(result["workflow"]["1"]["class_type"], "UNETLoader")
        self.assertEqual(result["workflow"]["1"]["inputs"]["unet_name"], "")
        self.assertEqual(result["ui_workflow"]["nodes"][-1]["type"], "SaveImage")

    def test_flux_build_marks_real_compatible_components_ready(self):
        assets = [
            ("diffusion_model", "flux1-dev.safetensors"), ("text_encoder", "clip_l.safetensors"),
            ("text_encoder", "t5xxl_fp8.safetensors"), ("vae", "ae.safetensors"),
        ]
        models = [{"type": kind, "family": "flux", "name": name, "comfy_name": name} for kind, name in assets]
        request = {"profile_id": "flux", "prompt": "a mountain lake", "diffusion_model": assets[0][1],
                   "clip_name1": assets[1][1], "clip_name2": assets[2][1], "vae_model": assets[3][1], "seed": 1234}
        with patch("app.main.scan_current_models", new=AsyncMock(return_value=models)):
            response = TestClient(app).post("/api/workflow/build", json=request)
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertTrue(result["ready"])
        self.assertEqual(result["missing_assets"], [])
        self.assertEqual(result["workflow"]["1"]["inputs"]["unet_name"], assets[0][1])
        self.assertEqual(result["workflow"]["2"]["inputs"]["clip_name2"], assets[2][1])

    def test_flux_run_is_blocked_until_compatible_assets_are_selected(self):
        assets = [
            {"type": "diffusion_model", "family": "unknown", "name": "z_image_turbo_bf16.safetensors", "comfy_name": "z_image_turbo_bf16.safetensors"},
            {"type": "text_encoder", "family": "unknown", "name": "qwen_3_4b.safetensors", "comfy_name": "qwen_3_4b.safetensors"},
            {"type": "vae", "family": "flux", "name": "ae.safetensors", "comfy_name": "ae.safetensors"},
        ]
        with patch("app.main.client") as client_factory, patch("app.main.scan_current_models", new=AsyncMock(return_value=assets)):
            client_factory.return_value.status = AsyncMock(return_value={"online": True, "url": "https://comfy"})
            client_factory.return_value.queue = AsyncMock(return_value="should-not-queue")
            response = TestClient(app).post("/api/workflow/run", json={"profile_id": "flux", "prompt": "a mountain lake"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("モデルが不足", response.json()["detail"])
        client_factory.return_value.queue.assert_not_awaited()

    def test_img2img_build_rejects_missing_and_unknown_upload_ids(self):
        model = {"name": "base.safetensors", "comfy_name": "base.safetensors", "type": "checkpoint", "family": "sdxl"}
        with patch("app.main.resolve_model", new=AsyncMock(return_value=model)), patch("app.main._uploaded_images", {}):
            missing = TestClient(app).post("/api/workflow/build", json={"model": "base.safetensors", "generation_type": "img2img", "prompt": "x"})
            unknown = TestClient(app).post("/api/workflow/build", json={"model": "base.safetensors", "generation_type": "img2img", "input_image_id": "not-found", "prompt": "x"})
        self.assertEqual(missing.status_code, 400)
        self.assertIn("入力画像が必要", missing.json()["detail"])
        self.assertEqual(unknown.status_code, 404)

    def test_img2img_build_uses_uploaded_input_image_and_reports_dimensions(self):
        model = {"name": "base.safetensors", "comfy_name": "base.safetensors", "type": "checkpoint", "family": "sdxl"}
        image = {"upload_id": "u1", "comfy_name": "cwa_u1.png", "name": "source.png", "width": 720, "height": 512, "created_at": 1e20}
        with patch("app.main.resolve_model", new=AsyncMock(return_value=model)), patch("app.main._uploaded_images", {"u1": image}):
            response = TestClient(app).post("/api/workflow/build", json={
                "model": "base.safetensors", "generation_type": "img2img", "input_image_id": "u1",
                "prompt": "soft watercolor", "denoise": 0.4, "seed": 42,
            })
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["generation_type"], "img2img")
        self.assertEqual(result["input_image"], {"name": "source.png", "width": 720, "height": 512})
        self.assertEqual(result["workflow"]["4"]["inputs"]["image"], "cwa_u1.png")
        self.assertEqual(result["workflow"]["6"]["inputs"]["denoise"], 0.4)
        self.assertEqual(result["ui_workflow"]["nodes"][3]["widgets_values"], ["cwa_u1.png", "image"])

    def test_img2img_denooise_out_of_range_is_rejected_by_request_validation(self):
        response = TestClient(app).post("/api/workflow/build", json={
            "model": "base.safetensors", "generation_type": "img2img", "input_image_id": "u1", "prompt": "x", "denoise": 1.2,
        })
        self.assertEqual(response.status_code, 422)

    def test_sd15_img2img_stays_disabled_until_profile_capability_is_enabled(self):
        model = {"name": "v1-5.safetensors", "comfy_name": "v1-5.safetensors", "type": "checkpoint", "family": "sd15"}
        with patch("app.main.resolve_model", new=AsyncMock(return_value=model)):
            response = TestClient(app).post("/api/workflow/build", json={"model": "v1-5.safetensors", "generation_type": "img2img", "prompt": "x"})
        self.assertEqual(response.status_code, 400)
        self.assertIn("img2imgは現在対応していません", response.json()["detail"])

    def test_img2img_run_sends_image_latent_and_denoise_to_comfy(self):
        model = {"name": "base.safetensors", "comfy_name": "base.safetensors", "type": "checkpoint", "family": "sdxl"}
        image = {"upload_id": "u1", "comfy_name": "cwa_u1.png", "name": "source.png", "width": 512, "height": 512, "created_at": 1e20}
        with patch("app.main.resolve_model", new=AsyncMock(return_value=model)), patch("app.main._uploaded_images", {"u1": image}), patch("app.main.client") as client_factory:
            client_factory.return_value.status = AsyncMock(return_value={"online": True, "url": "https://comfy"})
            client_factory.return_value.queue = AsyncMock(return_value="img-prompt")
            response = TestClient(app).post("/api/workflow/run", json={
                "model": "base.safetensors", "generation_type": "img2img", "input_image_id": "u1",
                "prompt": "watercolor", "denoise": 0.5, "seed": 42,
            })
        self.assertEqual(response.status_code, 200)
        queued = client_factory.return_value.queue.await_args.args[0]
        self.assertEqual(queued["4"]["class_type"], "LoadImage")
        self.assertEqual(queued["5"]["inputs"]["pixels"], ["4", 0])
        self.assertEqual(queued["6"]["inputs"]["latent_image"], ["5", 0])
        self.assertEqual(queued["6"]["inputs"]["denoise"], 0.5)
        self.assertEqual(response.json()["generation_type"], "img2img")

    def test_build_endpoint_rejects_missing_lora(self):
        model = {"name": "base.safetensors", "comfy_name": "base.safetensors", "type": "checkpoint", "family": "sdxl"}
        with patch("app.main.resolve_model", new=AsyncMock(return_value=model)), patch("app.main.scan_current_models", new=AsyncMock(return_value=[])):
            response = TestClient(app).post("/api/workflow/build", json={"model": "base.safetensors", "prompt": "x", "lora": "missing.safetensors"})
        self.assertEqual(response.status_code, 400)

    def test_clear_lora_family_mismatch_blocks_queue(self):
        checkpoint = {"name": "base.safetensors", "comfy_name": "base.safetensors", "type": "checkpoint", "family": "sdxl",
                      "classification": {"family": "sdxl", "confidence": "high", "source": "metadata"}}
        lora = {"name": "sd15-style.safetensors", "comfy_name": "sd15-style.safetensors", "type": "lora", "family": "sd15",
                "classification": {"family": "sd15", "confidence": "high", "source": "metadata"}}
        with patch("app.main.resolve_model", new=AsyncMock(return_value=checkpoint)), \
             patch("app.main.scan_current_models", new=AsyncMock(return_value=[checkpoint, lora])), \
             patch("app.main.client") as client_factory:
            client_factory.return_value.status = AsyncMock(return_value={"online": True, "url": "https://comfy"})
            client_factory.return_value.queue = AsyncMock(return_value="must-not-queue")
            response = TestClient(app).post("/api/workflow/run", json={
                "model": checkpoint["comfy_name"], "lora": lora["comfy_name"], "prompt": "portrait",
            })
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"]["code"], "lora_incompatible")
        client_factory.return_value.queue.assert_not_awaited()

    def test_unknown_lora_is_warning_and_workflow_can_be_built(self):
        checkpoint = {"name": "base.safetensors", "comfy_name": "base.safetensors", "type": "checkpoint", "family": "sdxl",
                      "classification": {"family": "sdxl", "confidence": "high", "source": "metadata"}}
        lora = {"name": "unknown-style.safetensors", "comfy_name": "unknown-style.safetensors", "type": "lora", "family": "unknown",
                "classification": {"family": "unknown", "confidence": "unknown", "source": "unknown"}}
        with patch("app.main.resolve_model", new=AsyncMock(return_value=checkpoint)), \
             patch("app.main.scan_current_models", new=AsyncMock(return_value=[checkpoint, lora])):
            response = TestClient(app).post("/api/workflow/build", json={
                "model": checkpoint["comfy_name"], "lora": lora["comfy_name"], "prompt": "portrait",
            })
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["lora_compatibility"]["status"], "unknown")

    def test_lora_compatibility_endpoint_returns_model_classifications(self):
        checkpoint = {"name": "pony.safetensors", "comfy_name": "pony.safetensors", "type": "checkpoint", "family": "sdxl",
                      "classification": {"family": "sdxl", "variant": "pony", "confidence": "high", "source": "metadata"}}
        lora = {"name": "pony-style.safetensors", "comfy_name": "pony-style.safetensors", "type": "lora", "family": "sdxl",
                "classification": {"family": "sdxl", "variant": "pony", "confidence": "high", "source": "metadata"}}
        with patch("app.main.scan_current_models", new=AsyncMock(return_value=[checkpoint, lora])):
            response = TestClient(app).get("/api/models/compatibility", params={"checkpoint": checkpoint["comfy_name"]})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["checkpoint_classification"]["variant"], "pony")
        self.assertEqual(response.json()["loras"][0]["compatibility"]["status"], "compatible")

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

    def test_workflow_status_explains_flux_loader_failure(self):
        with patch("app.main.client") as client_factory:
            client_factory.return_value.prompt_status = AsyncMock(return_value={
                "status": "ERROR", "prompt_id": "bad", "errors": [["execution_error", {"node_type": "DualCLIPLoader"}]],
            })
            response = TestClient(app).get("/api/workflow/status/bad")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Flux", response.json()["message"])

    def test_upscale_build_and_ui_workflow_use_parent_model_and_uploaded_image(self):
        upscale = {"name": "4x-UltraSharp.pth", "comfy_name": "4x-UltraSharp.pth", "type": "upscale_model", "size": 1000, "scale": 4, "family": "unknown", "recognized": True}
        image = {"upload_id": "u1", "comfy_name": "cwa_input.png", "name": "source.png", "width": 320, "height": 240, "created_at": 1e20}
        with patch("app.main.client") as client_factory, patch("app.main._uploaded_images", {"u1": image}):
            client_factory.return_value.status = AsyncMock(return_value={"online": True})
            client_factory.return_value.available_models = AsyncMock(return_value=[upscale])
            client_factory.return_value.object_info = AsyncMock(return_value={"name": "native node"})
            response = TestClient(app).post("/api/workflow/build", json={
                "generation_type": "upscale", "input_image_id": "u1", "upscale_model": "4x-UltraSharp.pth",
            })
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["generation_type"], "upscale")
        self.assertEqual(result["workflow"]["1"]["class_type"], "LoadImage")
        self.assertEqual(result["workflow"]["2"]["inputs"]["model_name"], "4x-UltraSharp.pth")
        self.assertEqual(result["workflow"]["3"]["inputs"]["image"], ["1", 0])
        self.assertEqual(result["ui_workflow"]["nodes"][0]["widgets_values"], ["cwa_input.png", "image"])
        self.assertEqual(result["input_image"], {"name": "source.png", "width": 320, "height": 240})

    def test_upscale_save_ui_creates_canvas_workflow(self):
        upscale = {"name": "RealESRGAN_x4plus.pth", "comfy_name": "RealESRGAN_x4plus.pth", "type": "upscale_model", "size": 1000, "scale": 4, "family": "unknown"}
        image = {"upload_id": "u1", "comfy_name": "cwa_input.png", "name": "source.png", "width": 256, "height": 128, "created_at": 1e20}
        with TemporaryDirectory() as directory, patch("app.main.OUTPUT_DIR", Path(directory)), \
             patch("app.main.client") as client_factory, patch("app.main._uploaded_images", {"u1": image}):
            client_factory.return_value.status = AsyncMock(return_value={"online": True})
            client_factory.return_value.available_models = AsyncMock(return_value=[upscale])
            client_factory.return_value.object_info = AsyncMock(return_value={"name": "native node"})
            response = TestClient(app).post("/api/workflow/save-ui", json={
                "generation_type": "upscale", "input_image_id": "u1", "upscale_model": "RealESRGAN_x4plus.pth",
            })
            self.assertEqual(response.status_code, 200)
            saved = json.loads((Path(directory) / response.json()["filename"]).read_text(encoding="utf-8"))
        self.assertTrue(response.json()["filename"].endswith(".workflow.json"))
        self.assertEqual({node["type"] for node in saved["nodes"]}, {"LoadImage", "UpscaleModelLoader", "ImageUpscaleWithModel", "SaveImage"})

    def test_upscale_run_queues_native_nodes_and_reports_estimated_dimensions(self):
        upscale = {"name": "4x-UltraSharp.pth", "comfy_name": "4x-UltraSharp.pth", "type": "upscale_model", "size": 1000, "scale": 4, "family": "unknown"}
        image = {"upload_id": "u1", "comfy_name": "cwa_input.png", "name": "source.png", "width": 300, "height": 180, "created_at": 1e20}
        with patch("app.main.client") as client_factory, patch("app.main._uploaded_images", {"u1": image}):
            client_factory.return_value.status = AsyncMock(return_value={"online": True})
            client_factory.return_value.available_models = AsyncMock(return_value=[upscale])
            client_factory.return_value.object_info = AsyncMock(return_value={"name": "native node"})
            client_factory.return_value.queue = AsyncMock(return_value="upscale-prompt")
            response = TestClient(app).post("/api/workflow/run", json={
                "generation_type": "upscale", "input_image_id": "u1", "upscale_model": "4x-UltraSharp.pth",
            })
        self.assertEqual(response.status_code, 200)
        queued = client_factory.return_value.queue.await_args.args[0]
        self.assertEqual(queued["1"]["class_type"], "LoadImage")
        self.assertEqual(queued["2"]["class_type"], "UpscaleModelLoader")
        self.assertEqual(queued["3"]["class_type"], "ImageUpscaleWithModel")
        self.assertEqual(response.json()["estimated_output_size"], {"width": 1200, "height": 720})

    def test_upscale_rejects_missing_parent_model_and_offline_comfy(self):
        with patch("app.main.client") as client_factory:
            client_factory.return_value.status = AsyncMock(return_value={"online": True})
            client_factory.return_value.available_models = AsyncMock(return_value=[])
            response = TestClient(app).post("/api/workflow/build", json={
                "generation_type": "upscale", "input_image_id": "u1", "upscale_model": "missing.pth",
            })
        self.assertEqual(response.status_code, 400)
        self.assertIn("親機ComfyUIに見つかりません", response.json()["detail"])

        with patch("app.main.client") as client_factory:
            client_factory.return_value.status = AsyncMock(return_value={"online": False})
            response = TestClient(app).post("/api/workflow/run", json={
                "generation_type": "upscale", "input_image_id": "u1", "upscale_model": "model.pth",
            })
        self.assertEqual(response.status_code, 503)

    def test_diagnostics_endpoint_returns_parent_inventory_and_missing_assets(self):
        report = {
            "online": True, "models": [
                {"name": "v1-5-pruned.safetensors", "comfy_name": "v1-5-pruned.safetensors", "type": "checkpoint", "family": "sd15"},
                {"name": "lora-misfiled.safetensors", "comfy_name": "lora-misfiled.safetensors", "type": "checkpoint", "family": "unknown", "metadata_role": "lora"},
                {"name": "unclassified.safetensors", "comfy_name": "unclassified.safetensors", "type": "checkpoint", "family": "unknown", "metadata_role": "unverified"},
                {"name": "RealESRGAN_x4plus.pth", "comfy_name": "RealESRGAN_x4plus.pth", "type": "upscale_model", "family": "unknown"},
            ],
            "model_counts": {}, "sd15": {}, "upscale": {"ready": True, "models": []},
        }
        with patch("app.main.client") as client_factory:
            client_factory.return_value.diagnostics = AsyncMock(return_value=report)
            response = TestClient(app).get("/api/diagnostics")
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["model_counts"]["upscale_models"], 1)
        self.assertEqual(result["sd15"]["checkpoint_count"], 1)
        self.assertEqual(result["sd15"]["unclassified_checkpoint_names"], ["unclassified.safetensors"])

    def test_diagnostics_marks_comfy_offline_as_blocking(self):
        with patch("app.main.client") as client_factory:
            client_factory.return_value.diagnostics = AsyncMock(return_value={"online": False, "url": "https://comfy"})
            response = TestClient(app).get("/api/diagnostics")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["issues"][0]["severity"], "BLOCKING")

    def test_comfy_models_endpoint_uses_parent_api_inventory(self):
        parent_models = [{"name": "4x-parent.pth", "comfy_name": "4x-parent.pth", "type": "upscale_model", "recognized": True}]
        with patch("app.main.client") as client_factory, patch("app.main.scan_current_models", new=AsyncMock()) as local_scan:
            client_factory.return_value.available_models = AsyncMock(return_value=parent_models)
            response = TestClient(app).get("/api/comfy/models")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), parent_models)
        client_factory.return_value.available_models.assert_awaited_once()
        local_scan.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
