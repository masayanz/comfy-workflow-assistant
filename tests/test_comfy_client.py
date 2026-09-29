import unittest
from unittest.mock import patch

import httpx

from app.services.comfy_client import ComfyClient


class ComfyClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_status_reports_online(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"system": {}}))
        original = httpx.AsyncClient

        def client_factory(*args, **kwargs):
            return original(*args, **{**kwargs, "transport": transport})

        with patch("app.services.comfy_client.httpx.AsyncClient", side_effect=client_factory):
            result = await ComfyClient("http://localhost:8188").status()
        self.assertTrue(result["online"])
        self.assertEqual(result["url"], "http://localhost:8188")

    async def test_queue_returns_prompt_id(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"prompt_id": "abc"}))
        original = httpx.AsyncClient

        def client_factory(*args, **kwargs):
            return original(*args, **{**kwargs, "transport": transport})

        with patch("app.services.comfy_client.httpx.AsyncClient", side_effect=client_factory):
            result = await ComfyClient("http://localhost:8188").queue({"1": {}})
        self.assertEqual(result, "abc")

    async def test_queue_rejects_comfy_node_errors(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"node_errors": {"1": {"errors": ["missing checkpoint"]}}}))
        original = httpx.AsyncClient

        def client_factory(*args, **kwargs):
            return original(*args, **{**kwargs, "transport": transport})

        with patch("app.services.comfy_client.httpx.AsyncClient", side_effect=client_factory):
            with self.assertRaisesRegex(RuntimeError, "受け付けませんでした"):
                await ComfyClient("http://localhost:8188").queue({"1": {}})

    async def test_upload_image_posts_unique_file_to_comfy_input_subfolder(self):
        def handler(request):
            self.assertEqual(request.url.path, "/upload/image")
            self.assertIn(b"name=\"type\"", request.content)
            self.assertIn(b"input", request.content)
            self.assertIn(b'name="subfolder"', request.content)
            self.assertIn(b"cwa_123.png", request.content)
            return httpx.Response(200, json={"name": "cwa_123.png", "subfolder": "", "type": "input"})
        transport = httpx.MockTransport(handler)
        original = httpx.AsyncClient

        def client_factory(*args, **kwargs):
            return original(*args, **{**kwargs, "transport": transport})

        with patch("app.services.comfy_client.httpx.AsyncClient", side_effect=client_factory):
            result = await ComfyClient("http://localhost:8188").upload_image("cwa_123.png", b"png-data", "image/png", "")
        self.assertEqual(result["type"], "input")
        self.assertEqual(result["name"], "cwa_123.png")

    async def test_prompt_status_extracts_completed_image_metadata(self):
        def handler(request):
            if request.url.path.endswith("/history/abc"):
                return httpx.Response(200, json={"abc": {
                    "status": {"status_str": "success", "completed": True, "messages": []},
                    "outputs": {"7": {"images": [{"filename": "result.png", "subfolder": "", "type": "output"}]}},
                }})
            return httpx.Response(200, json={"queue_running": [], "queue_pending": []})
        transport = httpx.MockTransport(handler)
        original = httpx.AsyncClient

        def client_factory(*args, **kwargs):
            return original(*args, **{**kwargs, "transport": transport})

        with patch("app.services.comfy_client.httpx.AsyncClient", side_effect=client_factory):
            result = await ComfyClient("http://localhost:8188").prompt_status("abc")
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual(result["images"][0]["filename"], "result.png")

    async def test_prompt_status_detects_comfy_error(self):
        def handler(request):
            if request.url.path.endswith("/history/bad"):
                return httpx.Response(200, json={"bad": {"status": {
                    "status_str": "error", "messages": [["execution_error", {"node_type": "CheckpointLoaderSimple"}]],
                }}})
            return httpx.Response(200, json={"queue_running": [], "queue_pending": []})
        transport = httpx.MockTransport(handler)
        original = httpx.AsyncClient

        def client_factory(*args, **kwargs):
            return original(*args, **{**kwargs, "transport": transport})

        with patch("app.services.comfy_client.httpx.AsyncClient", side_effect=client_factory):
            result = await ComfyClient("http://localhost:8188").prompt_status("bad")
        self.assertEqual(result["status"], "ERROR")
        self.assertIn("CheckpointLoaderSimple", str(result["errors"]))

    async def test_available_models_uses_parent_folders_and_exposes_file_sizes(self):
        folder_names = ["checkpoints", "diffusion_models", "text_encoders", "loras", "vae", "controlnet", "upscale_models"]
        files = {
            "checkpoints": [{"name": "portrait.safetensors", "size": 123, "modified": 1700000000, "pathIndex": 0}],
            "diffusion_models": [{"name": "flux1-dev.safetensors", "size": 456, "modified": 1700000000, "pathIndex": 0}],
            "text_encoders": [{"name": "clip_l.safetensors", "size": 789, "modified": 1700000000, "pathIndex": 0},
                               {"name": "t5xxl_fp8.safetensors", "size": 987, "modified": 1700000000, "pathIndex": 0}],
            "loras": [], "vae": [],
            "controlnet": [], "upscale_models": [{"name": "4x-UltraSharp.pth", "size": 4000, "modified": 1700000000, "pathIndex": 0}],
        }
        def handler(request):
            path = request.url.path
            if path == "/experiment/models":
                return httpx.Response(200, json=[{"name": name, "folders": [], "extensions": []} for name in folder_names])
            if path.startswith("/experiment/models/"):
                folder = path.rsplit("/", 1)[-1]
                return httpx.Response(200, json=files[folder])
            if path.startswith("/object_info/"):
                node = path.rsplit("/", 1)[-1]
                if node == "VAELoader":
                    required = {"vae_name": [["pixel_space"], {}]}
                elif node == "UpscaleModelLoader":
                    required = {"model_name": ["COMBO", {"options": []}]}
                else:
                    required = {}
                return httpx.Response(200, json={node: {"input": {"required": required}}})
            return httpx.Response(404)
        transport = httpx.MockTransport(handler)
        original = httpx.AsyncClient

        def client_factory(*args, **kwargs):
            return original(*args, **{**kwargs, "transport": transport})

        with patch("app.services.comfy_client.httpx.AsyncClient", side_effect=client_factory):
            result = await ComfyClient("http://localhost:8188").available_models()
        checkpoint = next(item for item in result if item["type"] == "checkpoint")
        self.assertEqual(checkpoint["comfy_name"], "portrait.safetensors")
        self.assertEqual(checkpoint["source"], "comfy_api")
        self.assertEqual(checkpoint["size"], 123)
        vae_option = next(item for item in result if item["type"] == "vae" and item["name"] == "pixel_space")
        self.assertEqual(vae_option["inventory_source"], "loader_enum")
        self.assertIsNone(vae_option["size"])
        self.assertTrue(any(item["type"] == "diffusion_model" and item["family"] == "flux" for item in result))
        self.assertEqual({item["comfy_name"] for item in result if item["type"] == "text_encoder"}, {"clip_l.safetensors", "t5xxl_fp8.safetensors"})
        upscale = next(item for item in result if item["type"] == "upscale_model")
        self.assertEqual(upscale["size"], 4000)
        self.assertEqual(upscale["scale"], 4)
        self.assertEqual(upscale["inventory_source"], "model_folder")
        self.assertTrue(upscale["recognized"])

    async def test_diagnostics_reads_checkpoint_metadata_and_flags_misfiled_lora(self):
        folder_names = ["checkpoints", "upscale_models"]
        folder_paths = {
            "checkpoints": [{"name": "checkpoints", "folders": ["E:\\ComfyUI\\models\\checkpoints"], "extensions": [".safetensors"]}],
            "upscale_models": [{"name": "upscale_models", "folders": ["E:\\ComfyUI\\models\\upscale_models"], "extensions": [".pth"]}],
        }
        files = {
            "checkpoints": [
                {"name": "unknown.safetensors", "size": 100, "modified": 1700000000, "pathIndex": 0},
                {"name": "lora-in-checkpoints.safetensors", "size": 200, "modified": 1700000000, "pathIndex": 0},
            ],
            "upscale_models": [],
        }
        custom_node = {"python_module": "custom_nodes.example_pack.nodes"}

        def handler(request):
            path = request.url.path
            if path == "/system_stats":
                return httpx.Response(200, json={"system": {"argv": ["E:\\ComfyUI\\main.py"], "comfyui_version": "0.37.0"}, "devices": []})
            if path == "/experiment/models":
                return httpx.Response(200, json=[folder_paths[name][0] for name in folder_names])
            if path.startswith("/experiment/models/"):
                return httpx.Response(200, json=files[path.rsplit("/", 1)[-1]])
            if path == "/object_info":
                nodes = {name: {"python_module": "nodes"} for name in ("LoadImage", "UpscaleModelLoader", "ImageUpscaleWithModel", "SaveImage")}
                nodes["ExampleNode"] = custom_node
                return httpx.Response(200, json=nodes)
            if path.startswith("/object_info/"):
                node = path.rsplit("/", 1)[-1]
                required = {"model_name": ["COMBO", {"options": []}]} if node == "UpscaleModelLoader" else {}
                return httpx.Response(200, json={node: {"input": {"required": required}}})
            if path == "/view_metadata/checkpoints":
                filename = request.url.params.get("filename", "")
                if filename == "lora-in-checkpoints.safetensors":
                    return httpx.Response(200, json={"ss_network_module": "networks.lora"})
                return httpx.Response(200, json={})
            return httpx.Response(404)

        transport = httpx.MockTransport(handler)
        original = httpx.AsyncClient

        def client_factory(*args, **kwargs):
            return original(*args, **{**kwargs, "transport": transport})

        with patch("app.services.comfy_client.httpx.AsyncClient", side_effect=client_factory):
            result = await ComfyClient("http://localhost:8188").diagnostics()
        self.assertTrue(result["online"])
        self.assertEqual(result["server"]["comfyui_root"], "E:\\ComfyUI")
        self.assertEqual(result["checkpoint_metadata"]["misplaced_loras"], [
            {"name": "lora-in-checkpoints.safetensors", "comfy_name": "lora-in-checkpoints.safetensors"},
        ])
        self.assertEqual(result["custom_nodes"]["packs"][0]["name"], "example_pack")


if __name__ == "__main__":
    unittest.main()
