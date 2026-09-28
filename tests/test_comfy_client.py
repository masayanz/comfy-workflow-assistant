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

    async def test_available_models_uses_comfy_object_info(self):
        def handler(request):
            node = request.url.path.rsplit("/", 1)[-1]
            fields = {
                "CheckpointLoaderSimple": ("ckpt_name",), "UNETLoader": ("unet_name",),
                "DualCLIPLoader": ("clip_name1", "clip_name2"), "LoraLoader": ("lora_name",),
                "VAELoader": ("vae_name",), "ControlNetLoader": ("control_net_name",),
                "UpscaleModelLoader": ("model_name",),
            }[node]
            values = {"ckpt_name": ["portrait.safetensors"], "unet_name": ["flux1-dev.safetensors"],
                      "clip_name1": ["clip_l.safetensors"], "clip_name2": ["t5xxl_fp8.safetensors"]}
            required = {name: [values.get(name, []), {}] for name in fields}
            return httpx.Response(200, json={node: {"input": {"required": required}}})
        transport = httpx.MockTransport(handler)
        original = httpx.AsyncClient

        def client_factory(*args, **kwargs):
            return original(*args, **{**kwargs, "transport": transport})

        with patch("app.services.comfy_client.httpx.AsyncClient", side_effect=client_factory):
            result = await ComfyClient("http://localhost:8188").available_models()
        self.assertEqual(result[0]["comfy_name"], "portrait.safetensors")
        self.assertEqual(result[0]["source"], "comfy_api")
        self.assertTrue(any(item["type"] == "diffusion_model" and item["family"] == "flux" for item in result))
        self.assertEqual({item["comfy_name"] for item in result if item["type"] == "text_encoder"}, {"clip_l.safetensors", "t5xxl_fp8.safetensors"})


if __name__ == "__main__":
    unittest.main()
