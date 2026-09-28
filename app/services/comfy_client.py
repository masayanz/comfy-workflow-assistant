import asyncio
import logging
from urllib.parse import urljoin

import httpx

logger = logging.getLogger("comfy_workflow_builder.comfy")


class ComfyClient:
    def __init__(self, base_url: str, timeout: float = 10):
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = timeout

    async def status(self) -> dict:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(urljoin(self.base_url, "system_stats"))
                response.raise_for_status()
            return {"online": True, "url": self.base_url.rstrip("/")}
        except (httpx.HTTPError, ValueError):
            return {"online": False, "url": self.base_url.rstrip("/")}

    async def queue(self, workflow: dict) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(urljoin(self.base_url, "prompt"), json={"prompt": workflow})
                response.raise_for_status()
                body = response.json()
            except httpx.HTTPStatusError as exc:
                logger.error("ComfyUI /prompt rejected request: HTTP %s body=%s workflow=%s", exc.response.status_code, exc.response.text[:4000], workflow)
                raise
        if body.get("node_errors"):
            logger.error("ComfyUI node_errors=%s workflow=%s", body["node_errors"], workflow)
            raise RuntimeError("ComfyUIがワークフローを受け付けませんでした。必要なノードやモデルを確認してください。")
        prompt_id = body.get("prompt_id")
        if not prompt_id:
            raise RuntimeError("ComfyUIから実行IDが返されませんでした。")
        return prompt_id

    async def object_info(self, node_name: str) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(urljoin(self.base_url, f"object_info/{node_name}"))
            response.raise_for_status()
            return response.json().get(node_name, {})

    async def upload_image(self, filename: str, content: bytes, content_type: str, subfolder: str) -> dict:
        """Upload a validated image to ComfyUI's input folder for LoadImage."""
        async with httpx.AsyncClient(timeout=max(self.timeout, 60)) as client:
            response = await client.post(
                urljoin(self.base_url, "upload/image"),
                data={"type": "input", "subfolder": subfolder, "overwrite": "false"},
                files={"image": (filename, content, content_type)},
            )
            response.raise_for_status()
            body = response.json()
        if not isinstance(body, dict) or not isinstance(body.get("name"), str):
            raise RuntimeError("ComfyUIからアップロード画像名が返されませんでした。")
        return body

    async def available_models(self) -> list[dict]:
        model_nodes = {
            "CheckpointLoaderSimple": ("checkpoint", ("ckpt_name",)),
            "UNETLoader": ("diffusion_model", ("unet_name",)),
            "DualCLIPLoader": ("text_encoder", ("clip_name1", "clip_name2")),
            "LoraLoader": ("lora", ("lora_name",)),
            "VAELoader": ("vae", ("vae_name",)),
            "ControlNetLoader": ("controlnet", ("control_net_name",)),
            "UpscaleModelLoader": ("upscale_model", ("model_name",)),
        }
        node_data = await asyncio.gather(*(self.object_info(name) for name in model_nodes), return_exceptions=True)
        result = []
        from app.services.model_classifier import classify_model

        for (node_name, (kind, input_names)), data in zip(model_nodes.items(), node_data):
            if isinstance(data, Exception):
                logger.warning("Could not read ComfyUI node info for %s: %s", node_name, data)
                continue
            inputs = data.get("input", {}).get("required", {})
            names = set()
            for input_name in input_names:
                choices = inputs.get(input_name, [[]])[0]
                if isinstance(choices, list):
                    names.update(name for name in choices if isinstance(name, str))
            for name in sorted(names):
                if not isinstance(name, str):
                    continue
                result.append({
                    "name": name.rsplit("/", 1)[-1],
                    "comfy_name": name,
                    "type": kind,
                    "size": None,
                    "path": f"comfy-api:{kind}:{name}",
                    "modified": None,
                    "extension": name.rsplit(".", 1)[-1].lower() if "." in name else "",
                    "family": classify_model(name),
                    "source": "comfy_api",
                })
        return sorted(result, key=lambda item: (item["type"], item["name"].casefold()))

    async def prompt_status(self, prompt_id: str) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            history_response, queue_response = await asyncio.gather(
                client.get(urljoin(self.base_url, f"history/{prompt_id}")),
                client.get(urljoin(self.base_url, "queue")),
            )
            history_response.raise_for_status()
            queue_response.raise_for_status()
            history = history_response.json().get(prompt_id)
            queue = queue_response.json()
        if history:
            status = history.get("status", {})
            messages = status.get("messages", [])
            if status.get("status_str") == "error":
                logger.error("ComfyUI execution error prompt_id=%s status=%s", prompt_id, status)
                return {"status": "ERROR", "prompt_id": prompt_id, "errors": messages}
            images = [image for output in history.get("outputs", {}).values() for image in output.get("images", [])]
            if status.get("completed") or (status.get("status_str") == "success" and images):
                return {"status": "COMPLETED", "prompt_id": prompt_id, "images": images}
        if any(len(item) > 1 and item[1] == prompt_id for item in queue.get("queue_running", [])):
            return {"status": "RUNNING", "prompt_id": prompt_id}
        if any(len(item) > 1 and item[1] == prompt_id for item in queue.get("queue_pending", [])):
            return {"status": "QUEUED", "prompt_id": prompt_id}
        return {"status": "QUEUED", "prompt_id": prompt_id}

    async def image(self, filename: str, subfolder: str = "", image_type: str = "output") -> tuple[bytes, str]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(urljoin(self.base_url, "view"), params={
                "filename": filename, "subfolder": subfolder, "type": image_type,
            })
            response.raise_for_status()
            return response.content, response.headers.get("content-type", "image/png")
