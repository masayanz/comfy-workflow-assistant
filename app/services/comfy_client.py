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

    async def available_models(self) -> list[dict]:
        model_nodes = {
            "CheckpointLoaderSimple": ("checkpoint", "ckpt_name"),
            "LoraLoader": ("lora", "lora_name"),
            "VAELoader": ("vae", "vae_name"),
            "ControlNetLoader": ("controlnet", "control_net_name"),
            "UpscaleModelLoader": ("upscale_model", "model_name"),
        }
        node_data = await asyncio.gather(*(self.object_info(name) for name in model_nodes), return_exceptions=True)
        result = []
        from app.services.model_classifier import classify_model

        for (node_name, (kind, input_name)), data in zip(model_nodes.items(), node_data):
            if isinstance(data, Exception):
                logger.warning("Could not read ComfyUI node info for %s: %s", node_name, data)
                continue
            choices = data.get("input", {}).get("required", {}).get(input_name, [[]])[0]
            if not isinstance(choices, list):
                continue
            for name in choices:
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
