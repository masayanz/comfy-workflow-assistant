import asyncio
import logging
import re
from datetime import datetime
from urllib.parse import urljoin

import httpx
from app.services.model_profile_service import ModelProfileService

logger = logging.getLogger("comfy_workflow_builder.comfy")

MODEL_FOLDER_TYPES = {
    "checkpoint": ("checkpoints",),
    "diffusion_model": ("diffusion_models", "unet"),
    "text_encoder": ("text_encoders", "clip"),
    "lora": ("loras",),
    "vae": ("vae",),
    "controlnet": ("controlnet", "t2i_adapter"),
    "upscale_model": ("upscale_models",),
}

MODEL_LOADER_INPUTS = {
    "checkpoint": ("CheckpointLoaderSimple", ("ckpt_name",)),
    "diffusion_model": ("UNETLoader", ("unet_name",)),
    "text_encoder": ("DualCLIPLoader", ("clip_name1", "clip_name2")),
    "lora": ("LoraLoader", ("lora_name",)),
    "vae": ("VAELoader", ("vae_name",)),
    "controlnet": ("ControlNetLoader", ("control_net_name",)),
    "upscale_model": ("UpscaleModelLoader", ("model_name",)),
}


def _upscale_factor(name: str) -> int | None:
    match = re.search(r"(?:^|[^0-9])([2-8])x(?:[^a-z0-9]|$)|x([2-8])(?:[^0-9]|$)", name, re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1) or match.group(2))


def _timestamp(value: object) -> str | None:
    if not isinstance(value, (int, float)):
        return None
    try:
        return datetime.fromtimestamp(value).astimezone().isoformat(timespec="seconds")
    except (OSError, OverflowError, ValueError):
        return None


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

    async def model_metadata(self, folder: str, filename: str) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                urljoin(self.base_url, f"view_metadata/{folder}"),
                params={"filename": filename},
            )
            if response.status_code == 404:
                return {}
            response.raise_for_status()
            body = response.json()
            return body if isinstance(body, dict) else {}

    async def _model_folder_catalog(self) -> list[dict]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(urljoin(self.base_url, "experiment/models"))
            if response.status_code == 404:
                return []
            response.raise_for_status()
            folders = response.json()
        if not isinstance(folders, list):
            return []
        return [item for item in folders if isinstance(item, dict) and isinstance(item.get("name"), str)]

    async def _models_in_folder(self, folder: str, *, experimental: bool) -> list[dict]:
        path = f"experiment/models/{folder}" if experimental else f"models/{folder}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(urljoin(self.base_url, path))
                if response.status_code == 404:
                    return []
                response.raise_for_status()
                body = response.json()
        except (httpx.HTTPError, ValueError):
            logger.warning("Could not list ComfyUI models in %s", folder, exc_info=True)
            return []
        if not isinstance(body, list):
            return []
        return [
            {"name": item, "size": None, "modified": None, "path_index": None}
            if isinstance(item, str) else {
                "name": item.get("name"),
                "size": item.get("size") if isinstance(item.get("size"), int) else None,
                "modified": item.get("modified"),
                "path_index": item.get("pathIndex"),
            }
            for item in body
            if isinstance(item, str) or isinstance(item, dict)
        ]

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
        catalog = await self._model_folder_catalog()
        experimental = bool(catalog)
        if not experimental:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(urljoin(self.base_url, "models"))
                response.raise_for_status()
                folder_names = response.json()
            folder_names = set(folder_names) if isinstance(folder_names, list) else set()
        else:
            folder_names = {item["name"] for item in catalog}

        categories = []
        for model_type, candidates in MODEL_FOLDER_TYPES.items():
            folders = [folder for folder in candidates if folder in folder_names]
            if not folders:
                continue
            categories.extend((model_type, folder) for folder in folders)
        folder_results = await asyncio.gather(
            *(self._models_in_folder(folder, experimental=experimental) for _, folder in categories),
            return_exceptions=True,
        )
        result = []
        from app.services.model_classifier import classify_model

        seen = set()
        for (kind, folder), folder_items in zip(categories, folder_results):
            if isinstance(folder_items, Exception):
                logger.warning("Could not read ComfyUI model folder %s: %s", folder, folder_items)
                continue
            for item in folder_items:
                name = item.get("name")
                if not isinstance(name, str) or not name:
                    continue
                key = (kind, name.replace("\\", "/"))
                if key in seen:
                    continue
                seen.add(key)
                result.append({
                    "name": name.rsplit("/", 1)[-1],
                    "comfy_name": name.replace("\\", "/"),
                    "type": kind,
                    "size": item.get("size"),
                    "path": f"comfy-api:{kind}:{name}",
                    "modified": _timestamp(item.get("modified")),
                    "extension": name.rsplit(".", 1)[-1].lower() if "." in name else "",
                    "family": classify_model(name),
                    "source": "comfy_api",
                    "inventory_source": "model_folder",
                    "recognized": True,
                    "path_index": item.get("path_index"),
                    "scale": _upscale_factor(name) if kind == "upscale_model" else None,
                })

        # Dynamic COMBO inputs such as virtual built-in VAE choices may not be files
        # in a model folder. Keep choices published by the actual loader node.
        node_data = await asyncio.gather(
            *(self.object_info(name) for name, _ in MODEL_LOADER_INPUTS.values()),
            return_exceptions=True,
        )
        for (kind, (node_name, input_names)), data in zip(MODEL_LOADER_INPUTS.items(), node_data):
            if isinstance(data, Exception) or not data:
                continue
            required = data.get("input", {}).get("required", {})
            for input_name in input_names:
                raw = required.get(input_name)
                if not isinstance(raw, list) or not raw:
                    continue
                choices = raw[0]
                if choices == "COMBO" and len(raw) > 1 and isinstance(raw[1], dict):
                    choices = raw[1].get("options", [])
                if not isinstance(choices, list):
                    continue
                for name in choices:
                    if not isinstance(name, str) or not name or (kind, name) in seen:
                        continue
                    seen.add((kind, name))
                    result.append({
                        "name": name.rsplit("/", 1)[-1],
                        "comfy_name": name.replace("\\", "/"),
                        "type": kind,
                        "size": None,
                        "path": f"comfy-api:{kind}:{name}",
                        "modified": None,
                        "extension": name.rsplit(".", 1)[-1].lower() if "." in name else "",
                        "family": classify_model(name),
                        "source": "comfy_api",
                        "inventory_source": "loader_enum",
                        "recognized": True,
                        "path_index": None,
                        "scale": _upscale_factor(name) if kind == "upscale_model" else None,
                    })
        return sorted(result, key=lambda item: (item["type"], item["name"].casefold()))

    async def diagnostics(self) -> dict:
        status = await self.status()
        if not status["online"]:
            return {"online": False, "url": status["url"]}

        async def get_json(path: str) -> dict | list:
            async with httpx.AsyncClient(timeout=max(self.timeout, 20)) as client:
                response = await client.get(urljoin(self.base_url, path))
                response.raise_for_status()
                return response.json()

        stats, catalog, node_catalog, models = await asyncio.gather(
            get_json("system_stats"), self._model_folder_catalog(), get_json("object_info"), self.available_models(),
            return_exceptions=True,
        )
        if isinstance(stats, Exception):
            logger.warning("Could not read ComfyUI system_stats: %s", stats)
            stats = {}
        if isinstance(catalog, Exception):
            logger.warning("Could not read ComfyUI model folders: %s", catalog)
            catalog = []
        if isinstance(node_catalog, Exception):
            logger.warning("Could not read ComfyUI object_info: %s", node_catalog)
            node_catalog = {}
        if isinstance(models, Exception):
            logger.warning("Could not read ComfyUI model inventory: %s", models)
            models = []

        system = stats.get("system", {}) if isinstance(stats, dict) else {}
        devices = stats.get("devices", []) if isinstance(stats, dict) else []
        argv = system.get("argv", []) if isinstance(system, dict) else []
        comfy_root = None
        for value in argv:
            if isinstance(value, str) and value.replace("\\", "/").rstrip("/").lower().endswith("/main.py"):
                comfy_root = value.rsplit("\\", 1)[0].rsplit("/", 1)[0]
                break

        folder_info = {item.get("name"): item for item in catalog if isinstance(item, dict)} if isinstance(catalog, list) else {}
        type_names = {item.get("name") for item in catalog if isinstance(item, dict)} if isinstance(catalog, list) else set()
        aliases = {"text_encoders": ("text_encoders", "clip"), "diffusion_models": ("diffusion_models", "unet"), "controlnet": ("controlnet", "t2i_adapter")}
        selected_folders = {}
        for kind, candidates in {
            "checkpoints": ("checkpoints",), "loras": ("loras",), "vae": ("vae",),
            "diffusion_models": aliases["diffusion_models"], "text_encoders": aliases["text_encoders"],
            "upscale_models": ("upscale_models",), "controlnet": aliases["controlnet"],
        }.items():
            selected_folders[kind] = [path for folder in candidates if folder in type_names for path in folder_info[folder].get("folders", [])]

        modules = {}
        for node_name, node_data in (node_catalog.items() if isinstance(node_catalog, dict) else []):
            module = node_data.get("python_module", "") if isinstance(node_data, dict) else ""
            if not isinstance(module, str) or not module.startswith("custom_nodes."):
                continue
            package = module[len("custom_nodes."):].split(".", 1)[0]
            modules.setdefault(package, set()).add(node_name)
        custom_node_packs = [{"name": name, "node_count": len(nodes)} for name, nodes in sorted(modules.items())]

        required_upscale_nodes = ("LoadImage", "UpscaleModelLoader", "ImageUpscaleWithModel", "SaveImage")
        node_names = set(node_catalog) if isinstance(node_catalog, dict) else set()
        unclassified_checkpoints = [item for item in models if item.get("type") == "checkpoint" and item.get("family") == "unknown"]
        metadata_items = [item for item in unclassified_checkpoints if item.get("extension") == "safetensors"]
        checkpoint_metadata = await asyncio.gather(
            *(self.model_metadata("checkpoints", item.get("comfy_name", item["name"]))
              for item in metadata_items),
            return_exceptions=True,
        )
        for item, metadata in zip(metadata_items, checkpoint_metadata):
            if isinstance(metadata, Exception):
                continue
            architecture = str(metadata.get("modelspec.architecture", "")).lower()
            network_module = str(metadata.get("ss_network_module", "")).lower()
            if architecture.endswith("/lora") or network_module.endswith(".lora"):
                item["metadata_role"] = "lora"
            elif "stable-diffusion-xl" in architecture:
                item["family"] = "sdxl"
                item["metadata_role"] = "checkpoint"
            elif any(marker in architecture for marker in ("stable-diffusion-v1-5", "stable-diffusion-v1.5", "sd-v1-5")):
                item["family"] = "sd15"
                item["metadata_role"] = "checkpoint"
            else:
                item["metadata_role"] = "unverified"
        upscale_models = [item for item in models if item.get("type") == "upscale_model"]
        checkpoints = [item for item in models if item.get("type") == "checkpoint"]
        sd15_checkpoints = [item for item in checkpoints if item.get("family") == "sd15"]
        unclassified_checkpoints = [item["name"] for item in checkpoints if item.get("family") == "unknown"]
        misplaced_loras = [
            {"name": item["name"], "comfy_name": item.get("comfy_name", item["name"])}
            for item in checkpoints if item.get("metadata_role") == "lora"
        ]

        try:
            flux_profile = ModelProfileService().get_profile("flux")
            missing_flux = []
            flux_matches = {}
            for component in flux_profile.ui.model_components:
                matches = [
                    item for item in models
                    if item.get("type") == component.asset_type
                    and item.get("family") == component.family
                    and (not component.name_pattern or re.search(component.name_pattern, item.get("comfy_name", item["name"]), re.IGNORECASE))
                ]
                flux_matches[component.key] = [item["name"] for item in matches]
                if component.required and not matches:
                    missing_flux.append(component.label)
        except (KeyError, ValueError):
            missing_flux = ["Flux Model Profileを読み込めません"]
            flux_matches = {}

        category_counts = {
            kind: sum(item.get("type") == model_type for item in models)
            for kind, model_type in {
                "checkpoints": "checkpoint", "loras": "lora", "vae": "vae",
                "diffusion_models": "diffusion_model", "text_encoders": "text_encoder",
                "upscale_models": "upscale_model", "controlnet": "controlnet",
            }.items()
        }
        return {
            "online": True,
            "server": {
                "version": system.get("comfyui_version"),
                "python_version": system.get("python_version"),
                "pytorch_version": system.get("pytorch_version"),
                "os": system.get("os"),
                "comfyui_root": comfy_root,
                "devices": devices,
            },
            "model_source": "comfy_api",
            "model_folder_paths": selected_folders,
            "model_counts": category_counts,
            "models": models,
            "upscale": {
                "node_available": {name: name in node_names for name in required_upscale_nodes},
                "ready": all(name in node_names for name in required_upscale_nodes) and bool(upscale_models),
                "models": upscale_models,
            },
            "sd15": {
                "checkpoint_count": len(sd15_checkpoints),
                "checkpoints": [item["name"] for item in sd15_checkpoints],
                "unclassified_checkpoint_names": unclassified_checkpoints,
            },
            "checkpoint_metadata": {
                "misplaced_loras": misplaced_loras,
                "unverified_checkpoint_names": [
                    item["name"] for item in checkpoints if item.get("metadata_role") == "unverified"
                ],
            },
            "flux": {
                "missing_assets": missing_flux,
                "available_components": flux_matches,
            },
            "custom_nodes": {
                "source": "object_info.python_module",
                "pack_count": len(custom_node_packs),
                "node_count": sum(item["node_count"] for item in custom_node_packs),
                "packs": custom_node_packs,
            },
            "node_count": len(node_names),
        }

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
