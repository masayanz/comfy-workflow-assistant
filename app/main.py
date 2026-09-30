import asyncio
import json
import io
import logging
import ntpath
import re
import time
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode, urlparse

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from app.schemas.workflow import WorkflowBuildRequest
from app.schemas.workflow_editor import WorkflowEditPrepareRequest, WorkflowEditRequest
from app.services.comfy_client import ComfyClient
from app.services.model_scanner import custom_node_count, find_comfy_root
from app.services.model_metadata import classify_asset, classify_compatibility
from app.services.model_profile_service import ModelProfileService
from app.services.settings import get_settings, save_settings
from app.services.workflow_analyzer import analyze_workflow_json
from app.services.workflow_builder import build_definition, build_workflow, to_api_prompt, to_ui_workflow
from app.services.workflow_editor import apply_workflow_patches, editable_manifest, ui_workflow_to_api_prompt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "generated_workflows"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "app.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    encoding="utf-8",
)
logger = logging.getLogger("comfy_workflow_builder")
app = FastAPI(title="Comfy Workflow Builder", version="0.1.0")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
_models_cache: list[dict] = []
_models_cache_at = 0.0
_models_cache_url = ""
_models_cache_lock = asyncio.Lock()
MODEL_CACHE_TTL_SECONDS = 15
MAX_IMAGE_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_WORKFLOW_IMPORT_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 64_000_000
UPLOAD_TTL_SECONDS = 30 * 24 * 60 * 60
COMFY_UPLOAD_SUBFOLDER = ""
_uploaded_images: dict[str, dict] = {}

IMAGE_FORMATS = {
    ".png": ("PNG", "image/png"),
    ".jpg": ("JPEG", "image/jpeg"),
    ".jpeg": ("JPEG", "image/jpeg"),
    ".webp": ("WEBP", "image/webp"),
}


@app.middleware("http")
async def limit_upload_request_body(request, call_next):
    if request.url.path == "/api/uploads/image":
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit() and int(content_length) > MAX_IMAGE_UPLOAD_BYTES + 65_536:
            return JSONResponse(status_code=413, content={"detail": "画像ファイルは20MB以下にしてください。"})
    if request.url.path == "/api/workflow/import":
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit() and int(content_length) > MAX_WORKFLOW_IMPORT_BYTES + 65_536:
            return JSONResponse(status_code=413, content={"detail": "Workflow JSONは10MB以下にしてください。"})
    if request.url.path.startswith("/api/workflow/edit/"):
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit() and int(content_length) > MAX_WORKFLOW_IMPORT_BYTES * 2 + 256 * 1024:
            return JSONResponse(status_code=413, content={"detail": "Workflowと編集内容の合計は10MB以下にしてください。"})
    return await call_next(request)


class SettingsUpdate(BaseModel):
    comfy_url: str = Field(default="http://127.0.0.1:8188", max_length=500)
    comfy_path: str = Field(default="", max_length=2000)
    port: int = Field(default=7865, ge=1024, le=65535)
    open_browser: bool = True


def client() -> ComfyClient:
    return ComfyClient(get_settings()["comfy_url"])


def current_root() -> str | None:
    settings = get_settings()
    return find_comfy_root(settings.get("comfy_path") or None)


def apply_model_overrides(models: list[dict]) -> list[dict]:
    settings = get_settings()
    overrides = settings.get("model_families", {})
    classifications = settings.get("model_classifications", {})
    for item in models:
        name = item.get("comfy_name", item.get("name", ""))
        manual = classifications.get(f"{item.get('type')}:{name}")
        if not manual and item.get("type") == "checkpoint" and name in overrides:
            manual = {"family": overrides[name]}
        if manual:
            item["classification"] = classify_asset(name, manual=manual)
            item["family"] = item["classification"]["family"]
            item["variant"] = item["classification"]["variant"]
            item["metadata_role"] = "lora" if item.get("type") == "lora" else "checkpoint"
    return models


async def parent_model_inventory(*, force_refresh: bool = False) -> list[dict]:
    global _models_cache, _models_cache_at, _models_cache_url
    comfy = client()
    cache_url = comfy.base_url
    now = time.monotonic()
    if not force_refresh and cache_url == _models_cache_url and now - _models_cache_at < MODEL_CACHE_TTL_SECONDS:
        return [dict(item) for item in _models_cache]
    async with _models_cache_lock:
        now = time.monotonic()
        if not force_refresh and cache_url == _models_cache_url and now - _models_cache_at < MODEL_CACHE_TTL_SECONDS:
            return [dict(item) for item in _models_cache]
        _models_cache = await comfy.available_models()
        _models_cache_url = cache_url
        _models_cache_at = time.monotonic()
        logger.info("親機ComfyUI APIからモデルを取得しました: %d件", len(_models_cache))
        return [dict(item) for item in _models_cache]


async def scan_current_models(*, force_refresh: bool = False) -> list[dict]:
    global _models_cache_at
    try:
        models = await parent_model_inventory(force_refresh=force_refresh)
    except Exception:
        logger.exception("親機ComfyUI APIからモデル一覧を取得できませんでした")
        if force_refresh:
            _models_cache_at = 0.0
        return []
    return apply_model_overrides(models)


async def resolve_model(name: str) -> dict:
    models = await scan_current_models()
    found = next((item for item in models if item["type"] == "checkpoint" and item.get("comfy_name", item["name"]) == name), None)
    if not found:
        raise HTTPException(status_code=400, detail="選択したCheckpointが見つかりません。モデルを再スキャンしてください。")
    return found


async def resolve_workflow_target(payload: WorkflowBuildRequest, *, require_flux_assets: bool = False):
    """Resolve a checkpoint profile or validate Flux split-model assets."""
    if payload.generation_type == "upscale":
        if not payload.upscale_model:
            raise HTTPException(status_code=400, detail="Upscale Modelを選択してください。")
        comfy = client()
        if not (await comfy.status())["online"]:
            raise HTTPException(status_code=503, detail="ComfyUIに接続できないため、Upscale Modelを確認できません。")
        try:
            model = next((item for item in await comfy.available_models()
                          if item["type"] == "upscale_model"
                          and item.get("comfy_name", item["name"]) == payload.upscale_model), None)
            if not model:
                raise HTTPException(status_code=400, detail="選択したUpscale Modelが親機ComfyUIに見つかりません。models/upscale_modelsへ配置後、再スキャンしてください。")
            missing_nodes = [name for name in ("LoadImage", "UpscaleModelLoader", "ImageUpscaleWithModel", "SaveImage")
                             if not await comfy.object_info(name)]
            if missing_nodes:
                raise HTTPException(status_code=400, detail="親機ComfyUIにUpscale用ノードがありません: " + "、".join(missing_nodes))
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("親機ComfyUIのUpscale Modelを検証できませんでした")
            raise HTTPException(status_code=502, detail="親機ComfyUIからUpscale Model一覧を取得できません。接続とComfyUIログを確認してください。") from exc
        return "upscale", model, True, []

    requested_profile = payload.profile_id
    if requested_profile == "flux":
        try:
            profile = ModelProfileService().get_profile("flux")
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=500, detail="Flux Model Profileを読み込めません。") from exc
        if not profile.enabled or payload.generation_type not in profile.supported_generation_types:
            raise HTTPException(status_code=400, detail=f"{profile.name}の{payload.generation_type}は現在対応していません。")
        models = await scan_current_models()
        missing = []
        for component in profile.ui.model_components:
            selected = getattr(payload, component.key)
            if not selected:
                if component.required:
                    missing.append(component.label)
                continue
            match = next((item for item in models
                          if item["type"] == component.asset_type
                          and item["family"] == component.family
                          and (not component.name_pattern or re.search(component.name_pattern, item.get("comfy_name", item["name"]), re.IGNORECASE))
                          and item.get("comfy_name", item["name"]) == selected), None)
            if not match:
                raise HTTPException(status_code=400, detail=f"{component.label}がComfyUIから見つかりません。モデルを再スキャンしてください。")
        if payload.clip_name1 and payload.clip_name1 == payload.clip_name2:
            raise HTTPException(status_code=400, detail="FluxのCLIP-LとT5XXLには別々のモデルを指定してください。")
        if require_flux_assets and missing:
            raise HTTPException(status_code=400, detail="Flux生成に必要なモデルが不足しています: " + "、".join(missing))
        return "flux", {"name": profile.name, "family": "flux", "type": "profile"}, not missing, missing
    model = await resolve_model(payload.model)
    if model["family"] == "flux":
        raise HTTPException(status_code=400, detail="Fluxは分割モデル構成です。Flux Profileを選択してUNET、Text Encoder、VAEを指定してください。")
    if requested_profile and requested_profile != model["family"]:
        raise HTTPException(status_code=400, detail="選択したモデルとModel Profileが一致しません。")
    try:
        profile = ModelProfileService().get_profile(model["family"])
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="選択したモデルのModel Profileを確認してください。") from exc
    if not profile.enabled or payload.generation_type not in profile.supported_generation_types:
        raise HTTPException(status_code=400, detail=f"{profile.name}の{payload.generation_type}は現在対応していません。")
    return model["family"], model, True, []


def resolve_workflow_input(payload: WorkflowBuildRequest) -> dict | None:
    if payload.generation_type == "txt2img":
        if payload.input_image_id:
            raise HTTPException(status_code=400, detail="txt2imgでは入力画像を指定できません。")
        return None
    if not payload.input_image_id:
        label = "upscale" if payload.generation_type == "upscale" else "img2img"
        raise HTTPException(status_code=400, detail=f"{label}には入力画像が必要です。画像をアップロードしてください。")
    now = time.time()
    for upload_id in [key for key, value in _uploaded_images.items() if now - value["created_at"] > UPLOAD_TTL_SECONDS]:
        _uploaded_images.pop(upload_id, None)
    image = _uploaded_images.get(payload.input_image_id)
    if not image:
        raise HTTPException(status_code=404, detail="入力画像が見つからないか期限切れです。画像を再アップロードしてください。")
    return image


def cleanup_local_uploads(now: float | None = None) -> int:
    """Delete only expired app-generated files from a confirmed local ComfyUI folder."""
    root = current_root()
    host = (urlparse(get_settings().get("comfy_url", "")).hostname or "").lower()
    if not root or host not in {"localhost", "127.0.0.1", "::1"}:
        return 0
    try:
        root_path = Path(root).resolve()
        input_root = (root_path / "input").resolve()
        upload_dir = input_root
        if not upload_dir.is_dir():
            return 0
        cutoff = (now if now is not None else time.time()) - UPLOAD_TTL_SECONDS
        removed = 0
        for path in upload_dir.iterdir():
            if path.is_symlink() or not re.fullmatch(r"cwa_[0-9a-f]{32}\.(png|jpg|webp)", path.name, re.I):
                continue
            try:
                resolved = path.resolve()
                if resolved.parent != upload_dir or not path.is_file() or path.stat().st_mtime >= cutoff:
                    continue
                path.unlink()
                removed += 1
            except OSError:
                logger.warning("Could not remove expired app upload: %s", path.name)
        return removed
    except OSError:
        logger.exception("Could not inspect local ComfyUI app upload folder")
        return 0


async def validate_and_upload_image(image: UploadFile) -> dict:
    original_name = image.filename or ""
    extension = Path(original_name.replace("\\", "/")).suffix.lower()
    expected = IMAGE_FORMATS.get(extension)
    if not expected:
        raise HTTPException(status_code=415, detail="PNG、JPEG、WEBP画像を選択してください。")
    content = await image.read(MAX_IMAGE_UPLOAD_BYTES + 1)
    if len(content) > MAX_IMAGE_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="画像ファイルは20MB以下にしてください。")
    if not content:
        raise HTTPException(status_code=400, detail="画像ファイルが空です。")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as decoded:
                actual_format = decoded.format
                width, height = decoded.size
                if getattr(decoded, "n_frames", 1) != 1:
                    raise HTTPException(status_code=415, detail="アニメーション画像はアップロードできません。静止画像を選択してください。")
                if width <= 0 or height <= 0 or width > 16_384 or height > 16_384 or width * height > MAX_IMAGE_PIXELS:
                    raise HTTPException(status_code=413, detail="画像のサイズが上限を超えています。")
                decoded.verify()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning, Warning) as exc:
        raise HTTPException(status_code=415, detail="画像データを読み取れません。PNG、JPEG、WEBPの実画像を選択してください。") from exc
    if actual_format != expected[0]:
        raise HTTPException(status_code=415, detail="ファイル拡張子と画像形式が一致しません。")
    if image.content_type and image.content_type not in {expected[1], "application/octet-stream", "image/jpg" if actual_format == "JPEG" else expected[1]}:
        raise HTTPException(status_code=415, detail="画像のContent-Typeと実際の形式が一致しません。")

    upload_id = uuid.uuid4().hex
    generated_name = f"cwa_{upload_id}{extension if extension != '.jpeg' else '.jpg'}"
    comfy = client()
    if not (await comfy.status())["online"]:
        raise HTTPException(status_code=503, detail="ComfyUIに接続できないため、入力画像を転送できません。")
    removed = cleanup_local_uploads()
    if removed:
        logger.info("Removed %d expired app-managed input image(s)", removed)
    try:
        response = await comfy.upload_image(generated_name, content, expected[1], COMFY_UPLOAD_SUBFOLDER)
    except httpx.HTTPStatusError as exc:
        logger.error("ComfyUI image upload failed: HTTP %s body=%s", exc.response.status_code, exc.response.text[:2000])
        raise HTTPException(status_code=502, detail="ComfyUIへ画像をアップロードできませんでした。接続とComfyUIログを確認してください。") from exc
    except httpx.HTTPError as exc:
        logger.exception("ComfyUI image upload failed")
        raise HTTPException(status_code=502, detail="ComfyUIへ画像をアップロードできませんでした。接続を確認してください。") from exc
    except (RuntimeError, ValueError) as exc:
        logger.exception("ComfyUI returned an invalid image upload response")
        raise HTTPException(status_code=502, detail="ComfyUIから画像アップロードの正しい応答がありませんでした。ComfyUIの状態を確認してください。") from exc
    returned_name = response["name"]
    returned_subfolder = response.get("subfolder", COMFY_UPLOAD_SUBFOLDER)
    if Path(returned_name).name != returned_name or returned_subfolder != COMFY_UPLOAD_SUBFOLDER:
        logger.error("ComfyUI returned unexpected image location: %s", response)
        raise HTTPException(status_code=502, detail="ComfyUIから安全でない画像保存先が返されました。")
    record = {
        "upload_id": upload_id,
        "comfy_name": f"{returned_subfolder}/{returned_name}" if returned_subfolder else returned_name,
        "name": ntpath.basename(original_name.replace("/", "\\"))[:255],
        "width": width,
        "height": height,
        "created_at": time.time(),
    }
    _uploaded_images[upload_id] = record
    logger.info("Input image uploaded id=%s name=%s size=%sx%s", upload_id, record["comfy_name"], width, height)
    return record


async def validate_lora(name: str | None, checkpoint: dict | str | None = None) -> dict | None:
    if not name:
        return None
    models = await scan_current_models()
    lora = next((item for item in models if item.get("type") == "lora" and item.get("comfy_name", item["name"]) == name), None)
    if not lora:
        raise HTTPException(status_code=400, detail="選択したLoRAが見つかりません。モデルを再スキャンしてください。")
    if not checkpoint:
        return None
    if isinstance(checkpoint, str):
        checkpoint = next((item for item in models if item.get("type") == "checkpoint" and item.get("comfy_name", item["name"]) == checkpoint), None)
    if not checkpoint:
        raise HTTPException(status_code=400, detail="選択したCheckpointが親機ComfyUIに見つかりません。")
    result = classify_compatibility(checkpoint, lora)
    result["checkpoint_name"] = checkpoint.get("name")
    result["lora_name"] = lora.get("name")
    if result["status"] == "incompatible":
        logger.warning("LoRA compatibility blocked checkpoint=%s lora=%s result=%s", checkpoint.get("comfy_name"), name, result)
        raise HTTPException(status_code=400, detail={
            "code": "lora_incompatible", "message": "選択したLoRAはCheckpointと互換性がありません。別のLoRAを選択してください。",
            "compatibility": result,
        })
    return result


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(ROOT / "templates" / "index.html")


@app.get("/api/status")
async def app_status():
    removed = cleanup_local_uploads()
    if removed:
        logger.info("Removed %d expired app-managed input image(s)", removed)
    models = await scan_current_models()
    settings = get_settings()
    return {
        "comfy": await client().status(),
        "models": sum(1 for item in models if item["type"] == "checkpoint"),
        "loras": sum(1 for item in models if item["type"] == "lora"),
        "custom_nodes": custom_node_count(current_root()),
        "comfy_path": current_root(),
        "model_source": "comfy_api",
        "settings": settings,
    }


@app.get("/api/diagnostics")
async def diagnostics():
    try:
        result = await client().diagnostics()
        if not result.get("online"):
            result["issues"] = [{"severity": "BLOCKING", "code": "comfy_offline", "message": "親機ComfyUIへ接続できません。URLと起動状態を確認してください。"}]
            return result
        models = apply_model_overrides(result.get("models", []))
        result["models"] = models
        counts = result.setdefault("model_counts", {})
        for label, kind in {
            "checkpoints": "checkpoint", "loras": "lora", "vae": "vae",
            "diffusion_models": "diffusion_model", "text_encoders": "text_encoder",
            "upscale_models": "upscale_model", "controlnet": "controlnet",
        }.items():
            counts[label] = sum(item.get("type") == kind for item in models)
        checkpoints = [item for item in models if item.get("type") == "checkpoint"]
        sd15 = result.setdefault("sd15", {})
        sd15_models = [item for item in checkpoints if item.get("family") == "sd15"]
        sd15["checkpoint_count"] = len(sd15_models)
        sd15["checkpoints"] = [item["name"] for item in sd15_models]
        sd15["unclassified_checkpoint_names"] = [
            item["name"] for item in checkpoints
            if item.get("family") == "unknown" and item.get("metadata_role") != "lora"
        ]
        refreshed_codes = {"checkpoint_unclassified", "sd15_checkpoint_missing"}
        result["issues"] = [issue for issue in result.get("issues", []) if issue.get("code") not in refreshed_codes]
        if sd15["unclassified_checkpoint_names"]:
            result["issues"].append({"severity": "WARNING", "code": "checkpoint_unclassified", "message": f"Checkpointの系統を判定できないファイルが{len(sd15['unclassified_checkpoint_names'])}件あります。"})
        if not sd15_models:
            result["issues"].append({"severity": "WARNING", "code": "sd15_checkpoint_missing", "message": "SD1.5 Workflowはありますが、系統を確認できるCheckpointがありません。"})
        return result
    except Exception as exc:
        logger.exception("親機ComfyUI診断を取得できませんでした")
        raise HTTPException(status_code=502, detail="ComfyUI診断情報を取得できません。接続とComfyUIログを確認してください。") from exc


@app.get("/api/comfy/status")
async def comfy_status():
    return await client().status()


@app.get("/api/models")
async def models():
    return await scan_current_models()


@app.get("/api/comfy/models")
async def comfy_models():
    """Return models recognized by the configured parent ComfyUI API."""
    try:
        return apply_model_overrides(await parent_model_inventory())
    except Exception as exc:
        logger.exception("親機ComfyUIからモデル一覧を取得できませんでした")
        raise HTTPException(status_code=502, detail="親機ComfyUIから認識済みモデル一覧を取得できません。接続とComfyUIログを確認してください。") from exc


@app.post("/api/uploads/image")
async def upload_image(image: UploadFile = File(...)):
    try:
        record = await validate_and_upload_image(image)
        return {key: value for key, value in record.items() if key != "created_at"}
    finally:
        await image.close()


@app.post("/api/models/scan")
async def scan():
    items = await scan_current_models(force_refresh=True)
    return {"items": items, "comfy_path": current_root(), "source": "comfy_api"}


class ModelClassification(BaseModel):
    model: str
    family: str = Field(min_length=1, max_length=50)
    asset_type: str = Field(default="checkpoint", min_length=1, max_length=50)
    variant: str | None = Field(default=None, max_length=50)


@app.post("/api/models/classify")
async def classify(payload: ModelClassification):
    if payload.asset_type not in {"checkpoint", "lora"}:
        raise HTTPException(status_code=422, detail="CheckpointまたはLoRAを指定してください。")
    if payload.asset_type == "checkpoint":
        model = await resolve_model(payload.model)
    else:
        models = await scan_current_models()
        model = next((item for item in models if item.get("type") == "lora" and item.get("comfy_name", item["name"]) == payload.model), None)
    if not model:
        raise HTTPException(status_code=400, detail="選択したモデルが親機ComfyUIに見つかりません。再スキャンしてください。")
    if payload.family != "unknown":
        try:
            ModelProfileService().get_profile(payload.family)
        except KeyError as exc:
            raise HTTPException(status_code=422, detail="選択したモデルProfileがありません。") from exc
        except ValueError as exc:
            raise HTTPException(status_code=500, detail="モデルProfileの設定を確認してください。") from exc
    if payload.variant and (payload.family != "sdxl" or payload.variant != "pony"):
        raise HTTPException(status_code=422, detail="variant=ponyはSDXL分類でのみ指定できます。")
    settings = get_settings()
    classifications = dict(settings.get("model_classifications", {}))
    classifications[f"{payload.asset_type}:{model['comfy_name']}"] = {"family": payload.family, "variant": payload.variant}
    settings["model_classifications"] = classifications
    if payload.asset_type == "checkpoint":
        overrides = dict(settings.get("model_families", {}))
        overrides[model["comfy_name"]] = payload.family
        settings["model_families"] = overrides
    save_settings(settings)
    model["classification"] = classify_asset(model["comfy_name"], manual=classifications[f"{payload.asset_type}:{model['comfy_name']}"])
    model["family"] = payload.family
    model["variant"] = payload.variant
    logger.info("モデル分類を更新しました: %s -> %s", model["comfy_name"], payload.family)
    return model


@app.get("/api/model-profiles")
async def model_profiles():
    try:
        return [profile.as_response() for profile in ModelProfileService().list_profiles()]
    except ValueError as exc:
        logger.exception("モデルProfileを読み込めませんでした")
        raise HTTPException(status_code=500, detail="モデルProfileの設定を確認してください。") from exc


@app.get("/api/model-profiles/{profile_id}")
async def model_profile(profile_id: str):
    try:
        return ModelProfileService().get_profile(profile_id).as_response()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="モデルProfileが見つかりません。") from exc
    except ValueError as exc:
        logger.exception("モデルProfileを読み込めませんでした: %s", profile_id)
        raise HTTPException(status_code=500, detail="モデルProfileの設定を確認してください。") from exc


@app.get("/api/loras")
async def loras():
    return [item for item in await scan_current_models() if item["type"] == "lora"]


@app.get("/api/models/compatibility")
async def lora_compatibility(checkpoint: str):
    models = await scan_current_models()
    selected = next((item for item in models if item.get("type") == "checkpoint" and item.get("comfy_name", item["name"]) == checkpoint), None)
    if not selected:
        raise HTTPException(status_code=400, detail="選択したCheckpointが親機ComfyUIに見つかりません。")
    return {
        "checkpoint": selected.get("name", checkpoint),
        "checkpoint_classification": selected.get("classification"),
        "loras": [{**lora, "compatibility": classify_compatibility(selected, lora)}
                   for lora in models if lora.get("type") == "lora"],
    }


@app.get("/api/settings")
async def settings():
    return get_settings()


@app.put("/api/settings")
async def update_settings(payload: SettingsUpdate):
    values = payload.model_dump()
    if not re.match(r"^https?://", values["comfy_url"], re.I):
        raise HTTPException(status_code=422, detail="ComfyUI URLはhttp://またはhttps://で入力してください。")
    result = save_settings({**get_settings(), **values})
    await scan_current_models()
    logger.info("設定を更新しました")
    return result


@app.post("/api/workflow/build")
async def build(payload: WorkflowBuildRequest):
    family, model, ready, missing = await resolve_workflow_target(payload)
    input_image = resolve_workflow_input(payload)
    lora_compatibility = None
    if family != "flux":
        validation = await validate_lora(payload.lora, model)
        lora_compatibility = validation if isinstance(validation, dict) else None
    try:
        definition = build_definition(payload, family, input_image["comfy_name"] if input_image else None)
        workflow = to_api_prompt(definition)
        ui_workflow = to_ui_workflow(definition)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.info("Workflowを生成しました: profile=%s model=%s ready=%s missing=%s", family, payload.model or payload.diffusion_model, ready, missing)
    return {
        "workflow": workflow, "ui_workflow": ui_workflow, "model": model,
        "generation_type": payload.generation_type, "ready": ready, "missing_assets": missing,
        "lora_compatibility": lora_compatibility,
        "input_image": {key: input_image[key] for key in ("name", "width", "height")} if input_image else None,
    }


@app.post("/api/workflow/save")
async def save_workflow(payload: WorkflowBuildRequest):
    family, model, _, _ = await resolve_workflow_target(payload, require_flux_assets=True)
    input_image = resolve_workflow_input(payload)
    if family != "flux":
        await validate_lora(payload.lora, model)
    try:
        workflow = build_workflow(payload, family, input_image["comfy_name"] if input_image else None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(payload.model or payload.diffusion_model or payload.upscale_model or family).stem)[:40] or "workflow"
    filename = f"{stamp}_{family}_{payload.generation_type}_{slug}.api.json"
    target = OUTPUT_DIR / filename
    target.write_text(json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Workflowを保存しました: %s", filename)
    return {"filename": filename, "download_url": f"/api/workflow/download/{filename}"}


@app.post("/api/workflow/save-ui")
async def save_ui_workflow(payload: WorkflowBuildRequest):
    family, model, _, _ = await resolve_workflow_target(payload, require_flux_assets=True)
    input_image = resolve_workflow_input(payload)
    if family != "flux":
        await validate_lora(payload.lora, model)
    try:
        definition = build_definition(payload, family, input_image["comfy_name"] if input_image else None)
        workflow = to_ui_workflow(definition)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(payload.model or payload.diffusion_model or payload.upscale_model or family).stem)[:40] or "workflow"
    filename = f"{stamp}_{payload.generation_type}_{slug}.workflow.json"
    target = OUTPUT_DIR / filename
    target.write_text(json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("ComfyUI UI Workflowを保存しました: %s", filename)
    return {"filename": filename, "download_url": f"/api/workflow/download/{filename}"}


@app.get("/api/workflow/download/{filename}")
async def download_workflow(filename: str):
    if not re.fullmatch(r"[\w.-]+\.json", filename):
        raise HTTPException(status_code=400, detail="ファイル名が正しくありません。")
    path = OUTPUT_DIR / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="保存したworkflowが見つかりません。")
    return FileResponse(path, filename=filename, media_type="application/json")


@app.post("/api/workflow/import")
async def import_workflow(file: UploadFile = File(...)):
    """Analyze a JSON workflow without saving it or sending it to ComfyUI."""
    try:
        filename = ntpath.basename((file.filename or "workflow.json").replace("/", "\\"))
        filename = "".join(character for character in filename if ord(character) >= 32 and ord(character) != 127)[:255] or "workflow.json"
        if Path(filename).suffix.lower() != ".json":
            raise HTTPException(status_code=415, detail="ComfyUI WorkflowのJSONファイルを選択してください。")
        raw = await file.read(MAX_WORKFLOW_IMPORT_BYTES + 1)
        if len(raw) > MAX_WORKFLOW_IMPORT_BYTES:
            raise HTTPException(status_code=413, detail="Workflow JSONは10MB以下にしてください。")

        try:
            result = analyze_workflow_json(raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        comfy_online = None
        if result.get("workflow_type") in {"comfyui_ui_workflow", "comfyui_api_prompt"}:
            inventory = None
            node_catalog = None
            comfy = client()
            status = await comfy.status()
            comfy_online = bool(status.get("online"))
            if comfy_online:
                model_result, catalog_result = await asyncio.gather(
                    parent_model_inventory(), comfy.object_info_catalog(), return_exceptions=True,
                )
                if isinstance(model_result, Exception):
                    logger.warning("Workflow import could not read parent model inventory: %s", model_result)
                else:
                    inventory = apply_model_overrides([dict(item) for item in model_result])
                if isinstance(catalog_result, Exception):
                    logger.warning("Workflow import could not read parent object_info: %s", catalog_result)
                else:
                    node_catalog = catalog_result
            result = analyze_workflow_json(raw, inventory=inventory, node_catalog=node_catalog, vram_gb=12.0)
        result["filename"] = filename
        result["comfy_online"] = comfy_online
        logger.info("Workflow JSONを解析しました: name=%s type=%s nodes=%d links=%d", filename, result["workflow_type"], result["node_count"], result["link_count"])
        return result
    finally:
        await file.close()


async def _workflow_edit_catalog():
    comfy = client()
    status = await comfy.status()
    if not status.get("online"):
        return False, None, None
    model_result, catalog_result = await asyncio.gather(
        parent_model_inventory(), comfy.object_info_catalog(), return_exceptions=True,
    )
    inventory = None
    node_catalog = None
    if isinstance(model_result, Exception):
        logger.warning("Workflow編集で親機モデル一覧を取得できません: %s", model_result)
    else:
        inventory = apply_model_overrides([dict(item) for item in model_result])
    if isinstance(catalog_result, Exception):
        logger.warning("Workflow編集で親機object_infoを取得できません: %s", catalog_result)
    else:
        node_catalog = catalog_result
    return True, inventory, node_catalog


def _workflow_from_edit_request(workflow_json: str) -> dict:
    try:
        workflow = json.loads(workflow_json)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise HTTPException(status_code=400, detail="Workflow JSONを解析できません。") from exc
    if not isinstance(workflow, dict):
        raise HTTPException(status_code=400, detail="Workflow JSONのルートはオブジェクトである必要があります。")
    return workflow


@app.post("/api/workflow/edit/prepare")
async def prepare_workflow_edit(payload: WorkflowEditPrepareRequest):
    workflow = _workflow_from_edit_request(payload.workflow_json)
    if not isinstance(workflow.get("nodes"), list) or not isinstance(workflow.get("links"), list):
        return {"editable": False, "reason": "ComfyUI API Promptは解析可能ですが、この画面での部分編集には未対応です。",
                "workflow_type": "comfyui_api_prompt", "fields": [], "unsupported_fields": []}
    online, inventory, node_catalog = await _workflow_edit_catalog()
    try:
        manifest = editable_manifest(workflow, inventory=inventory, node_catalog=node_catalog)
        initial = apply_workflow_patches(workflow, [], inventory=inventory, node_catalog=node_catalog)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    manifest["comfy_online"] = online
    manifest["validation"] = initial["validation"]
    return manifest


@app.post("/api/workflow/edit/apply")
async def apply_workflow_edit(payload: WorkflowEditRequest):
    workflow = _workflow_from_edit_request(payload.workflow_json)
    online, inventory, node_catalog = await _workflow_edit_catalog()
    try:
        result = apply_workflow_patches(
            workflow, [patch.model_dump() for patch in payload.patches],
            inventory=inventory, node_catalog=node_catalog,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409 if "編集開始時" in str(exc) or "読み直してください" in str(exc) else 400,
                            detail=str(exc)) from exc
    result["validation"]["comfy_online"] = online
    result.pop("workflow", None)
    return result


@app.post("/api/workflow/edit/queue")
async def queue_edited_workflow(payload: WorkflowEditRequest):
    workflow = _workflow_from_edit_request(payload.workflow_json)
    online, inventory, node_catalog = await _workflow_edit_catalog()
    if not online:
        raise HTTPException(status_code=503, detail="親機ComfyUIに接続できません。Queueへ送信していません。")
    try:
        edited = apply_workflow_patches(
            workflow, [patch.model_dump() for patch in payload.patches],
            inventory=inventory, node_catalog=node_catalog,
        )
        prompt = ui_workflow_to_api_prompt(edited["workflow"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    validation = edited["validation"]
    if not validation.get("can_queue"):
        raise HTTPException(status_code=422, detail={
            "message": "Workflowの検証で問題が見つかったため、Queueへ送信していません。",
            "blockers": validation.get("blockers", []), "warnings": validation.get("warnings", []),
        })
    seed = next((node["inputs"].get("seed", node["inputs"].get("noise_seed"))
                 for node in prompt.values() if node.get("class_type") in {"KSampler", "KSamplerAdvanced"}), None)
    try:
        prompt_id = await client().queue(prompt)
    except httpx.HTTPStatusError as exc:
        try:
            body = exc.response.json()
        except ValueError:
            body = {}
        logger.error("Edited ComfyUI workflow was rejected: HTTP %s response=%s node_errors=%s",
                     exc.response.status_code, exc.response.text[:4000], body.get("node_errors") if isinstance(body, dict) else None)
        raise HTTPException(status_code=502, detail="親ComfyUIが編集Workflowを受け付けませんでした。ComfyUIログを確認してください。") from exc
    except Exception as exc:
        logger.exception("Edited workflow could not be queued")
        raise HTTPException(status_code=502, detail="親ComfyUIへ編集Workflowを送信できませんでした。接続状態を確認してください。") from exc
    model_names = [item.get("name") for item in validation.get("models", [])]
    logger.info("Edited workflow queued prompt_id=%s nodes=%d models=%s seed=%s patches=%d",
                prompt_id, len(prompt), model_names, seed, len(edited["diff"]))
    return {"prompt_id": prompt_id, "status": "QUEUED", "seed": str(seed) if seed is not None else None,
            "diff": edited["diff"], "validation": validation}


@app.post("/api/workflow/run")
async def run_workflow(payload: WorkflowBuildRequest):
    status = await client().status()
    if not status["online"]:
        raise HTTPException(status_code=503, detail="ComfyUIに接続できません。設定のURLとComfyUIの起動状態を確認してください。")
    family, model, _, _ = await resolve_workflow_target(payload, require_flux_assets=True)
    input_image = resolve_workflow_input(payload)
    lora_compatibility = None
    if family != "flux":
        validation = await validate_lora(payload.lora, model)
        lora_compatibility = validation if isinstance(validation, dict) else None
    try:
        workflow = build_workflow(payload, family, input_image["comfy_name"] if input_image else None)
        comfy = client()
        sampler_node = next((node for node in workflow.values() if node["class_type"] == "KSampler"), None)
        seed = sampler_node["inputs"]["seed"] if sampler_node else None
        logger.info(
            "Sending workflow type=%s model=%s seed=%s size=%sx%s steps=%s cfg=%s sampler=%s denoise=%s input_image=%s lora=%s workflow=%s",
            payload.generation_type,
            payload.model or payload.diffusion_model or payload.upscale_model or family, seed, payload.width, payload.height,
            payload.steps, payload.cfg, payload.sampler, payload.denoise, input_image["comfy_name"] if input_image else None, payload.lora, workflow,
        )
        prompt_id = await comfy.queue(workflow)
        logger.info(
            "Workflow queued prompt_id=%s type=%s model=%s seed=%s size=%sx%s steps=%s cfg=%s sampler=%s denoise=%s input_image=%s lora=%s",
            prompt_id, payload.generation_type,
            payload.model or payload.diffusion_model or payload.upscale_model or family, seed, payload.width, payload.height,
            payload.steps, payload.cfg, payload.sampler, payload.denoise, input_image["comfy_name"] if input_image else None, payload.lora,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        logger.error("ComfyUI HTTP error status=%s response=%s", exc.response.status_code, exc.response.text[:4000])
        response_text = exc.response.text.lower()
        if "ckpt_name" in response_text or "checkpoint" in response_text:
            detail = "ComfyUIでCheckpointが見つかりません。モデル一覧を再スキャンしてください。"
        elif "upscalemodelloader" in response_text or "model_name" in response_text or "upscale model" in response_text:
            detail = "選択したUpscale Modelが親機ComfyUIから見つかりません。models/upscale_modelsへ配置後、モデルを再スキャンしてください。"
        elif any(token in response_text for token in ("unet_name", "diffusion model", "dualcliploader", "clip_name1", "clip_name2")):
            detail = "FluxのUNETまたはText EncoderがComfyUIから見つかりません。選択したFluxモデルを確認してください。"
        elif "vae_name" in response_text:
            detail = "選択したVAEがComfyUIから見つかりません。モデルを再スキャンしてください。"
        elif "loadimage" in response_text or "input image" in response_text:
            detail = "ComfyUIで入力画像を読み込めません。画像を再アップロードしてください。"
        else:
            detail = "ComfyUIがWorkflowを受け付けませんでした。ComfyUI APIのエラーをログで確認してください。"
        raise HTTPException(status_code=502, detail=detail) from exc
    except Exception as exc:
        logger.exception("ComfyUI実行に失敗しました")
        detail = str(exc) if isinstance(exc, RuntimeError) else "ComfyUI APIからエラーが返されました。設定とComfyUIログを確認してください。"
        raise HTTPException(status_code=502, detail=detail) from exc
    return {
        "prompt_id": prompt_id, "status": "QUEUED", "seed": seed,
        "generation_type": payload.generation_type,
        "input_image": {key: input_image[key] for key in ("name", "width", "height")} if input_image else None,
        "upscale_model": model.get("name") if family == "upscale" else None,
        "upscale_scale": model.get("scale") if family == "upscale" else None,
        "lora_compatibility": lora_compatibility,
        "estimated_output_size": ({"width": input_image["width"] * model["scale"], "height": input_image["height"] * model["scale"]}
                                  if family == "upscale" and model.get("scale") else None),
    }


@app.get("/api/workflow/status/{prompt_id}")
async def workflow_status(prompt_id: str):
    try:
        result = await client().prompt_status(prompt_id)
    except httpx.HTTPError as exc:
        logger.exception("Could not poll ComfyUI prompt_id=%s", prompt_id)
        raise HTTPException(status_code=502, detail="ComfyUIの実行状態を取得できません。接続を確認してください。") from exc
    if result["status"] == "ERROR":
        errors = result.get("errors", [])
        detail = json.dumps(errors, ensure_ascii=False).lower()
        if "ckpt_name" in detail or "checkpoint" in detail:
            message = "ComfyUIでCheckpointが見つかりません。選択したモデルを確認してください。"
        elif "upscalemodelloader" in detail or "imageupscalewithmodel" in detail:
            message = "Upscale Modelの読み込みまたは画像拡大に失敗しました。モデルの配置とVRAMを確認してください。"
        elif any(token in detail for token in ("unetloader", "dualcliploader", "fluxguidance", "unet_name", "clip_name1", "clip_name2")):
            message = "FluxのUNETまたはText Encoderでエラーが発生しました。Flux用モデルと接続を確認してください。"
        elif "vaeloader" in detail or "vae_name" in detail:
            message = "Flux用VAEの読み込みに失敗しました。選択したVAEを確認してください。"
        elif "loadimage" in detail or "vaeencode" in detail:
            message = "入力画像またはVAE Encodeでエラーが発生しました。入力画像とCheckpointのVAEを確認してください。"
        elif "ksampler" in detail or "sampler" in detail:
            message = "KSamplerの設定値がComfyUIで拒否されました。生成設定を確認してください。"
        else:
            message = "ComfyUIで画像生成に失敗しました。詳しい原因は logs/app.log を確認してください。"
        return {"status": "ERROR", "prompt_id": prompt_id, "message": message}
    images = [{
        **item,
        "url": "/api/comfy/image?" + urlencode({
            "filename": item.get("filename", ""),
            "subfolder": item.get("subfolder", ""),
            "type": item.get("type", "output"),
        }),
    } for item in result.get("images", [])]
    if result["status"] == "COMPLETED":
        logger.info("Workflow completed prompt_id=%s images=%s", prompt_id, [image.get("filename") for image in images])
    return {**result, "images": images}


@app.get("/api/comfy/image")
async def get_image(filename: str, subfolder: str = "", type: str = "output"):
    safe_subfolder = subfolder.replace("\\", "/")
    if (
        not filename or Path(filename).name != filename
        or type not in {"output", "temp", "input"}
        or Path(subfolder).is_absolute() or ".." in safe_subfolder.split("/")
    ):
        raise HTTPException(status_code=400, detail="画像指定が正しくありません。")
    try:
        content, content_type = await client().image(filename, subfolder, type)
        return Response(content, media_type=content_type)
    except Exception as exc:
        raise HTTPException(status_code=404, detail="ComfyUIから生成画像を取得できませんでした。") from exc
