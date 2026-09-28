import json
import logging
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.schemas.workflow import WorkflowBuildRequest
from app.services.comfy_client import ComfyClient
from app.services.model_scanner import custom_node_count, find_comfy_root, scan_models
from app.services.model_profile_service import ModelProfileService
from app.services.settings import get_settings, save_settings
from app.services.workflow_builder import build_definition, build_workflow, to_api_prompt, to_ui_workflow

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
    overrides = get_settings().get("model_families", {})
    for item in models:
        if item["type"] == "checkpoint" and item["comfy_name"] in overrides:
            item["family"] = overrides[item["comfy_name"]]
    return models


async def scan_current_models() -> list[dict]:
    global _models_cache
    root = current_root()
    local_models = scan_models(root) if root else []
    if local_models:
        _models_cache = local_models
    else:
        try:
            _models_cache = await client().available_models()
            logger.info("ComfyUI APIからモデルを取得しました: %d件", len(_models_cache))
        except Exception:
            logger.exception("ComfyUIからモデル一覧を取得できませんでした")
            _models_cache = []
    return apply_model_overrides(_models_cache)


async def resolve_model(name: str) -> dict:
    models = await scan_current_models()
    found = next((item for item in models if item["type"] == "checkpoint" and item.get("comfy_name", item["name"]) == name), None)
    if not found:
        raise HTTPException(status_code=400, detail="選択したCheckpointが見つかりません。モデルを再スキャンしてください。")
    return found


async def validate_lora(name: str | None) -> None:
    if not name:
        return
    models = await scan_current_models()
    available = {item.get("comfy_name", item["name"]) for item in models if item["type"] == "lora"}
    if name not in available:
        raise HTTPException(status_code=400, detail="選択したLoRAが見つかりません。モデルを再スキャンしてください。")


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(ROOT / "templates" / "index.html")


@app.get("/api/status")
async def app_status():
    models = await scan_current_models()
    settings = get_settings()
    return {
        "comfy": await client().status(),
        "models": sum(1 for item in models if item["type"] == "checkpoint"),
        "loras": sum(1 for item in models if item["type"] == "lora"),
        "custom_nodes": custom_node_count(current_root()),
        "comfy_path": current_root(),
        "model_source": "filesystem" if current_root() else "comfy_api",
        "settings": settings,
    }


@app.get("/api/comfy/status")
async def comfy_status():
    return await client().status()


@app.get("/api/models")
async def models():
    return await scan_current_models()


@app.post("/api/models/scan")
async def scan():
    items = await scan_current_models()
    return {"items": items, "comfy_path": current_root(), "source": "filesystem" if current_root() else "comfy_api"}


class ModelClassification(BaseModel):
    model: str
    family: str = Field(min_length=1, max_length=50)


@app.post("/api/models/classify")
async def classify(payload: ModelClassification):
    model = await resolve_model(payload.model)
    if payload.family != "unknown":
        try:
            ModelProfileService().get_profile(payload.family)
        except KeyError as exc:
            raise HTTPException(status_code=422, detail="選択したモデルProfileがありません。") from exc
        except ValueError as exc:
            raise HTTPException(status_code=500, detail="モデルProfileの設定を確認してください。") from exc
    settings = get_settings()
    overrides = dict(settings.get("model_families", {}))
    overrides[model["comfy_name"]] = payload.family
    settings["model_families"] = overrides
    save_settings(settings)
    model["family"] = payload.family
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
    model = await resolve_model(payload.model)
    await validate_lora(payload.lora)
    try:
        definition = build_definition(payload, model["family"])
        workflow = to_api_prompt(definition)
        ui_workflow = to_ui_workflow(definition)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    logger.info("Workflowを生成しました: %s", payload.model)
    return {"workflow": workflow, "ui_workflow": ui_workflow, "model": model, "generation_type": "txt2img"}


@app.post("/api/workflow/save")
async def save_workflow(payload: WorkflowBuildRequest):
    model = await resolve_model(payload.model)
    await validate_lora(payload.lora)
    try:
        workflow = build_workflow(payload, model["family"])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(payload.model).stem)[:40] or "workflow"
    filename = f"{stamp}_{model['family']}_{slug}.api.json"
    target = OUTPUT_DIR / filename
    target.write_text(json.dumps(workflow, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Workflowを保存しました: %s", filename)
    return {"filename": filename, "download_url": f"/api/workflow/download/{filename}"}


@app.post("/api/workflow/save-ui")
async def save_ui_workflow(payload: WorkflowBuildRequest):
    model = await resolve_model(payload.model)
    await validate_lora(payload.lora)
    try:
        definition = build_definition(payload, model["family"])
        workflow = to_ui_workflow(definition)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", Path(payload.model).stem)[:40] or "workflow"
    filename = f"{stamp}_{slug}.workflow.json"
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


@app.post("/api/workflow/run")
async def run_workflow(payload: WorkflowBuildRequest):
    status = await client().status()
    if not status["online"]:
        raise HTTPException(status_code=503, detail="ComfyUIに接続できません。設定のURLとComfyUIの起動状態を確認してください。")
    model = await resolve_model(payload.model)
    await validate_lora(payload.lora)
    try:
        workflow = build_workflow(payload, model["family"])
        comfy = client()
        seed = workflow["5"]["inputs"]["seed"]
        logger.info(
            "Sending workflow model=%s seed=%s size=%sx%s steps=%s cfg=%s sampler=%s lora=%s workflow=%s",
            payload.model, seed, payload.width, payload.height,
            payload.steps, payload.cfg, payload.sampler, payload.lora, workflow,
        )
        prompt_id = await comfy.queue(workflow)
        logger.info(
            "Workflow queued prompt_id=%s model=%s seed=%s size=%sx%s steps=%s cfg=%s sampler=%s lora=%s",
            prompt_id, payload.model, seed, payload.width, payload.height,
            payload.steps, payload.cfg, payload.sampler, payload.lora,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        logger.error("ComfyUI HTTP error status=%s response=%s", exc.response.status_code, exc.response.text[:4000])
        response_text = exc.response.text.lower()
        if "ckpt_name" in response_text or "checkpoint" in response_text:
            detail = "ComfyUIでCheckpointが見つかりません。モデル一覧を再スキャンしてください。"
        else:
            detail = "ComfyUIがWorkflowを受け付けませんでした。ComfyUI APIのエラーをログで確認してください。"
        raise HTTPException(status_code=502, detail=detail) from exc
    except Exception as exc:
        logger.exception("ComfyUI実行に失敗しました")
        detail = str(exc) if isinstance(exc, RuntimeError) else "ComfyUI APIからエラーが返されました。設定とComfyUIログを確認してください。"
        raise HTTPException(status_code=502, detail=detail) from exc
    return {"prompt_id": prompt_id, "status": "QUEUED", "seed": seed}


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
