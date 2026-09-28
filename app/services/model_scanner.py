from datetime import datetime
from pathlib import Path

MODEL_DIRS = {
    "checkpoint": "checkpoints",
    "diffusion_model": "diffusion_models",
    "text_encoder": "text_encoders",
    "lora": "loras",
    "vae": "vae",
    "controlnet": "controlnet",
    "upscale_model": "upscale_models",
}
MODEL_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}


def scan_models(comfy_root: str | Path | None) -> list[dict]:
    if not comfy_root:
        return []
    root = Path(comfy_root).expanduser()
    models_root = root / "models"
    if not models_root.is_dir():
        return []
    from app.services.model_classifier import classify_model

    result = []
    for kind, relative in MODEL_DIRS.items():
        directory = models_root / relative
        try:
            directory = directory.resolve()
        except OSError:
            continue
        if not directory.is_dir():
            continue
        try:
            paths = directory.rglob("*")
            for path in paths:
                if not path.is_file() or path.suffix.lower() not in MODEL_EXTENSIONS:
                    continue
                try:
                    stat = path.stat()
                    result.append({
                        "name": path.name,
                        "comfy_name": path.relative_to(directory).as_posix(),
                        "type": kind,
                        "size": stat.st_size,
                        "path": str(path.resolve()),
                        "modified": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
                        "extension": path.suffix.lower(),
                        "family": classify_model(path.name, path.parent),
                    })
                except OSError:
                    continue
        except OSError:
            continue
    return sorted(result, key=lambda item: (item["type"], item["name"].casefold()))


def find_comfy_root(configured: str | None = None) -> str | None:
    candidates = [configured] if configured else []
    candidates += [
        str(Path.home() / "ComfyUI"),
        str(Path.home() / "StabilityMatrix" / "Data" / "Packages" / "ComfyUI"),
        str(Path.home() / "AppData" / "Roaming" / "StabilityMatrix" / "Data" / "Packages" / "ComfyUI"),
    ]
    for candidate in candidates:
        if candidate and (Path(candidate).expanduser() / "models").is_dir():
            return str(Path(candidate).expanduser().resolve())
    return None


def custom_node_count(comfy_root: str | None) -> int:
    if not comfy_root:
        return 0
    directory = Path(comfy_root or "") / "custom_nodes"
    try:
        return sum(1 for path in directory.iterdir() if path.is_dir())
    except OSError:
        return 0
