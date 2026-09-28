import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SETTINGS_PATH = ROOT / "data" / "settings.json"
DEFAULTS = {
    "comfy_url": "http://127.0.0.1:8188",
    "comfy_path": "",
    "port": 7865,
    "open_browser": True,
}


def get_settings() -> dict:
    try:
        saved = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        return {**DEFAULTS, **saved}
    except (OSError, ValueError):
        return DEFAULTS.copy()


def save_settings(settings: dict) -> dict:
    merged = {**DEFAULTS, **settings}
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = SETTINGS_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(SETTINGS_PATH)
    return merged

