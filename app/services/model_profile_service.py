import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE_DIR = ROOT / "model_profiles"


class ModelCapabilities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    txt2img: bool = False
    img2img: bool = False
    lora: bool = False
    controlnet: bool = False
    upscale: bool = False


class ModelComponentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(pattern=r"^(diffusion_model|clip_name1|clip_name2|vae_model)$")
    label: str = Field(min_length=1, max_length=100)
    asset_type: str = Field(pattern=r"^(diffusion_model|text_encoder|vae)$")
    family: str = Field(min_length=1, max_length=50)
    required: bool = True
    name_pattern: str | None = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def validate_name_pattern(self):
        if self.name_pattern:
            try:
                re.compile(self.name_pattern)
            except re.error as exc:
                raise ValueError("model componentのname_patternが不正です") from exc
        return self


class ModelProfileUI(BaseModel):
    model_config = ConfigDict(extra="forbid")

    show_cfg: bool = True
    show_negative_prompt: bool = True
    show_guidance: bool = False
    model_components: list[ModelComponentInput] = Field(default_factory=list)


class ModelProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    name: str = Field(min_length=1, max_length=100)
    enabled: bool
    default_width: int | None = Field(default=None, ge=64, le=4096)
    default_height: int | None = Field(default=None, ge=64, le=4096)
    default_steps: int | None = Field(default=None, ge=1, le=150)
    default_cfg: float | None = Field(default=None, ge=0, le=50)
    default_sampler: str | None = Field(default=None, min_length=1, max_length=100)
    default_scheduler: str | None = Field(default=None, min_length=1, max_length=100)
    supported_generation_types: list[str] = Field(default_factory=list)
    supports_lora: bool
    capabilities: ModelCapabilities
    architecture: str = "checkpoint"
    default_guidance: float | None = Field(default=None, ge=0, le=100)
    resolution_multiple: int = Field(default=1, ge=1, le=128)
    ui: ModelProfileUI = Field(default_factory=ModelProfileUI)

    @model_validator(mode="after")
    def validate_capabilities(self):
        supported = {"txt2img", "img2img"}
        if set(self.supported_generation_types) - supported:
            raise ValueError("supported_generation_typesに未対応の値があります")
        if len(set(self.supported_generation_types)) != len(self.supported_generation_types):
            raise ValueError("supported_generation_typesに重複があります")
        if self.capabilities.txt2img != ("txt2img" in self.supported_generation_types):
            raise ValueError("txt2img capabilityとsupported_generation_typesが一致しません")
        if self.capabilities.img2img != ("img2img" in self.supported_generation_types):
            raise ValueError("img2img capabilityとsupported_generation_typesが一致しません")
        if self.capabilities.lora != self.supports_lora:
            raise ValueError("lora capabilityとsupports_loraが一致しません")
        if self.architecture not in {"checkpoint", "flux_split"}:
            raise ValueError("architectureに未対応の値があります")
        if self.architecture == "flux_split" and self.capabilities.txt2img and self.default_guidance is None:
            raise ValueError("Flux Profileにはdefault_guidanceが必要です")
        if len({component.key for component in self.ui.model_components}) != len(self.ui.model_components):
            raise ValueError("ui.model_componentsのkeyが重複しています")
        if self.architecture == "flux_split" and self.capabilities.txt2img:
            required_roles = {"diffusion_model", "clip_name1", "clip_name2", "vae_model"}
            configured_roles = {component.key for component in self.ui.model_components if component.required}
            if configured_roles != required_roles:
                raise ValueError("Flux Profileには必須モデル構成4種が必要です")
        defaults = (self.default_width, self.default_height, self.default_steps, self.default_cfg, self.default_sampler, self.default_scheduler)
        if self.enabled and (not self.supported_generation_types or any(value is None for value in defaults)):
            raise ValueError("有効なProfileには対応生成タイプと全デフォルト値が必要です")
        if not self.enabled and any(value is not None for value in defaults) and any(value is None for value in defaults):
            raise ValueError("Profileのデフォルト値はすべて設定するか、すべて省略してください")
        return self

    def as_response(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "enabled": self.enabled,
            "defaults": {
                "width": self.default_width,
                "height": self.default_height,
                "steps": self.default_steps,
                "cfg": self.default_cfg,
                "sampler": self.default_sampler,
                "scheduler": self.default_scheduler,
            },
            "supported_generation_types": list(self.supported_generation_types),
            "supports_lora": self.supports_lora,
            "capabilities": self.capabilities.model_dump(),
            "architecture": self.architecture,
            "default_guidance": self.default_guidance,
            "resolution_multiple": self.resolution_multiple,
            "ui": self.ui.model_dump(),
        }


class ModelProfileService:
    def __init__(self, profile_dir: str | Path = DEFAULT_PROFILE_DIR):
        self.profile_dir = Path(profile_dir)

    def list_profiles(self) -> list[ModelProfile]:
        if not self.profile_dir.is_dir():
            raise ValueError(f"モデルプロファイルのディレクトリがありません: {self.profile_dir}")
        profiles = []
        ids = set()
        for path in sorted(self.profile_dir.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                profile = ModelProfile.model_validate(raw)
            except (OSError, json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(f"モデルプロファイルが不正です ({path.name}): {exc}") from exc
            if profile.id != path.stem:
                raise ValueError(f"モデルプロファイルIDとファイル名が一致しません: {path.name}")
            if profile.id in ids:
                raise ValueError(f"モデルプロファイルIDが重複しています: {profile.id}")
            ids.add(profile.id)
            profiles.append(profile)
        if not profiles:
            raise ValueError(f"モデルプロファイルがありません: {self.profile_dir}")
        return profiles

    def get_profile(self, profile_id: str) -> ModelProfile:
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", profile_id):
            raise KeyError(profile_id)
        for profile in self.list_profiles():
            if profile.id == profile_id:
                return profile
        raise KeyError(profile_id)
