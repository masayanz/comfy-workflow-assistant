from typing import Literal

from pydantic import BaseModel, Field, model_validator


class WorkflowBuildRequest(BaseModel):
    model: str = ""
    profile_id: str | None = None
    diffusion_model: str | None = None
    clip_name1: str | None = None
    clip_name2: str | None = None
    vae_model: str | None = None
    generation_type: Literal["txt2img", "img2img", "upscale"] = "txt2img"
    input_image_id: str | None = None
    upscale_model: str | None = None
    denoise: float = Field(default=0.5, ge=0.0, le=1.0)
    prompt: str = Field(default="", max_length=12000)
    negative_prompt: str = ""
    width: int | None = Field(default=None, ge=64, le=4096)
    height: int | None = Field(default=None, ge=64, le=4096)
    steps: int | None = Field(default=None, ge=1, le=150)
    cfg: float | None = Field(default=None, ge=0, le=50)
    guidance: float | None = Field(default=None, ge=0, le=100)
    sampler: str | None = None
    scheduler: str | None = None
    seed: int | None = None
    lora: str | None = None
    lora_weight: float = Field(default=0.8, ge=-2, le=2)

    @model_validator(mode="after")
    def require_prompt_for_generation(self):
        if self.generation_type != "upscale" and not self.prompt.strip():
            raise ValueError("txt2imgとimg2imgにはPromptが必要です。")
        return self
