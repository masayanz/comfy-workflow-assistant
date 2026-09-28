from pydantic import BaseModel, Field


class WorkflowBuildRequest(BaseModel):
    model: str
    prompt: str = Field(min_length=1, max_length=12000)
    negative_prompt: str = ""
    width: int = Field(default=1024, ge=64, le=4096)
    height: int = Field(default=1024, ge=64, le=4096)
    steps: int = Field(default=28, ge=1, le=150)
    cfg: float = Field(default=6.0, ge=0, le=50)
    sampler: str = "euler"
    seed: int | None = None
    lora: str | None = None
    lora_weight: float = Field(default=0.8, ge=-2, le=2)

