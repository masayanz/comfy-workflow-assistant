from typing import Any

from pydantic import BaseModel, Field


class WorkflowEditPatch(BaseModel):
    node_id: str = Field(min_length=1, max_length=128)
    field: str = Field(min_length=1, max_length=64)
    expected_old: Any
    new_value: Any = None


class WorkflowEditRequest(BaseModel):
    workflow_json: str = Field(min_length=2, max_length=10 * 1024 * 1024)
    patches: list[WorkflowEditPatch] = Field(default_factory=list, max_length=500)


class WorkflowEditPrepareRequest(BaseModel):
    workflow_json: str = Field(min_length=2, max_length=10 * 1024 * 1024)
