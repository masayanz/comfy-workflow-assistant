from dataclasses import dataclass
import json
from pathlib import Path
import secrets

from app.schemas.workflow import WorkflowBuildRequest

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Connection:
    node_id: int
    output_index: int


@dataclass(frozen=True)
class WorkflowNodeDefinition:
    node_id: int
    node_type: str
    inputs: dict[str, object]


@dataclass(frozen=True)
class WorkflowDefinition:
    """Format-independent logical graph used to create both ComfyUI formats."""

    family: str
    nodes: tuple[WorkflowNodeDefinition, ...]


@dataclass(frozen=True)
class UiNodeSpec:
    input_sockets: tuple[tuple[str, str], ...]
    output_sockets: tuple[tuple[str, str], ...]
    widget_order: tuple[str, ...] = ()
    has_output: bool = True


UI_NODE_SPECS = {
    "CheckpointLoaderSimple": UiNodeSpec((), (("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")), ("ckpt_name",)),
    "CLIPTextEncode": UiNodeSpec((("clip", "CLIP"),), (("CONDITIONING", "CONDITIONING"),), ("text",)),
    "EmptyLatentImage": UiNodeSpec((), (("LATENT", "LATENT"),), ("width", "height", "batch_size")),
    "KSampler": UiNodeSpec(
        (("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"), ("latent_image", "LATENT")),
        (("LATENT", "LATENT"),),
        ("seed", "seed_control", "steps", "cfg", "sampler_name", "scheduler", "denoise"),
    ),
    "VAEDecode": UiNodeSpec((("samples", "LATENT"), ("vae", "VAE")), (("IMAGE", "IMAGE"),)),
    "SaveImage": UiNodeSpec((("images", "IMAGE"),), (), ("filename_prefix",), has_output=False),
    "LoraLoader": UiNodeSpec((("model", "MODEL"), ("clip", "CLIP")), (("MODEL", "MODEL"), ("CLIP", "CLIP")), ("lora_name", "strength_model", "strength_clip")),
}


def build_definition(request: WorkflowBuildRequest, family: str = "sdxl") -> WorkflowDefinition:
    if family not in {"sd15", "sdxl"}:
        raise ValueError("txt2imgはSD1.5またはSDXL Checkpointを選択してください。")

    nodes = [
        WorkflowNodeDefinition(1, "CheckpointLoaderSimple", {"ckpt_name": request.model}),
        WorkflowNodeDefinition(2, "CLIPTextEncode", {"text": request.prompt, "clip": Connection(1, 1)}),
        WorkflowNodeDefinition(3, "CLIPTextEncode", {"text": request.negative_prompt, "clip": Connection(1, 1)}),
        WorkflowNodeDefinition(4, "EmptyLatentImage", {"width": request.width, "height": request.height, "batch_size": 1}),
        WorkflowNodeDefinition(5, "KSampler", {
            "seed": request.seed if request.seed is not None else secrets.randbelow(2**32),
            "seed_control": "fixed",
            "steps": request.steps,
            "cfg": request.cfg,
            "sampler_name": request.sampler,
            "scheduler": "normal",
            "denoise": 1.0,
            "model": Connection(1, 0),
            "positive": Connection(2, 0),
            "negative": Connection(3, 0),
            "latent_image": Connection(4, 0),
        }),
        WorkflowNodeDefinition(6, "VAEDecode", {"samples": Connection(5, 0), "vae": Connection(1, 2)}),
        WorkflowNodeDefinition(7, "SaveImage", {"images": Connection(6, 0), "filename_prefix": "ComfyWorkflowBuilder"}),
    ]

    if request.lora:
        nodes.append(WorkflowNodeDefinition(8, "LoraLoader", {
            "model": Connection(1, 0),
            "clip": Connection(1, 1),
            "lora_name": request.lora,
            "strength_model": request.lora_weight,
            "strength_clip": request.lora_weight,
        }))
        nodes[1].inputs["clip"] = Connection(8, 1)
        nodes[2].inputs["clip"] = Connection(8, 1)
        nodes[4].inputs["model"] = Connection(8, 0)

    definition = WorkflowDefinition(family=family, nodes=tuple(nodes))
    validate_definition(definition)
    return definition


def to_api_prompt(definition: WorkflowDefinition) -> dict:
    prompt = {}
    for node in definition.nodes:
        prompt[str(node.node_id)] = {
            "class_type": node.node_type,
            "inputs": {
                name: [str(value.node_id), value.output_index] if isinstance(value, Connection) else value
                for name, value in node.inputs.items()
                if name != "seed_control"
            },
        }
    validate_workflow(prompt)
    return prompt


def to_ui_workflow(definition: WorkflowDefinition) -> dict:
    if definition.family != "sdxl":
        raise ValueError("ComfyUIキャンバス用Workflow JSONの作成は現在SDXL txt2imgに対応しています。")

    links = []
    next_link_id = 1
    for node in definition.nodes:
        for input_name, value in node.inputs.items():
            if not isinstance(value, Connection):
                continue
            target_spec = UI_NODE_SPECS[node.node_type]
            target_slot = next(index for index, item in enumerate(target_spec.input_sockets) if item[0] == input_name)
            origin_node = next(item for item in definition.nodes if item.node_id == value.node_id)
            origin_spec = UI_NODE_SPECS[origin_node.node_type]
            output_name, output_type = origin_spec.output_sockets[value.output_index]
            links.append({
                "id": next_link_id,
                "origin_id": value.node_id,
                "origin_slot": value.output_index,
                "target_id": node.node_id,
                "target_slot": target_slot,
                "type": output_type,
            })
            next_link_id += 1

    nodes = []
    execution_order = _execution_order(definition)
    for definition_node in definition.nodes:
        spec = UI_NODE_SPECS[definition_node.node_type]
        node_links = [link for link in links if link["origin_id"] == definition_node.node_id]
        incoming_links = [link for link in links if link["target_id"] == definition_node.node_id]
        inputs = []
        for slot_index, (name, socket_type) in enumerate(spec.input_sockets):
            link = next((item for item in incoming_links if item["target_slot"] == slot_index), None)
            inputs.append({"name": name, "type": socket_type, "link": link["id"] if link else None})
        outputs = []
        for slot_index, (name, socket_type) in enumerate(spec.output_sockets):
            slot_links = [link["id"] for link in node_links if link["origin_slot"] == slot_index]
            outputs.append({"name": name, "type": socket_type, "links": slot_links or None, "slot_index": slot_index})

        node = {
            "id": definition_node.node_id,
            "type": definition_node.node_type,
            "pos": _node_position(definition, definition_node.node_id),
            "size": _node_size(definition_node.node_type),
            "flags": {},
            "order": execution_order[definition_node.node_id],
            "mode": 0,
            "inputs": inputs,
            "outputs": outputs,
            "properties": {"Node name for S&R": definition_node.node_type},
            "widgets_values": [definition_node.inputs[name] for name in spec.widget_order],
        }
        if not spec.has_output:
            node["outputs"] = []
        nodes.append(node)

    workflow = {
        "last_node_id": max(node.node_id for node in definition.nodes),
        "last_link_id": len(links),
        "nodes": nodes,
        "links": [[link[key] for key in ("id", "origin_id", "origin_slot", "target_id", "target_slot", "type")] for link in links],
        "groups": [],
        "config": {},
        "extra": {"ds": {"scale": 0.55, "offset": [0, 0]}},
        "version": 0.4,
    }
    validate_ui_workflow(workflow)
    return workflow


def _node_position(definition: WorkflowDefinition, node_id: int) -> list[int]:
    if any(node.node_type == "LoraLoader" for node in definition.nodes):
        positions = {
            1: [40, 310], 8: [440, 310], 2: [850, 80], 3: [850, 410],
            4: [850, 760], 5: [1330, 310], 6: [1780, 310], 7: [2160, 310],
        }
    else:
        positions = {
            1: [40, 310], 2: [470, 70], 3: [470, 390], 4: [470, 760],
            5: [930, 310], 6: [1380, 310], 7: [1760, 310],
        }
    return positions[node_id]


def _node_size(node_type: str) -> list[int]:
    return {
        "CheckpointLoaderSimple": [315, 98],
        "CLIPTextEncode": [400, 200],
        "EmptyLatentImage": [315, 106],
        "KSampler": [315, 262],
        "VAEDecode": [210, 58],
        "SaveImage": [315, 270],
        "LoraLoader": [315, 150],
    }[node_type]


def _execution_order(definition: WorkflowDefinition) -> dict[int, int]:
    by_id = {node.node_id: node for node in definition.nodes}
    remaining = {node_id: {value.node_id for value in node.inputs.values() if isinstance(value, Connection)} for node_id, node in by_id.items()}
    ordered = []
    while remaining:
        ready = sorted(node_id for node_id, dependencies in remaining.items() if not dependencies)
        if not ready:
            raise ValueError("WorkflowDefinitionのノード接続に循環があります。")
        for node_id in ready:
            ordered.append(node_id)
            del remaining[node_id]
        for dependencies in remaining.values():
            dependencies.difference_update(ready)
    return {node_id: index for index, node_id in enumerate(ordered)}


def build_workflow(request: WorkflowBuildRequest, family: str = "sdxl") -> dict:
    """Backward-compatible API Prompt builder."""
    return to_api_prompt(build_definition(request, family))


def validate_definition(definition: WorkflowDefinition) -> None:
    ids = {node.node_id for node in definition.nodes}
    if len(ids) != len(definition.nodes):
        raise ValueError("WorkflowDefinition内のnode idが重複しています。")
    for node in definition.nodes:
        if node.node_type not in UI_NODE_SPECS:
            raise ValueError(f"未対応のノードです: {node.node_type}")
        for value in node.inputs.values():
            if isinstance(value, Connection) and value.node_id not in ids:
                raise ValueError(f"ノード {node.node_id} が存在しない接続先を参照しています。")


def validate_ui_workflow(workflow: dict) -> None:
    nodes = workflow.get("nodes")
    links = workflow.get("links")
    if not isinstance(nodes, list) or not isinstance(links, list):
        raise ValueError("ComfyUI Workflow JSONのnodesまたはlinksが正しくありません。")
    node_ids = {node.get("id") for node in nodes}
    if len(node_ids) != len(nodes):
        raise ValueError("ComfyUI Workflow JSON内のnode idが重複しています。")
    link_ids = set()
    for link in links:
        if len(link) != 6:
            raise ValueError("ComfyUI Workflow JSONのlink形式が正しくありません。")
        link_id, origin_id, origin_slot, target_id, target_slot, _ = link
        if origin_id not in node_ids or target_id not in node_ids:
            raise ValueError("ComfyUI Workflow JSONが存在しないノードを参照しています。")
        if link_id in link_ids:
            raise ValueError("ComfyUI Workflow JSON内のlink idが重複しています。")
        link_ids.add(link_id)
        origin = next(node for node in nodes if node["id"] == origin_id)
        target = next(node for node in nodes if node["id"] == target_id)
        if origin_slot >= len(origin["outputs"]) or target_slot >= len(target["inputs"]):
            raise ValueError("ComfyUI Workflow JSONのlink slotが範囲外です。")
        if link_id not in (origin["outputs"][origin_slot].get("links") or []):
            raise ValueError("ComfyUI Workflow JSONのoutput link参照が一致しません。")
        if target["inputs"][target_slot].get("link") != link_id:
            raise ValueError("ComfyUI Workflow JSONのinput link参照が一致しません。")


def validate_workflow(workflow: dict) -> None:
    required = {"1", "2", "3", "4", "5", "6", "7"}
    if not isinstance(workflow, dict) or not required.issubset(workflow):
        raise ValueError("ワークフローテンプレートに必須ノードがありません。")
    for node_id, node in workflow.items():
        if not isinstance(node, dict) or "class_type" not in node or not isinstance(node.get("inputs"), dict):
            raise ValueError(f"ノード {node_id} の形式が正しくありません。")
        for value in node["inputs"].values():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                if value[0] not in workflow:
                    raise ValueError(f"ノード {node_id} が存在しない接続先を参照しています。")
