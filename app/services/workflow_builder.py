from dataclasses import dataclass
import json
from pathlib import Path
import secrets

from app.schemas.workflow import WorkflowBuildRequest
from app.services.model_profile_service import ModelProfile, ModelProfileService

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
    generation_type: str = "txt2img"
    input_image: str | None = None
    denoise: float | None = None


@dataclass(frozen=True)
class UiNodeSpec:
    input_sockets: tuple[tuple[str, str], ...]
    output_sockets: tuple[tuple[str, str], ...]
    widget_order: tuple[str, ...] = ()
    has_output: bool = True
    widget_values_suffix: tuple[object, ...] = ()


UI_NODE_SPECS = {
    "CheckpointLoaderSimple": UiNodeSpec((), (("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")), ("ckpt_name",)),
    "UNETLoader": UiNodeSpec((), (("MODEL", "MODEL"),), ("unet_name", "weight_dtype")),
    "DualCLIPLoader": UiNodeSpec((), (("CLIP", "CLIP"),), ("clip_name1", "clip_name2", "type", "device")),
    "VAELoader": UiNodeSpec((), (("VAE", "VAE"),), ("vae_name",)),
    "LoadImage": UiNodeSpec((), (("IMAGE", "IMAGE"), ("MASK", "MASK")), ("image",), widget_values_suffix=("image",)),
    "VAEEncode": UiNodeSpec((("pixels", "IMAGE"), ("vae", "VAE")), (("LATENT", "LATENT"),)),
    "CLIPTextEncode": UiNodeSpec((("clip", "CLIP"),), (("CONDITIONING", "CONDITIONING"),), ("text",)),
    "FluxGuidance": UiNodeSpec((("conditioning", "CONDITIONING"),), (("CONDITIONING", "CONDITIONING"),), ("guidance",)),
    "EmptySD3LatentImage": UiNodeSpec((), (("LATENT", "LATENT"),), ("width", "height", "batch_size")),
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


def build_definition(request: WorkflowBuildRequest, family: str = "sdxl", input_image: str | None = None) -> WorkflowDefinition:
    try:
        profile = ModelProfileService().get_profile(family)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"モデルProfileが見つからないか不正です: {family}") from exc
    if not profile.enabled or request.generation_type not in profile.supported_generation_types:
        raise ValueError(f"{profile.name}の{request.generation_type}は現在対応していません。")
    if request.lora and not profile.capabilities.lora:
        raise ValueError(f"{profile.name}はLoRAに対応していません。")

    if request.generation_type == "img2img":
        if profile.architecture != "checkpoint":
            raise ValueError(f"{profile.name}のimg2imgは現在対応していません。")
        if not input_image:
            raise ValueError("img2imgには入力画像が必要です。画像をアップロードしてください。")
        definition = _build_img2img_definition(request, profile, input_image)
        validate_definition(definition)
        return definition
    if input_image:
        raise ValueError("txt2imgでは入力画像を指定できません。")

    settings = resolve_generation_settings(request, profile)

    if profile.architecture == "flux_split":
        if settings["width"] % profile.resolution_multiple or settings["height"] % profile.resolution_multiple:
            raise ValueError(f"Fluxの幅と高さは{profile.resolution_multiple}の倍数にしてください。")
        definition = _build_flux_definition(request, profile, settings)
        validate_definition(definition)
        return definition

    nodes = [
        WorkflowNodeDefinition(1, "CheckpointLoaderSimple", {"ckpt_name": request.model}),
        WorkflowNodeDefinition(2, "CLIPTextEncode", {"text": request.prompt, "clip": Connection(1, 1)}),
        WorkflowNodeDefinition(3, "CLIPTextEncode", {"text": request.negative_prompt, "clip": Connection(1, 1)}),
        WorkflowNodeDefinition(4, "EmptyLatentImage", {"width": settings["width"], "height": settings["height"], "batch_size": 1}),
        WorkflowNodeDefinition(5, "KSampler", {
            "seed": request.seed if request.seed is not None else secrets.randbelow(2**32),
            "seed_control": "fixed",
            "steps": settings["steps"],
            "cfg": settings["cfg"],
            "sampler_name": settings["sampler"],
            "scheduler": settings["scheduler"],
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

    definition = WorkflowDefinition(family=family, nodes=tuple(nodes), generation_type="txt2img")
    validate_definition(definition)
    return definition


def _build_img2img_definition(request: WorkflowBuildRequest, profile: ModelProfile, input_image: str) -> WorkflowDefinition:
    nodes = [
        WorkflowNodeDefinition(1, "CheckpointLoaderSimple", {"ckpt_name": request.model}),
        WorkflowNodeDefinition(2, "CLIPTextEncode", {"text": request.prompt, "clip": Connection(1, 1)}),
        WorkflowNodeDefinition(3, "CLIPTextEncode", {"text": request.negative_prompt, "clip": Connection(1, 1)}),
        WorkflowNodeDefinition(4, "LoadImage", {"image": input_image}),
        WorkflowNodeDefinition(5, "VAEEncode", {"pixels": Connection(4, 0), "vae": Connection(1, 2)}),
        WorkflowNodeDefinition(6, "KSampler", {
            "seed": request.seed if request.seed is not None else secrets.randbelow(2**32),
            "seed_control": "fixed",
            "steps": request.steps if request.steps is not None else profile.default_steps,
            "cfg": request.cfg if request.cfg is not None else profile.default_cfg,
            "sampler_name": request.sampler or profile.default_sampler,
            "scheduler": request.scheduler or profile.default_scheduler,
            "denoise": request.denoise,
            "model": Connection(1, 0),
            "positive": Connection(2, 0),
            "negative": Connection(3, 0),
            "latent_image": Connection(5, 0),
        }),
        WorkflowNodeDefinition(7, "VAEDecode", {"samples": Connection(6, 0), "vae": Connection(1, 2)}),
        WorkflowNodeDefinition(8, "SaveImage", {"images": Connection(7, 0), "filename_prefix": "ComfyWorkflowBuilder"}),
    ]
    if request.lora:
        nodes.append(WorkflowNodeDefinition(9, "LoraLoader", {
            "model": Connection(1, 0),
            "clip": Connection(1, 1),
            "lora_name": request.lora,
            "strength_model": request.lora_weight,
            "strength_clip": request.lora_weight,
        }))
        nodes[1].inputs["clip"] = Connection(9, 1)
        nodes[2].inputs["clip"] = Connection(9, 1)
        nodes[5].inputs["model"] = Connection(9, 0)
    return WorkflowDefinition(
        family=profile.id,
        nodes=tuple(nodes),
        generation_type="img2img",
        input_image=input_image,
        denoise=request.denoise,
    )


def _build_flux_definition(request: WorkflowBuildRequest, profile: ModelProfile, settings: dict) -> WorkflowDefinition:
    diffusion_model = request.diffusion_model or request.model
    guidance = request.guidance if request.guidance is not None else profile.default_guidance
    nodes = (
        WorkflowNodeDefinition(1, "UNETLoader", {
            "unet_name": diffusion_model or "",
            "weight_dtype": "default",
        }),
        WorkflowNodeDefinition(2, "DualCLIPLoader", {
            "clip_name1": request.clip_name1 or "",
            "clip_name2": request.clip_name2 or "",
            "type": "flux",
            "device": "default",
        }),
        WorkflowNodeDefinition(3, "VAELoader", {"vae_name": request.vae_model or ""}),
        WorkflowNodeDefinition(4, "CLIPTextEncode", {"text": request.prompt, "clip": Connection(2, 0)}),
        WorkflowNodeDefinition(5, "FluxGuidance", {"conditioning": Connection(4, 0), "guidance": guidance}),
        WorkflowNodeDefinition(6, "EmptySD3LatentImage", {
            "width": settings["width"], "height": settings["height"], "batch_size": 1,
        }),
        WorkflowNodeDefinition(7, "KSampler", {
            "seed": request.seed if request.seed is not None else secrets.randbelow(2**32),
            "seed_control": "fixed",
            "steps": settings["steps"],
            "cfg": settings["cfg"],
            "sampler_name": settings["sampler"],
            "scheduler": settings["scheduler"],
            "denoise": 1.0,
            "model": Connection(1, 0),
            "positive": Connection(5, 0),
            # CFG=1 makes negative conditioning irrelevant; Flux UI omits a negative prompt.
            "negative": Connection(5, 0),
            "latent_image": Connection(6, 0),
        }),
        WorkflowNodeDefinition(8, "VAEDecode", {"samples": Connection(7, 0), "vae": Connection(3, 0)}),
        WorkflowNodeDefinition(9, "SaveImage", {"images": Connection(8, 0), "filename_prefix": "ComfyWorkflowBuilder"}),
    )
    return WorkflowDefinition(family=profile.id, nodes=nodes)


def resolve_generation_settings(request: WorkflowBuildRequest, profile: ModelProfile) -> dict:
    """Fill omitted request values from the selected model profile."""
    return {
        "width": request.width if request.width is not None else profile.default_width,
        "height": request.height if request.height is not None else profile.default_height,
        "steps": request.steps if request.steps is not None else profile.default_steps,
        "cfg": request.cfg if request.cfg is not None else profile.default_cfg,
        "sampler": request.sampler or profile.default_sampler,
        "scheduler": request.scheduler or profile.default_scheduler,
    }


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
    try:
        profile = ModelProfileService().get_profile(definition.family)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"モデルProfileが見つからないか不正です: {definition.family}") from exc
    capability = getattr(profile.capabilities, definition.generation_type, False)
    if not profile.enabled or not capability:
        raise ValueError(f"{profile.name}の{definition.generation_type} ComfyUI Workflow JSONには現在対応していません。")

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
            "widgets_values": [definition_node.inputs[name] for name in spec.widget_order] + list(spec.widget_values_suffix),
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
    if definition.family == "flux":
        positions = {
            1: [40, 300], 2: [40, 570], 3: [470, 820], 4: [470, 570], 5: [900, 570],
            6: [900, 900], 7: [1320, 400], 8: [1760, 400], 9: [2110, 400],
        }
        return positions[node_id]
    if definition.generation_type == "img2img":
        if any(node.node_type == "LoraLoader" for node in definition.nodes):
            positions = {
                1: [30, 300], 9: [410, 300], 2: [810, 60], 3: [810, 370],
                4: [30, 720], 5: [490, 790], 6: [1300, 360], 7: [1730, 360], 8: [2080, 360],
            }
        else:
            positions = {
                1: [30, 300], 2: [470, 60], 3: [470, 370], 4: [30, 720],
                5: [470, 790], 6: [1000, 360], 7: [1430, 360], 8: [1770, 360],
            }
        return positions[node_id]
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
        "UNETLoader": [315, 98],
        "DualCLIPLoader": [315, 150],
        "VAELoader": [315, 58],
        "LoadImage": [315, 350],
        "VAEEncode": [210, 58],
        "CLIPTextEncode": [400, 200],
        "FluxGuidance": [315, 82],
        "EmptySD3LatentImage": [315, 106],
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


def build_workflow(request: WorkflowBuildRequest, family: str = "sdxl", input_image: str | None = None) -> dict:
    """Backward-compatible API Prompt builder."""
    return to_api_prompt(build_definition(request, family, input_image))


def validate_definition(definition: WorkflowDefinition) -> None:
    if definition.generation_type not in {"txt2img", "img2img"}:
        raise ValueError("WorkflowDefinitionのgeneration_typeが正しくありません。")
    if definition.generation_type == "img2img":
        if not definition.input_image or definition.denoise is None or not 0.0 <= definition.denoise <= 1.0:
            raise ValueError("img2imgには入力画像と0.0〜1.0のdenoiseが必要です。")
    elif definition.input_image is not None or definition.denoise is not None:
        raise ValueError("txt2imgにimg2img用の値は指定できません。")
    ids = {node.node_id for node in definition.nodes}
    if len(ids) != len(definition.nodes):
        raise ValueError("WorkflowDefinition内のnode idが重複しています。")
    by_id = {node.node_id: node for node in definition.nodes}
    for node in definition.nodes:
        if node.node_type not in UI_NODE_SPECS:
            raise ValueError(f"未対応のノードです: {node.node_type}")
        spec = UI_NODE_SPECS[node.node_type]
        socket_types = dict(spec.input_sockets)
        allowed_inputs = set(socket_types) | set(spec.widget_order)
        if set(node.inputs) - allowed_inputs:
            raise ValueError(f"ノード {node.node_id} に未定義の入力があります。")
        if set(socket_types) - set(node.inputs):
            raise ValueError(f"ノード {node.node_id} の必須接続が不足しています。")
        for input_name, value in node.inputs.items():
            if not isinstance(value, Connection):
                if input_name not in spec.widget_order:
                    raise ValueError(f"ノード {node.node_id} の入力 {input_name} は接続されていません。")
                continue
            if input_name not in socket_types:
                raise ValueError(f"ノード {node.node_id} のwidget {input_name} にノード接続は指定できません。")
            if value.node_id not in ids:
                raise ValueError(f"ノード {node.node_id} が存在しない接続元を参照しています。")
            origin = by_id[value.node_id]
            origin_spec = UI_NODE_SPECS[origin.node_type]
            if not isinstance(value.output_index, int) or not 0 <= value.output_index < len(origin_spec.output_sockets):
                raise ValueError(f"ノード {node.node_id} が存在しない出力slotを参照しています。")
            _, output_type = origin_spec.output_sockets[value.output_index]
            if output_type != socket_types[input_name]:
                raise ValueError(f"ノード {node.node_id} の接続slot型が一致しません。")


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
        if origin_slot < 0 or target_slot < 0 or origin_slot >= len(origin["outputs"]) or target_slot >= len(target["inputs"]):
            raise ValueError("ComfyUI Workflow JSONのlink slotが範囲外です。")
        if link[5] != origin["outputs"][origin_slot].get("type") or link[5] != target["inputs"][target_slot].get("type"):
            raise ValueError("ComfyUI Workflow JSONのlink socket型が一致しません。")
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
