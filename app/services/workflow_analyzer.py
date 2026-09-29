"""Read-only analysis of ComfyUI UI workflows and API prompts."""

from __future__ import annotations

import json
from collections.abc import Mapping

from app.services.model_metadata import classify_asset, classify_compatibility

MAX_NODES = 2000
MAX_LINKS = 10000
MODEL_KIND_LABELS = {
    "checkpoint": "Checkpoint", "lora": "LoRA", "vae": "VAE", "diffusion_model": "Diffusion Model",
    "text_encoder": "Text Encoder", "upscale_model": "Upscale Model", "controlnet": "ControlNet",
    "clip_vision": "CLIP Vision", "ipadapter": "IPAdapter",
}
SOCKET_TYPES = {
    "MODEL", "CLIP", "VAE", "CONDITIONING", "LATENT", "IMAGE", "MASK",
    "CONTROL_NET", "UPSCALE_MODEL", "CLIP_VISION", "STYLE_MODEL", "GLIGEN",
    "IPADAPTER", "STRING", "ANY", "AUDIO", "GUIDER", "SAMPLER", "SIGMAS",
}

NODE_WIDGET_ORDER = {
    "CheckpointLoaderSimple": ("ckpt_name",),
    "CheckpointLoader": ("ckpt_name",),
    "UNETLoader": ("unet_name", "weight_dtype"),
    "DualCLIPLoader": ("clip_name1", "clip_name2", "type", "device"),
    "CLIPLoader": ("clip_name", "type", "device"),
    "VAELoader": ("vae_name",),
    "LoraLoader": ("lora_name", "strength_model", "strength_clip"),
    "LoraLoaderModelOnly": ("lora_name", "strength_model"),
    "ControlNetLoader": ("control_net_name",),
    "ControlNetLoaderAdvanced": ("control_net_name",),
    "CLIPVisionLoader": ("clip_name",),
    "IPAdapterModelLoader": ("ipadapter_file",),
    "UpscaleModelLoader": ("model_name",),
    "KSampler": ("seed", "control_after_generate", "steps", "cfg", "sampler_name", "scheduler", "denoise"),
    "KSamplerAdvanced": ("add_noise", "noise_seed", "control_after_generate", "steps", "cfg", "sampler_name", "scheduler", "start_at_step", "end_at_step", "return_with_leftover_noise"),
    "EmptyLatentImage": ("width", "height", "batch_size"),
    "EmptySD3LatentImage": ("width", "height", "batch_size"),
    "CLIPTextEncode": ("text",),
    "FluxGuidance": ("guidance",),
    "LoadImage": ("image",),
    "SaveImage": ("filename_prefix",),
}

MODEL_FIELDS = {
    "checkpoint": {"ckpt_name", "checkpoint_name", "checkpoint"},
    "lora": {"lora_name", "lora", "lora_file", "lora_path"},
    "vae": {"vae_name"},
    "diffusion_model": {"unet_name", "diffusion_model"},
    "text_encoder": {"clip_name", "clip_name1", "clip_name2", "text_encoder_name", "t5_name"},
    "upscale_model": {"upscale_model_name", "model_name"},
    "controlnet": {"control_net_name", "controlnet_name"},
    "clip_vision": {"clip_vision_name"},
    "ipadapter": {"ipadapter_file", "ipadapter_name"},
}

SUPPORTED_NODE_TYPES = set(NODE_WIDGET_ORDER) | {
    "LoadImage", "ImageUpscaleWithModel", "VAEEncode", "VAEEncodeForInpaint",
    "SetLatentNoiseMask", "VAEDecode", "VAEDecodeTiled", "LoraLoader",
    "CLIPTextEncodeSDXL", "CLIPTextEncodeFlux", "ConditioningZeroOut",
    "ControlNetApply", "ControlNetApplyAdvanced", "IPAdapter", "IPAdapterAdvanced",
    "SaveImage", "PreviewImage", "EmptyLatentImage", "EmptySD3LatentImage",
    "Note", "PrimitiveNode", "Reroute",
}
UI_ONLY_NODE_TYPES = {"Note", "PrimitiveNode", "Reroute"}


def _normal_name(value: str) -> str:
    return value.replace("\\", "/").strip().casefold()


def _find_inventory_item(inventory: list[dict] | None, kind: str, name: str) -> dict | None:
    if inventory is None:
        return None
    normalized = _normal_name(name)
    same_type = [item for item in inventory if item.get("type") == kind]
    exact = next((item for item in same_type if _normal_name(item.get("comfy_name", item.get("name", ""))) == normalized), None)
    if exact:
        return exact
    basename = normalized.rsplit("/", 1)[-1]
    matches = [item for item in same_type if _normal_name(str(item.get("comfy_name", item.get("name", ""))).rsplit("/", 1)[-1]) == basename]
    return matches[0] if len(matches) == 1 else None


def _inventory_match_status(inventory: list[dict] | None, kind: str, name: str) -> tuple[str, dict | None]:
    if inventory is None:
        return "unknown", None
    match = _find_inventory_item(inventory, kind, name)
    if match:
        return "found", match
    normalized = _normal_name(name)
    basename = normalized.rsplit("/", 1)[-1]
    same_type = [item for item in inventory if item.get("type") == kind]
    basename_matches = [item for item in same_type if _normal_name(str(item.get("comfy_name", item.get("name", ""))).rsplit("/", 1)[-1]) == basename]
    return ("unknown", None) if len(basename_matches) > 1 else ("missing", None)


def _node_widget_names(node_type: str, node: Mapping, node_catalog: Mapping | None) -> list[str]:
    if node_type in NODE_WIDGET_ORDER:
        return list(NODE_WIDGET_ORDER[node_type])

    explicit = []
    inputs = node.get("inputs")
    if isinstance(inputs, list):
        for item in inputs:
            if isinstance(item, Mapping):
                widget = item.get("widget")
                if isinstance(widget, Mapping) and isinstance(widget.get("name"), str):
                    explicit.append(widget["name"])
    if explicit:
        return explicit

    catalog_node = node_catalog.get(node_type) if isinstance(node_catalog, Mapping) else None
    if isinstance(catalog_node, Mapping):
        schema = catalog_node.get("input", {})
        names = []
        if isinstance(schema, Mapping):
            for section in ("required", "optional"):
                values = schema.get(section, {})
                if not isinstance(values, Mapping):
                    continue
                for name, spec in values.items():
                    if isinstance(spec, list) and spec:
                        kind = spec[0]
                        if isinstance(kind, str) and kind.upper() not in SOCKET_TYPES:
                            names.append(name)
        if names:
            return names
    return []


def _ui_widget_values(node: Mapping, node_type: str, node_catalog: Mapping | None) -> dict:
    values = node.get("widgets_values")
    if not isinstance(values, list):
        return {}
    names = _node_widget_names(node_type, node, node_catalog)
    return {name: values[index] for index, name in enumerate(names) if index < len(values)}


def _parse_ui_nodes(workflow: Mapping, node_catalog: Mapping | None) -> tuple[list[dict], list[dict], list[str]]:
    source_nodes = workflow.get("nodes")
    source_links = workflow.get("links")
    warnings = []
    nodes = []
    links = []
    if not isinstance(source_nodes, list):
        warnings.append("nodesが配列ではありません。")
        source_nodes = []
    if len(source_nodes) > MAX_NODES:
        warnings.append(f"ノード数が上限{MAX_NODES}を超えたため、先頭{MAX_NODES}件のみ解析しました。")
        source_nodes = source_nodes[:MAX_NODES]

    for raw in source_nodes:
        if not isinstance(raw, Mapping):
            warnings.append("形式が不正なノードを1件スキップしました。")
            continue
        node_id = raw.get("id")
        node_type = raw.get("type")
        if not isinstance(node_type, str) or not node_type:
            warnings.append(f"id={node_id!r}のノードにtypeがありません。")
            node_type = "UnknownNode"
        inputs = raw.get("inputs") if isinstance(raw.get("inputs"), list) else []
        outputs = raw.get("outputs") if isinstance(raw.get("outputs"), list) else []
        nodes.append({
            "id": str(node_id) if node_id is not None else "?",
            "type": node_type,
            "title": raw.get("title") if isinstance(raw.get("title"), str) else None,
            "widgets": _ui_widget_values(raw, node_type, node_catalog),
            "widget_values": raw.get("widgets_values") if isinstance(raw.get("widgets_values"), list) else [],
            "inputs": [item for item in inputs if isinstance(item, Mapping)],
            "outputs": [item for item in outputs if isinstance(item, Mapping)],
            "properties": raw.get("properties") if isinstance(raw.get("properties"), Mapping) else {},
        })

    if not isinstance(source_links, list):
        warnings.append("linksが配列ではありません。")
        source_links = []
    if len(source_links) > MAX_LINKS:
        warnings.append(f"リンク数が上限{MAX_LINKS}を超えたため、先頭{MAX_LINKS}件のみ解析しました。")
        source_links = source_links[:MAX_LINKS]
    for raw in source_links:
        if isinstance(raw, list) and len(raw) == 6:
            link_id, origin_id, origin_slot, target_id, target_slot, link_type = raw[:6]
        elif isinstance(raw, Mapping):
            link_id, origin_id, origin_slot = raw.get("id"), raw.get("origin_id"), raw.get("origin_slot")
            target_id, target_slot, link_type = raw.get("target_id"), raw.get("target_slot"), raw.get("type")
        else:
            warnings.append("形式が不正なlinkを1件スキップしました。")
            continue
        links.append({"id": link_id, "origin_id": str(origin_id), "origin_slot": origin_slot,
                      "target_id": str(target_id), "target_slot": target_slot, "type": link_type})
    return nodes, links, warnings


def _parse_api_nodes(workflow: Mapping, node_catalog: Mapping | None) -> tuple[list[dict], list[dict], list[str]]:
    warnings = []
    nodes = []
    links = []
    truncated = False
    links_truncated = False
    for node_id, raw in workflow.items():
        if len(nodes) >= MAX_NODES:
            truncated = True
            break
        if not isinstance(raw, Mapping) or not isinstance(raw.get("class_type"), str) or not isinstance(raw.get("inputs"), Mapping):
            warnings.append(f"API Promptのnode {node_id!r}はclass_typeまたはinputsが不正です。")
            continue
        node_type = raw["class_type"]
        inputs = dict(raw["inputs"])
        widgets = {key: value for key, value in inputs.items() if not _is_connection(value)}
        displayed_inputs = [
            {"name": name, "connection": [str(value[0]), value[1]]}
            if _is_connection(value) else {"name": name, "value": value}
            for name, value in inputs.items()
        ]
        schema = node_catalog.get(node_type, {}) if isinstance(node_catalog, Mapping) else {}
        output_names = schema.get("output_name", []) if isinstance(schema, Mapping) else []
        output_types = schema.get("output", []) if isinstance(schema, Mapping) else []
        displayed_outputs = []
        if isinstance(output_names, list) and isinstance(output_types, list):
            for index, output_type in enumerate(output_types):
                displayed_outputs.append({
                    "name": output_names[index] if index < len(output_names) else f"output_{index}",
                    "type": output_type,
                    "slot_index": index,
                    "links": [],
                })
        nodes.append({"id": str(node_id), "type": node_type, "title": None, "widgets": widgets,
                      "widget_values": [], "inputs": displayed_inputs, "outputs": displayed_outputs, "properties": {}})
        for input_name, value in inputs.items():
            if _is_connection(value):
                if len(links) < MAX_LINKS:
                    links.append({"id": f"{node_id}:{input_name}", "origin_id": str(value[0]),
                                  "origin_slot": value[1], "target_id": str(node_id),
                                  "target_slot": input_name, "type": None})
                else:
                    links_truncated = True
    node_by_id = {node["id"]: node for node in nodes}
    for link in links:
        origin = node_by_id.get(link["origin_id"])
        slot = link["origin_slot"]
        if origin and isinstance(slot, int) and 0 <= slot < len(origin["outputs"]):
            origin["outputs"][slot]["links"].append(link["id"])
    if truncated:
        warnings.append(f"ノード数が上限{MAX_NODES}を超えたため、先頭{MAX_NODES}件のみ解析しました。")
    if links_truncated:
        warnings.append(f"リンク数が上限{MAX_LINKS}を超えたため、先頭{MAX_LINKS}件のみ解析しました。")
    return nodes, links, warnings


def _is_connection(value) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[0], (str, int)) and isinstance(value[1], int)


def _model_kind(node_type: str, field: str) -> str | None:
    lowered_type = node_type.casefold()
    for kind, fields in MODEL_FIELDS.items():
        if field in fields:
            if kind == "text_encoder" and "vision" in lowered_type:
                return "clip_vision"
            if kind == "upscale_model" and "ipadapter" in lowered_type:
                return "ipadapter"
            if kind == "upscale_model" and "controlnet" in lowered_type:
                return "controlnet"
            return kind
    if "checkpoint" in lowered_type and field in {"model_name", "ckpt_name"}:
        return "checkpoint"
    if "lora" in lowered_type and field in {"model_name", "name"}:
        return "lora"
    if "vae" in lowered_type and field == "model_name":
        return "vae"
    if "upscale" in lowered_type and field == "model_name":
        return "upscale_model"
    if "controlnet" in lowered_type and field == "model_name":
        return "controlnet"
    if "clipvision" in lowered_type and field == "model_name":
        return "clip_vision"
    if "ipadapter" in lowered_type and field == "model_name":
        return "ipadapter"
    if ("unet" in lowered_type or "diffusionmodel" in lowered_type) and field == "model_name":
        return "diffusion_model"
    return None


def _extract_models(nodes: list[dict], inventory: list[dict] | None) -> list[dict]:
    refs = []
    for node in nodes:
        node_ref_start = len(refs)
        for field, value in node["widgets"].items():
            kind = _model_kind(node["type"], field)
            if kind and isinstance(value, str) and value.strip():
                refs.append({"node_id": node["id"], "node_type": node["type"], "field": field,
                             "type": kind, "name": value.strip()})
        if len(refs) == node_ref_start:
            properties_models = node.get("properties", {}).get("models", [])
            if isinstance(properties_models, list):
                directory_types = {
                    "checkpoints": "checkpoint", "loras": "lora", "vae": "vae",
                    "diffusion_models": "diffusion_model", "unet": "diffusion_model",
                    "text_encoders": "text_encoder", "clip": "text_encoder",
                    "upscale_models": "upscale_model", "controlnet": "controlnet",
                    "clip_vision": "clip_vision", "ipadapter": "ipadapter",
                }
                for asset in properties_models:
                    if not isinstance(asset, Mapping) or not isinstance(asset.get("name"), str):
                        continue
                    kind = directory_types.get(str(asset.get("directory", "")).casefold())
                    if kind:
                        refs.append({"node_id": node["id"], "node_type": node["type"], "field": "properties.models",
                                     "type": kind, "name": asset["name"].strip()})
    seen = set()
    results = []
    for ref in refs:
        key = (ref["type"], _normal_name(ref["name"]))
        if key in seen:
            continue
        seen.add(key)
        status, match = _inventory_match_status(inventory, ref["type"], ref["name"])
        classification = (match.get("classification") or {
            "family": match.get("family", "unknown"), "confidence": "unknown", "source": "inventory",
        }) if match else classify_asset(ref["name"])
        family = match.get("family", "unknown") if match else classification.get("family", "unknown")
        results.append({**ref, "status": status, "inventory_name": match.get("comfy_name", match.get("name")) if match else None,
                        "family": family, "classification": classification})
    return results


def _validate_links(nodes: list[dict], links: list[dict], workflow_type: str) -> list[str]:
    warnings = []
    node_by_id = {node["id"]: node for node in nodes}
    seen_link_ids = set()
    all_link_ids = {str(link["id"]) for link in links}
    if workflow_type == "comfyui_ui_workflow":
        for node in nodes:
            for socket in node["inputs"]:
                link_id = socket.get("link")
                if link_id is not None and str(link_id) not in all_link_ids:
                    warnings.append(f"ノード{node['id']}の入力が存在しないlink {link_id}を参照しています。")
            for socket in node["outputs"]:
                output_refs = socket.get("links")
                if output_refs is None:
                    continue
                if not isinstance(output_refs, list):
                    warnings.append(f"ノード{node['id']}の出力links形式が正しくありません。")
                    continue
                for link_id in output_refs:
                    if str(link_id) not in all_link_ids:
                        warnings.append(f"ノード{node['id']}の出力が存在しないlink {link_id}を参照しています。")
    for link in links:
        link_id_key = str(link["id"])
        if link_id_key in seen_link_ids:
            warnings.append(f"重複したlink idがあります: {link_id_key}")
        seen_link_ids.add(link_id_key)
        if link["origin_id"] not in node_by_id or link["target_id"] not in node_by_id:
            warnings.append(f"link {link['id']}は存在しないノードを参照しています。")
            continue
        if workflow_type == "comfyui_ui_workflow":
            origin_outputs = node_by_id[link["origin_id"]]["outputs"]
            target_inputs = node_by_id[link["target_id"]]["inputs"]
            if not isinstance(link["origin_slot"], int) or not 0 <= link["origin_slot"] < len(origin_outputs):
                warnings.append(f"link {link['id']}の出力slotが範囲外です。")
            else:
                output = origin_outputs[link["origin_slot"]]
                output_links = output.get("links")
                if not isinstance(output_links, list):
                    output_links = []
                if link["id"] not in output_links:
                    warnings.append(f"link {link['id']}が出力ノード側のlinksにありません。")
                if link["type"] is not None and output.get("type") != link["type"]:
                    warnings.append(f"link {link['id']}の出力socket型が一致しません。")
            if not isinstance(link["target_slot"], int) or not 0 <= link["target_slot"] < len(target_inputs):
                warnings.append(f"link {link['id']}の入力slotが範囲外です。")
            else:
                input_socket = target_inputs[link["target_slot"]]
                if input_socket.get("link") != link["id"]:
                    warnings.append(f"link {link['id']}が入力ノード側のlink参照と一致しません。")
                if link["type"] is not None and input_socket.get("type") != link["type"]:
                    warnings.append(f"link {link['id']}の入力socket型が一致しません。")
        elif not isinstance(link["origin_slot"], int) or link["origin_slot"] < 0:
            warnings.append(f"link {link['id']}の出力slotが正しくありません。")
    if warnings:
        warnings.append("Workflowのlink整合性を確認してください。解析可能なノード情報は表示しています。")
    return warnings


def _node_availability(nodes: list[dict], node_catalog: Mapping | None) -> tuple[list[dict], list[dict], list[dict]]:
    custom_nodes, missing_nodes, unparsed_nodes = [], [], []
    for node in nodes:
        node_type = node["type"]
        catalog_item = node_catalog.get(node_type) if isinstance(node_catalog, Mapping) else None
        if node_type in UI_ONLY_NODE_TYPES:
            availability, module = "available", None
        elif node_catalog is None:
            availability, module = "unknown", None
        elif isinstance(catalog_item, Mapping):
            availability = "available"
            module = catalog_item.get("python_module")
        else:
            availability, module = "missing", None
        node["availability"] = availability
        explicit_custom = isinstance(module, str) and module.startswith("custom_nodes.")
        registry_id = str(node.get("properties", {}).get("cnr_id", "")).casefold()
        has_custom_registry_id = bool(registry_id and registry_id != "comfy-core")
        node["custom_node"] = True if explicit_custom or (availability == "missing" and has_custom_registry_id) else (False if availability == "available" else None)
        if availability == "missing":
            missing_nodes.append({"type": node_type, "node_id": node["id"], "status": availability})
        if node["custom_node"] is True:
            custom_nodes.append({"type": node_type, "node_id": node["id"], "status": availability,
                                 "module": module})
        if node_type not in SUPPORTED_NODE_TYPES:
            unparsed_nodes.append({"type": node_type, "node_id": node["id"], "status": availability})
    return custom_nodes, missing_nodes, unparsed_nodes


def _parameters(nodes: list[dict]) -> dict:
    sampler_nodes = [node for node in nodes if node["type"] in {"KSampler", "KSamplerAdvanced"}]
    latent_nodes = [node for node in nodes if node["type"] in {"EmptyLatentImage", "EmptySD3LatentImage"}]
    lora_nodes = [node for node in nodes if "lora" in node["type"].casefold()]
    result = {"samplers": [], "latents": [], "loras": [], "prompts": []}
    for node in sampler_nodes:
        values = node["widgets"]
        sampler = {"node_id": node["id"], **{key: values[key] for key in (
            "seed", "steps", "cfg", "sampler_name", "scheduler", "denoise", "noise_seed",
        ) if key in values}}
        if "seed" not in sampler and "noise_seed" in sampler:
            sampler["seed"] = sampler["noise_seed"]
        result["samplers"].append(sampler)
    for node in latent_nodes:
        values = node["widgets"]
        result["latents"].append({"node_id": node["id"], **{key: values[key] for key in ("width", "height", "batch_size") if key in values}})
    for node in lora_nodes:
        values = node["widgets"]
        entry = {"node_id": node["id"]}
        for key in ("strength_model", "strength_clip", "strength"): 
            if key in values:
                entry[key] = values[key]
        result["loras"].append(entry)
    for node in nodes:
        if node["type"].startswith("CLIPTextEncode") and isinstance(node["widgets"].get("text"), str):
            result["prompts"].append({"node_id": node["id"], "node_type": node["type"], "text": node["widgets"]["text"]})
    return result


def _infer_workflow_type(nodes: list[dict]) -> str:
    types = {node["type"].casefold() for node in nodes}
    upscale = any("upscale" in value for value in types)
    inpaint = any("inpaint" in value or "noisemask" in value for value in types)
    img2img = any(value.startswith("vaeencode") for value in types) and not inpaint
    txt2img = "emptylatentimage" in types or "emptysd3latentimage" in types
    controlnet = any("controlnet" in value or "t2iadapter" in value for value in types)
    operations = [name for name, enabled in (("upscale", upscale), ("inpaint", inpaint), ("img2img", img2img), ("txt2img", txt2img)) if enabled]
    if len(operations) > 1:
        return "mixed"
    if controlnet:
        return "controlnet"
    if not operations:
        return "unknown"
    return operations[0]


def _topological_flow(nodes: list[dict], links: list[dict]) -> list[dict]:
    ids = [node["id"] for node in nodes]
    deps = {node_id: set() for node_id in ids}
    for link in links:
        if link["origin_id"] in deps and link["target_id"] in deps:
            deps[link["target_id"]].add(link["origin_id"])
    ordered = []
    remaining = {key: set(value) for key, value in deps.items()}
    while remaining:
        ready = [key for key in ids if key in remaining and not remaining[key]]
        if not ready:
            ordered.extend(key for key in ids if key in remaining)
            break
        ordered.extend(ready)
        for key in ready:
            remaining.pop(key)
        for requirements in remaining.values():
            requirements.difference_update(ready)
    lookup = {node["id"]: node for node in nodes}
    return [{"node_id": key, "type": lookup[key]["type"], "title": lookup[key]["title"],
             "from": sorted(deps.get(key, set()))} for key in ordered]


def _summary(nodes: list[dict], models: list[dict], parameters: dict, workflow_type: str) -> dict:
    checkpoint = next((item for item in models if item["type"] == "checkpoint"), None)
    base_model = checkpoint or next((item for item in models if item["type"] == "diffusion_model"), None)
    family = base_model.get("family") if base_model else None
    family_label = {"sdxl": "SDXL", "sd15": "SD1.5", "flux": "Flux"}.get(family, "モデル系統不明")
    if base_model and (base_model.get("classification") or {}).get("confidence") == "low":
        family_label += "候補"
    kind_label = {"txt2img": "txt2img", "img2img": "img2img", "upscale": "Upscale",
                  "inpaint": "inpaint", "controlnet": "ControlNet", "mixed": "複合"}.get(workflow_type, "用途不明")
    if workflow_type == "upscale":
        title = "Upscale Workflowです。"
    elif base_model:
        title = f"{family_label}の{kind_label} Workflowです。"
    else:
        title = f"{kind_label} Workflowです。"
    steps = []
    types = {node["type"] for node in nodes}
    if any(model["type"] == "checkpoint" for model in models):
        steps.append("Checkpointを読み込み")
    elif any(model["type"] == "diffusion_model" for model in models):
        steps.append("Diffusion ModelとText Encoderを読み込み")
    if any(model["type"] == "lora" for model in models):
        steps.append(f"LoRAを{sum(model['type'] == 'lora' for model in models)}件適用")
    encode_count = sum(node["type"].startswith("CLIPTextEncode") for node in nodes)
    if encode_count:
        steps.append(f"PromptをEncode ({encode_count}ノード)")
    if "EmptyLatentImage" in types or "EmptySD3LatentImage" in types:
        steps.append("空のLatentを作成")
    if any(node_type.startswith("VAEEncode") for node_type in types):
        steps.append("入力画像をLatentへEncode")
    if workflow_type == "controlnet":
        steps.append("ControlNet系ノードで条件を追加")
    if "LoadImage" in types:
        steps.append("入力画像を読み込み")
    if any("Upscale" in node_type for node_type in types):
        steps.append("Upscale Modelで拡大")
    if parameters["samplers"]:
        sampler = parameters["samplers"][0]
        detail = f"KSamplerで{sampler.get('steps')} steps生成" if "steps" in sampler else "KSamplerで生成"
        steps.append(detail)
    if "VAEDecode" in types or "VAEDecodeTiled" in types:
        steps.append("VAE Decode")
    if "SaveImage" in types:
        steps.append("画像を保存")
    if not steps:
        steps.append("解析可能な処理ノードを確認")
    return {"title": title, "workflow_type": workflow_type, "family": family or "unknown", "steps": steps,
            "lora_count": sum(model["type"] == "lora" for model in models)}


def _vram_warnings(nodes: list[dict], parameters: dict, models: list[dict], vram_gb: float | None) -> list[str]:
    warnings = []
    max_width = max((item.get("width", 0) for item in parameters["latents"] if isinstance(item.get("width"), int)), default=0)
    max_height = max((item.get("height", 0) for item in parameters["latents"] if isinstance(item.get("height"), int)), default=0)
    if max_width * max_height >= 4096 * 4096 or max(max_width, max_height) >= 4096:
        warnings.append("非常に大きい解像度です。RTX 3060 12GB環境ではVRAM負荷が高くなる可能性があります。")
    elif max_width * max_height >= 2048 * 2048:
        warnings.append("高解像度設定です。RTX 3060 12GB環境ではVRAM負荷が高くなる可能性があります。")
    control_nodes = [node["type"].casefold() for node in nodes if "controlnet" in node["type"].casefold() or "t2iadapter" in node["type"].casefold()]
    apply_count = sum("apply" in name or "t2iadapter" in name for name in control_nodes)
    loader_count = sum("loader" in name for name in control_nodes)
    control_count = apply_count or loader_count
    if control_count >= 2:
        warnings.append("ControlNet系ノードを複数使用しています。VRAM負荷が高くなる可能性があります。")
    elif control_count == 1 and max_width * max_height >= 2048 * 2048:
        warnings.append("高解像度とControlNet系ノードを併用しています。VRAM負荷が高くなる可能性があります。")
    if any(item["type"] == "diffusion_model" and ("flux" in item.get("family", "") or "flux" in item["name"].casefold()) for item in models):
        warnings.append("Flux系Diffusion Modelを使用しています。モデル設定や解像度によってVRAM負荷が高くなる可能性があります。")
    if vram_gb is not None and vram_gb <= 12 and len(nodes) >= 30:
        warnings.append("ノード数が多いため、RTX 3060 12GB環境ではVRAM負荷に注意してください。")
    return warnings


def analyze_workflow_json(raw: bytes | str | Mapping, *, inventory: list[dict] | None = None,
                          node_catalog: Mapping | None = None, vram_gb: float | None = 12.0) -> dict:
    """Analyze JSON content without executing or mutating any part of it."""
    if isinstance(raw, bytes):
        try:
            workflow = json.loads(raw.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("JSONを読み込めません。文字コードまたはJSON構文を確認してください。") from exc
    elif isinstance(raw, str):
        try:
            workflow = json.loads(raw)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("JSONを読み込めません。JSON構文を確認してください。") from exc
    else:
        workflow = raw
    if not isinstance(workflow, Mapping):
        raise ValueError("Workflow JSONのルートはオブジェクトである必要があります。")

    if isinstance(workflow.get("nodes"), list) or "links" in workflow:
        workflow_type = "comfyui_ui_workflow"
        nodes, links, warnings = _parse_ui_nodes(workflow, node_catalog)
    elif workflow and any(isinstance(value, Mapping) and "class_type" in value for value in workflow.values()):
        workflow_type = "comfyui_api_prompt"
        nodes, links, warnings = _parse_api_nodes(workflow, node_catalog)
    else:
        return {"valid": False, "workflow_type": "unknown", "analysis_only": True, "node_count": 0,
                "link_count": 0, "nodes": [], "links": [], "models": [], "loras": [],
                "missing_models": [], "unknown_models": [], "missing_nodes": [], "custom_nodes": [],
                "unknown_nodes": [], "warnings": ["ComfyUI UI WorkflowまたはAPI Promptの構造を判定できませんでした。"],
                "summary": {"title": "不明なJSON形式です。", "workflow_type": "unknown", "steps": []},
                "parameters": {}, "flow": [], "vram_warnings": []}

    if not nodes:
        warnings.append("ノードがありません。Workflow内容を確認してください。")
    if len({node["id"] for node in nodes}) != len(nodes):
        warnings.append("重複したnode idがあります。")
    warnings.extend(_validate_links(nodes, links, workflow_type))
    custom_nodes, missing_nodes, unknown_nodes = _node_availability(nodes, node_catalog)
    models = _extract_models(nodes, inventory)
    loras = [model for model in models if model["type"] == "lora"]
    checkpoints = [model for model in models if model["type"] == "checkpoint"]
    compatibility = []
    for checkpoint in checkpoints:
        checkpoint_item = _find_inventory_item(inventory, "checkpoint", checkpoint["name"])
        for lora in loras:
            lora_item = _find_inventory_item(inventory, "lora", lora["name"])
            if checkpoint_item and lora_item:
                compatibility.append({"checkpoint": checkpoint["name"], "lora": lora["name"],
                                      **classify_compatibility(checkpoint_item, lora_item)})
            else:
                compatibility.append({"checkpoint": checkpoint["name"], "lora": lora["name"],
                                      "status": "unknown", "message": "CheckpointまたはLoRAが親機在庫にないため互換性を照合できません。"})
    parameters = _parameters(nodes)
    workflow_kind = _infer_workflow_type(nodes)
    summary = _summary(nodes, models, parameters, workflow_kind)
    vram_warnings = _vram_warnings(nodes, parameters, models, vram_gb)
    for model in models:
        if model["status"] == "missing":
            warnings.append(f"親機ComfyUIに{MODEL_KIND_LABELS.get(model['type'], model['type'])}がありません: {model['name']}")
        elif model["status"] == "unknown":
            warnings.append(f"親機ComfyUIの在庫と照合できません: {model['name']}")
    for node in missing_nodes:
        warnings.append(f"親機ComfyUIにノード型がありません: {node['type']}")
    if inventory is None:
        warnings.append("親機ComfyUIのモデル在庫を取得できず、モデル照合は不明です。")
    if node_catalog is None:
        warnings.append("親機ComfyUIのobject_infoを取得できず、ノード照合は不明です。")
    if any(item["status"] == "incompatible" for item in compatibility):
        warnings.append("CheckpointとLoRAに明確な系統不一致があります。")
    elif any(item["status"] in {"compatible_with_warning", "unknown"} for item in compatibility):
        warnings.append("CheckpointとLoRAの互換性に確認が必要です。")
    warnings.extend(vram_warnings)

    return {
        "valid": bool(nodes), "workflow_type": workflow_type, "generation_type": workflow_kind,
        "analysis_only": True, "node_count": len(nodes), "link_count": len(links),
        "nodes": nodes, "links": links, "models": models, "loras": loras,
        "missing_models": [item for item in models if item["status"] == "missing"],
        "unknown_models": [item for item in models if item["status"] == "unknown"],
        "missing_nodes": missing_nodes, "custom_nodes": custom_nodes, "unknown_nodes": unknown_nodes,
        "compatibility": compatibility, "warnings": list(dict.fromkeys(warnings)),
        "summary": summary, "parameters": parameters,
        "flow": _topological_flow(nodes, links), "vram_warnings": vram_warnings,
        "inventory_source": "comfy_api" if inventory is not None else None,
        "node_catalog_source": "object_info" if node_catalog is not None else None,
    }
