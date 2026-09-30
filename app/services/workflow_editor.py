"""Safe, semantic edits for a deliberately small set of ComfyUI UI Workflow fields."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from typing import Any

from app.services.model_metadata import classify_asset, classify_compatibility
from app.services.workflow_analyzer import analyze_workflow_json

MAX_WORKFLOW_BYTES = 10 * 1024 * 1024

# These indices describe the known core-node widgets in ComfyUI's serialized UI workflow.
# Unknown node layouts are never guessed or edited.
WIDGET_FIELDS = {
    "CheckpointLoaderSimple": {"checkpoint": ("ckpt_name", 0)},
    "LoraLoader": {"lora": ("lora_name", 0), "strength_model": ("strength_model", 1), "strength_clip": ("strength_clip", 2)},
    "LoraLoaderModelOnly": {"lora": ("lora_name", 0), "strength_model": ("strength_model", 1)},
    "KSampler": {
        "seed": ("seed", 0), "steps": ("steps", 2), "cfg": ("cfg", 3),
        "sampler": ("sampler_name", 4), "scheduler": ("scheduler", 5), "denoise": ("denoise", 6),
    },
    "KSamplerAdvanced": {
        "seed": ("noise_seed", 1), "steps": ("steps", 3), "cfg": ("cfg", 4),
        "sampler": ("sampler_name", 5), "scheduler": ("scheduler", 6),
    },
    "EmptyLatentImage": {"width": ("width", 0), "height": ("height", 1)},
    "EmptySD3LatentImage": {"width": ("width", 0), "height": ("height", 1)},
}

FIELD_LABELS = {
    "checkpoint": "Checkpoint", "lora": "LoRA", "strength_model": "MODEL Weight",
    "strength_clip": "CLIP Weight", "seed": "Seed", "steps": "Steps", "cfg": "CFG",
    "sampler": "Sampler", "scheduler": "Scheduler", "denoise": "Denoise",
    "width": "Width", "height": "Height",
}
NUMBER_FIELDS = {"seed", "steps", "cfg", "denoise", "width", "height", "strength_model", "strength_clip"}
INT_FIELDS = {"seed", "steps", "width", "height"}
WIDGETLESS_UI_NODES = {"Note", "MarkdownNote", "PrimitiveNode", "Reroute"}
API_UI_NODE_TYPES = {
    "CheckpointLoaderSimple", "UNETLoader", "DualCLIPLoader", "VAELoader", "LoraLoader", "LoraLoaderModelOnly",
    "CLIPTextEncode", "FluxGuidance", "EmptyLatentImage", "EmptySD3LatentImage", "KSampler", "KSamplerAdvanced",
    "VAEDecode", "SaveImage", "PreviewImage", "LoadImage", "VAEEncode", "UpscaleModelLoader", "ImageUpscaleWithModel",
}
API_OMIT_WIDGETS = {"control_after_generate"}


def _parse_workflow(raw: str | bytes | Mapping) -> dict:
    if isinstance(raw, bytes):
        if len(raw) > MAX_WORKFLOW_BYTES:
            raise ValueError("Workflow JSONは10MB以下にしてください。")
        raw = raw.decode("utf-8-sig")
    if isinstance(raw, str):
        if len(raw.encode("utf-8")) > MAX_WORKFLOW_BYTES:
            raise ValueError("Workflow JSONは10MB以下にしてください。")
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("Workflow JSONを解析できません。") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("Workflow JSONのルートはオブジェクトである必要があります。")
    if not isinstance(raw.get("nodes"), list) or not isinstance(raw.get("links"), list):
        raise ValueError("編集にはComfyUI UI Workflow JSONが必要です。API Prompt JSONは解析のみ対応です。")
    if len(raw.get("nodes", [])) > 2000 or len(raw.get("links", [])) > 10000:
        raise ValueError("Workflowのノードまたはリンク数が編集上限を超えています。")
    return copy.deepcopy(dict(raw))


def _id(node: Mapping) -> str:
    return str(node.get("id"))


def _link_record(raw: Any) -> dict | None:
    if isinstance(raw, list) and len(raw) == 6:
        return {"id": raw[0], "origin_id": str(raw[1]), "origin_slot": raw[2], "target_id": str(raw[3]), "target_slot": raw[4], "type": raw[5]}
    if isinstance(raw, Mapping):
        return {"id": raw.get("id"), "origin_id": str(raw.get("origin_id")), "origin_slot": raw.get("origin_slot"),
                "target_id": str(raw.get("target_id")), "target_slot": raw.get("target_slot"), "type": raw.get("type")}
    return None


def _link_records(workflow: Mapping) -> list[dict]:
    return [record for raw in workflow.get("links", []) if (record := _link_record(raw)) is not None]


def _nodes_by_id(workflow: Mapping) -> dict[str, dict]:
    nodes = workflow.get("nodes", [])
    return {_id(node): node for node in nodes if isinstance(node, dict) and "id" in node}


def _widget_index(node: Mapping, semantic: str) -> tuple[str, int] | None:
    spec = WIDGET_FIELDS.get(node.get("type"), {}).get(semantic)
    values = node.get("widgets_values")
    if not spec or not isinstance(values, list) or spec[1] >= len(values):
        return None
    return spec


def _widget_binding(workflow: Mapping, node: dict, semantic: str) -> tuple[dict, int] | None:
    """Resolve a widget field, following a connected PrimitiveNode when safe."""
    spec = _widget_index(node, semantic)
    if not spec:
        return None
    widget_name, index = spec
    node_id = _id(node)
    by_id = _nodes_by_id(workflow)
    link_by_id = {str(item["id"]): item for item in _link_records(workflow)}
    for input_item in node.get("inputs", []) if isinstance(node.get("inputs"), list) else []:
        if not isinstance(input_item, Mapping):
            continue
        input_name = input_item.get("name")
        widget = input_item.get("widget")
        bound_name = widget.get("name") if isinstance(widget, Mapping) else input_name
        if bound_name != widget_name or input_item.get("link") is None:
            continue
        link = link_by_id.get(str(input_item.get("link")))
        origin = by_id.get(link["origin_id"]) if link else None
        if origin and origin.get("type") == "PrimitiveNode":
            values = origin.get("widgets_values")
            if isinstance(values, list) and values:
                output = next((item for item in origin.get("outputs", []) if isinstance(item, Mapping)
                               and item.get("slot_index", 0) == (link or {}).get("origin_slot")), None)
                output_widget = output.get("widget") if isinstance(output, Mapping) else None
                source_name = output_widget.get("name") if isinstance(output_widget, Mapping) else None
                if source_name and source_name != widget_name:
                    return None
                return origin, 0
        # A different node supplies the value; changing the hidden widget would be ineffective.
        if origin:
            return None
    return node, index


def _resolve_value(workflow: Mapping, node: dict, semantic: str):
    binding = _widget_binding(workflow, node, semantic)
    return binding[0]["widgets_values"][binding[1]] if binding else None


def _uses_image_latent(workflow: Mapping, node: Mapping) -> bool:
    by_id = _nodes_by_id(workflow)
    links = {str(item["id"]): item for item in _link_records(workflow)}
    for input_item in node.get("inputs", []) if isinstance(node.get("inputs"), list) else []:
        if not isinstance(input_item, Mapping) or input_item.get("name") != "latent_image" or input_item.get("link") is None:
            continue
        link = links.get(str(input_item["link"]))
        source = by_id.get(link["origin_id"]) if link else None
        return bool(source and source.get("type") in {"VAEEncode", "VAEEncodeForInpaint"})
    return False


def _inventory_by_type(inventory: list[dict] | None, kind: str) -> list[dict]:
    return [item for item in inventory or [] if item.get("type") == kind]


def _classification(item: Mapping | None, name: str = "") -> dict:
    if item:
        return item.get("classification") or {
            "family": item.get("family", "unknown"), "variant": item.get("variant"),
            "confidence": "unknown", "source": "unknown",
        }
    return classify_asset(name)


def _find_model(inventory: list[dict] | None, kind: str, name: str) -> dict | None:
    target = name.replace("\\", "/").casefold()
    items = _inventory_by_type(inventory, kind)
    exact = [item for item in items if str(item.get("comfy_name", item.get("name", ""))).replace("\\", "/").casefold() == target]
    if len(exact) == 1:
        return exact[0]
    basename = target.rsplit("/", 1)[-1]
    matches = [item for item in items if str(item.get("comfy_name", item.get("name", ""))).replace("\\", "/").casefold().rsplit("/", 1)[-1] == basename]
    return matches[0] if len(matches) == 1 else None


def _asset_options(inventory: list[dict] | None, kind: str) -> list[dict]:
    return [{"value": item.get("comfy_name", item.get("name")), "name": item.get("name", item.get("comfy_name")),
             "family": item.get("family", "unknown"), "classification": _classification(item), "size": item.get("size")}
            for item in _inventory_by_type(inventory, kind)]


def _field_options(node_catalog: Mapping | None, node_type: str, field: str) -> list[str] | None:
    names = {"sampler": "sampler_name", "scheduler": "scheduler"}
    input_name = names.get(field)
    node = node_catalog.get(node_type, {}) if isinstance(node_catalog, Mapping) else {}
    inputs = node.get("input", {}) if isinstance(node, Mapping) else {}
    if not input_name or not isinstance(inputs, Mapping):
        return None
    for section in ("required", "optional"):
        definitions = inputs.get(section, {})
        spec = definitions.get(input_name) if isinstance(definitions, Mapping) else None
        if isinstance(spec, list) and spec and isinstance(spec[0], list):
            return [item for item in spec[0] if isinstance(item, str)]
    return None


def _compatibility_to_checkpoints(checkpoints: list[dict], lora: dict) -> list[dict]:
    result = []
    for checkpoint in checkpoints:
        result.append({"checkpoint": checkpoint["name"], "status": classify_compatibility(checkpoint, lora)["status"],
                       "message": classify_compatibility(checkpoint, lora)["message"]})
    return result


def editable_manifest(workflow: Mapping, *, inventory: list[dict] | None, node_catalog: Mapping | None) -> dict:
    parsed = _parse_workflow(workflow)
    by_id = _nodes_by_id(parsed)
    checkpoints = []
    for node in parsed["nodes"]:
        if not isinstance(node, dict) or node.get("type") != "CheckpointLoaderSimple":
            continue
        name = _resolve_value(parsed, node, "checkpoint")
        item = _find_model(inventory, "checkpoint", str(name)) if isinstance(name, str) else None
        checkpoints.append({"node_id": _id(node), "name": str(name or ""), "inventory": item,
                            "classification": _classification(item, str(name or ""))})

    fields = []
    unsupported = []
    seen_bindings = set()
    node_titles = {_id(node): node.get("title") or node.get("type") for node in parsed["nodes"] if isinstance(node, dict)}
    for node in parsed["nodes"]:
        if not isinstance(node, dict):
            continue
        node_type = node.get("type")
        sem_fields = WIDGET_FIELDS.get(node_type, {})
        for semantic in sem_fields:
            if semantic == "denoise" and not _uses_image_latent(parsed, node):
                continue
            binding = _widget_binding(parsed, node, semantic)
            if not binding:
                unsupported.append({"node_id": _id(node), "node_type": node_type, "field": semantic,
                                    "reason": "値が未保存か、別ノードから接続されているため安全にpatchできません。"})
                continue
            target_node, index = binding
            binding_key = (_id(target_node), semantic)
            if binding_key in seen_bindings:
                continue
            seen_bindings.add(binding_key)
            current = target_node["widgets_values"][index]
            field = {
                "key": f"{binding_key[0]}:{semantic}", "node_id": _id(node), "binding_node_id": binding_key[0],
                "node_type": node_type, "binding_node_type": target_node.get("type"), "node_title": node_titles.get(_id(node)),
                "field": semantic, "label": FIELD_LABELS[semantic],
                "value": str(current) if semantic in NUMBER_FIELDS else current,
                "value_type": "number" if semantic in NUMBER_FIELDS else "string",
                "options": None,
            }
            if semantic == "checkpoint":
                options = _asset_options(inventory, "checkpoint") or []
                current_loras = []
                for lora_node in parsed["nodes"]:
                    if isinstance(lora_node, dict) and lora_node.get("type") in {"LoraLoader", "LoraLoaderModelOnly"}:
                        lora_name = _resolve_value(parsed, lora_node, "lora")
                        lora_item = _find_model(inventory, "lora", str(lora_name)) if isinstance(lora_name, str) else None
                        if lora_item:
                            current_loras.append(lora_item)
                for option in options:
                    item = _find_model(inventory, "checkpoint", str(option["value"]))
                    option["lora_compatibility"] = [classify_compatibility(item, lora)["status"] for lora in current_loras] if item else []
                field["options"] = options
            elif semantic == "lora":
                lora_item = _find_model(inventory, "lora", str(current))
                options = _asset_options(inventory, "lora") or []
                for option in options:
                    candidate = _find_model(inventory, "lora", str(option["value"]))
                    option["compatibility"] = [classify_compatibility(checkpoint["inventory"], candidate)["status"]
                                                for checkpoint in checkpoints if checkpoint["inventory"] and candidate]
                field["options"] = options
                if lora_item:
                    field["compatibility"] = _compatibility_to_checkpoints(checkpoints, lora_item)
            elif semantic == "sampler":
                field["options"] = _field_options(node_catalog, node_type, semantic)
            elif semantic == "scheduler":
                field["options"] = _field_options(node_catalog, node_type, semantic)
            fields.append(field)

        if node_type in {"LoraLoader", "LoraLoaderModelOnly"}:
            loader_name = _resolve_value(parsed, node, "lora")
            loader = _find_model(inventory, "lora", str(loader_name)) if isinstance(loader_name, str) else None
            lora_record = {"name": str(loader_name or ""), "classification": _classification(loader, str(loader_name or ""))}
            removal_snapshot = {"lora": str(loader_name or "")}
            for field_name in ("strength_model", "strength_clip"):
                binding = _widget_binding(parsed, node, field_name)
                if binding:
                    removal_snapshot[field_name] = str(binding[0]["widgets_values"][binding[1]])
            fields.append({
                "key": f"{_id(node)}:remove_lora", "node_id": _id(node), "field": "remove_lora",
                "label": "LoRA Loader", "node_type": node_type, "node_title": node_titles.get(_id(node)),
                "value": removal_snapshot, "lora_name": str(loader_name or ""),
                "lora_options": _asset_options(inventory, "lora"),
                "compatibility": _compatibility_to_checkpoints(checkpoints, loader) if loader else [],
            })
    return {
        "editable": bool(fields), "fields": fields, "unsupported_fields": unsupported,
        "model_inventory_available": inventory is not None, "node_catalog_available": node_catalog is not None,
        "supported_nodes": sorted(WIDGET_FIELDS), "workflow_type": "comfyui_ui_workflow",
    }


def _node_title(node: Mapping) -> str:
    return str(node.get("title") or node.get("type") or "node")


def _same_expected(actual: Any, expected: Any, field: str) -> bool:
    if field == "remove_lora":
        return isinstance(expected, Mapping) and expected == actual
    if field in NUMBER_FIELDS:
        try:
            if field in INT_FIELDS:
                return int(actual) == int(expected) and float(actual) == int(actual)
            return math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-12)
        except (TypeError, ValueError, OverflowError):
            return False
    return type(actual) is type(expected) and actual == expected


def _normalize_new_value(field: str, value: Any) -> Any:
    if field in INT_FIELDS:
        text = str(value).strip()
        if not text or not text.lstrip("+").isdigit():
            raise ValueError(f"{FIELD_LABELS[field]}は整数で入力してください。")
        result = int(text)
        limits = {"seed": (0, 2**64 - 1), "steps": (1, 150), "width": (64, 4096), "height": (64, 4096)}[field]
        if not limits[0] <= result <= limits[1]:
            raise ValueError(f"{FIELD_LABELS[field]}が許容範囲外です。")
        return result
    if field in {"cfg", "denoise", "strength_model", "strength_clip"}:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{FIELD_LABELS[field]}は数値で入力してください。") from exc
        low, high = {"cfg": (0, 50), "denoise": (0, 1), "strength_model": (-2, 2), "strength_clip": (-2, 2)}[field]
        if not math.isfinite(result) or not low <= result <= high:
            raise ValueError(f"{FIELD_LABELS[field]}が許容範囲外です。")
        return result
    if field in {"checkpoint", "lora", "sampler", "scheduler"}:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{FIELD_LABELS[field]}を選択してください。")
        return value.strip()
    raise ValueError("対象外のWorkflow項目です。")


def _remove_lora(workflow: dict, node: dict) -> str:
    """Remove a core LoRA loader and reconnect its input sources to downstream users."""
    node_id = _id(node)
    node_type = node.get("type")
    pairs = {"LoraLoader": {0: 0, 1: 1}, "LoraLoaderModelOnly": {0: 0}}
    pass_through = pairs.get(node_type)
    records = _link_records(workflow)
    incoming = {record["target_slot"]: record for record in records if record["target_id"] == node_id}
    outgoing = [record for record in records if record["origin_id"] == node_id]
    replacements = {}
    by_id = _nodes_by_id(workflow)
    for record in outgoing:
        upstream_slot = pass_through.get(record["origin_slot"])
        source = incoming.get(upstream_slot) if upstream_slot is not None else None
        if source is None:
            raise ValueError("LoRA Loaderの接続を安全にバイパスできないため削除できません。")
        source_node = by_id.get(source["origin_id"])
        if source_node is None:
            raise ValueError("LoRA Loaderの上流nodeが見つからないため削除できません。")
        replacements[str(record["id"])] = (source["origin_id"], source["origin_slot"], source["id"], source_node.get("id"))

    delete_ids = {str(record["id"]) for record in incoming.values()}
    new_links = []
    for raw in workflow.get("links", []):
        record = _link_record(raw)
        if not record:
            new_links.append(raw)
            continue
        if str(record["id"]) in delete_ids:
            continue
        replacement = replacements.get(str(record["id"]))
        if replacement:
            record["origin_id"], record["origin_slot"] = replacement[0], replacement[1]
            if isinstance(raw, list):
                new_links.append([raw[0], replacements[str(record["id"])][3],
                                  record["origin_slot"], raw[3], raw[4], raw[5]])
            else:
                updated = dict(raw)
                updated["origin_id"], updated["origin_slot"] = replacements[str(record["id"])][3], record["origin_slot"]
                new_links.append(updated)
        else:
            new_links.append(raw)
    workflow["links"] = new_links
    workflow["nodes"] = [item for item in workflow["nodes"] if not (isinstance(item, Mapping) and _id(item) == node_id)]

    # Only upstream output link lists change; all other node metadata remains byte-for-byte equivalent in structure.
    touched_sources = {(source_id, source_slot, raw_id) for source_id, source_slot, _link_id, raw_id in replacements.values()}
    for source_id, source_slot, _raw_id in touched_sources:
        source_node = _nodes_by_id(workflow).get(source_id)
        if not source_node or not isinstance(source_node.get("outputs"), list):
            continue
        if not isinstance(source_slot, int) or not 0 <= source_slot < len(source_node["outputs"]):
            raise ValueError("LoRA Loaderの上流出力slotが不正なため削除できません。")
        socket = source_node["outputs"][source_slot]
        if not isinstance(socket, dict):
            raise ValueError("LoRA Loaderの上流出力形式が不正なため削除できません。")
        preserved = socket.get("links")
        if isinstance(preserved, list):
            preserved = [item for item in preserved if str(item) not in delete_ids]
        added = [record["id"] for record in outgoing
                 if str(record["id"]) in replacements
                 and replacements[str(record["id"])][0] == source_id
                 and replacements[str(record["id"])][1] == source_slot]
        socket["links"] = list(dict.fromkeys((preserved or []) + added)) or (None if preserved is None else [])
    return str(_resolve_value(workflow, node, "lora") or "LoRA")


def _apply_operations(original: dict, operations: list[dict], *, inventory: list[dict] | None,
                      node_catalog: Mapping | None) -> tuple[dict, list[dict]]:
    edited = copy.deepcopy(original)
    applied = []
    used = set()
    for operation in operations:
        if not isinstance(operation, Mapping):
            raise ValueError("Patch形式が正しくありません。")
        node_id = str(operation.get("node_id", ""))
        field = operation.get("field")
        key = (node_id, field)
        if key in used:
            raise ValueError("同じWorkflow項目へのPatchが重複しています。")
        used.add(key)
        node = _nodes_by_id(edited).get(node_id)
        if not node:
            raise ValueError(f"node ID {node_id}が見つかりません。Workflowを読み直してください。")
        expected = operation.get("expected_old")
        new_value = operation.get("new_value")
        if field == "remove_lora":
            if node.get("type") not in {"LoraLoader", "LoraLoaderModelOnly"}:
                raise ValueError("既知のLoRA Loader以外は削除できません。")
            if any(str(item.get("node_id")) == node_id and item.get("field") != "remove_lora" for item in operations):
                raise ValueError("LoRAを削除する場合、同じLoaderの置換やWeight変更は同時に指定できません。")
            actual = {"lora": str(_resolve_value(edited, node, "lora") or "")}
            for key_name in ("strength_model", "strength_clip"):
                binding = _widget_binding(edited, node, key_name)
                if binding:
                    actual[key_name] = str(binding[0]["widgets_values"][binding[1]])
            if actual != expected:
                raise ValueError("LoRA Loaderの現在値が編集開始時と異なります。元Workflowを再読込してください。")
            old_name = str(_resolve_value(edited, node, "lora") or "LoRA")
            _remove_lora(edited, node)
            applied.append({"node_id": node_id, "field": field, "label": "LoRA", "old": old_name, "new": "削除"})
            continue
        if field not in FIELD_LABELS:
            raise ValueError("編集対象外の項目です。")
        binding = _widget_binding(edited, node, field)
        if not binding:
            raise ValueError(f"{_node_title(node)}の{FIELD_LABELS[field]}は安全に編集できません。")
        target_node, index = binding
        actual = target_node["widgets_values"][index]
        if not _same_expected(actual, expected, field):
            raise ValueError(f"{_node_title(node)}の{FIELD_LABELS[field]}が編集開始時から変わっています。元Workflowを再読込してください。")
        normalized = _normalize_new_value(field, new_value)
        if field in {"checkpoint", "lora"}:
            kind = field
            if inventory is None:
                raise ValueError("親ComfyUIのモデル一覧を取得できないため、モデルを変更できません。")
            if not _find_model(inventory, kind, normalized):
                raise ValueError(f"親ComfyUIに指定した{FIELD_LABELS[field]}がありません: {normalized}")
        if field in {"sampler", "scheduler"}:
            names = _field_options(node_catalog, node.get("type"), field)
            if names is None:
                raise ValueError(f"親ComfyUIから{FIELD_LABELS[field]}の選択肢を取得できません。")
            if normalized not in names:
                raise ValueError(f"親ComfyUIで利用できない{FIELD_LABELS[field]}です: {normalized}")
        target_node["widgets_values"][index] = normalized
        applied.append({"node_id": node_id, "field": field, "label": FIELD_LABELS[field],
                        "old": str(actual), "new": str(normalized), "node_title": _node_title(node)})
    return edited, applied


def _effective_asset_records(workflow: Mapping, inventory: list[dict] | None) -> tuple[list[dict], list[dict]]:
    checkpoints, loras = [], []
    for node in workflow.get("nodes", []):
        if not isinstance(node, dict):
            continue
        if node.get("type") == "CheckpointLoaderSimple":
            name = _resolve_value(workflow, node, "checkpoint")
            if isinstance(name, str) and name:
                item = _find_model(inventory, "checkpoint", name)
                checkpoints.append({"name": name, "classification": _classification(item, name), "status": "found" if item else ("unknown" if inventory is None else "missing")})
        if node.get("type") in {"LoraLoader", "LoraLoaderModelOnly"}:
            name = _resolve_value(workflow, node, "lora")
            if isinstance(name, str) and name:
                item = _find_model(inventory, "lora", name)
                loras.append({"name": name, "classification": _classification(item, name), "status": "found" if item else ("unknown" if inventory is None else "missing")})
    return checkpoints, loras


def validate_edited_workflow(workflow: Mapping, *, inventory: list[dict] | None, node_catalog: Mapping | None) -> dict:
    blockers, warnings = [], []
    try:
        parsed = _parse_workflow(workflow)
    except ValueError as exc:
        return {"valid": False, "can_save": False, "can_queue": False, "blockers": [str(exc)], "warnings": []}
    analysis = analyze_workflow_json(parsed, inventory=inventory, node_catalog=node_catalog)
    nodes = parsed["nodes"]
    ids = [_id(node) for node in nodes if isinstance(node, Mapping) and "id" in node]
    if len(ids) != len(nodes) or len(ids) != len(set(ids)):
        blockers.append("Node IDが欠落または重複しています。")
    structural_warning = [message for message in analysis.get("warnings", [])
                          if any(token in message for token in ("link", "リンク", "存在しないノード", "重複したnode id", "typeがありません"))]
    blockers.extend(structural_warning)
    for model in analysis.get("models", []):
        if model.get("status") == "missing":
            blockers.append(f"親ComfyUIに{model.get('type')}がありません: {model.get('name')}")
        elif model.get("status") == "unknown":
            warnings.append(f"モデル在庫を照合できません: {model.get('name')}")
    for node in analysis.get("missing_nodes", []):
        if node["type"] in API_UI_NODE_TYPES or node["type"] in WIDGETLESS_UI_NODES:
            blockers.append(f"親ComfyUIにノードがありません: {node['type']}")
        else:
            warnings.append(f"未登録またはCustom Nodeをそのまま保持します（実行前に確認が必要）: {node['type']}")
    for node in analysis.get("unknown_nodes", []):
        warnings.append(f"編集未対応のノードを保持します: {node['type']} (ID {node['node_id']})")

    checkpoints, loras = _effective_asset_records(parsed, inventory)
    checkpoint_items = [_find_model(inventory, "checkpoint", item["name"]) for item in checkpoints]
    lora_items = [_find_model(inventory, "lora", item["name"]) for item in loras]
    for checkpoint in checkpoints:
        if checkpoint["status"] == "found":
            continue
        if checkpoint["status"] == "missing":
            blockers.append(f"Checkpointが親ComfyUIにありません: {checkpoint['name']}")
    for checkpoint, checkpoint_item in zip(checkpoints, checkpoint_items):
        if not checkpoint_item:
            continue
        for lora, lora_item in zip(loras, lora_items):
            if not lora_item:
                continue
            compatibility = classify_compatibility(checkpoint_item, lora_item)
            if compatibility["status"] == "incompatible":
                blockers.append(f"Checkpoint {checkpoint['name']} とLoRA {lora['name']} は互換性がありません。")
            elif compatibility["status"] in {"compatible_with_warning", "unknown"}:
                warnings.append(f"Checkpoint {checkpoint['name']} とLoRA {lora['name']}: {compatibility['message']}")

    sampler_invalid = []
    if isinstance(node_catalog, Mapping):
        for node in nodes:
            if not isinstance(node, dict) or node.get("type") not in {"KSampler", "KSamplerAdvanced"}:
                continue
            for field in ("sampler", "scheduler"):
                binding = _widget_binding(parsed, node, field)
                if not binding:
                    continue
                value = binding[0]["widgets_values"][binding[1]]
                options = _field_options(node_catalog, node.get("type"), field)
                if options is not None and value not in options:
                    message = f"{_node_title(node)}の{FIELD_LABELS[field]}が親ComfyUIで利用できません: {value}"
                    warnings.append(message)
                    sampler_invalid.append(message)

    try:
        ui_workflow_to_api_prompt(parsed)
        api_convertible = True
    except ValueError as exc:
        api_convertible = False
        warnings.append(str(exc))
    online = inventory is not None and node_catalog is not None
    if not online:
        warnings.append("親ComfyUIへ接続できないため、Queue実行の最終検証はできません。")
    if sampler_invalid:
        blockers.extend(sampler_invalid)
    blockers = list(dict.fromkeys(blockers))
    warnings = list(dict.fromkeys(warnings))
    valid = not blockers
    return {"valid": valid, "can_save": valid, "can_queue": valid and online and api_convertible,
            "api_convertible": api_convertible, "blockers": blockers, "warnings": warnings,
            "compatibility": [
                {"checkpoint": checkpoint["name"], "lora": lora["name"], **classify_compatibility(cp_item, lora_item)}
                for checkpoint, cp_item in zip(checkpoints, checkpoint_items) if cp_item
                for lora, lora_item in zip(loras, lora_items) if lora_item
            ], "models": analysis.get("models", []), "nodes": analysis.get("nodes", [])}


def apply_workflow_patches(raw: str | bytes | Mapping, patches: list[dict], *, inventory: list[dict] | None,
                           node_catalog: Mapping | None) -> dict:
    original = _parse_workflow(raw)
    edited, applied = _apply_operations(original, patches, inventory=inventory, node_catalog=node_catalog)
    validation = validate_edited_workflow(edited, inventory=inventory, node_catalog=node_catalog)
    for change in applied:
        if change["field"] != "checkpoint":
            continue
        old_item = _find_model(inventory, "checkpoint", change["old"])
        new_item = _find_model(inventory, "checkpoint", change["new"])
        if not old_item or not new_item:
            continue
        compatibility = classify_compatibility(old_item, new_item)
        if compatibility["status"] == "incompatible":
            validation["blockers"].append("Checkpoint変更でモデル系統が変わるため、このWorkflowの構成では安全に実行できません。")
        elif compatibility["status"] in {"compatible_with_warning", "unknown"}:
            validation["warnings"].append("Checkpointの系統またはPony分類に確認が必要です。")
    validation["blockers"] = list(dict.fromkeys(validation["blockers"]))
    validation["warnings"] = list(dict.fromkeys(validation["warnings"]))
    validation["valid"] = not validation["blockers"]
    validation["can_save"] = validation["valid"]
    validation["can_queue"] = validation["valid"] and validation["can_queue"]
    return {"workflow": edited, "workflow_json": json.dumps(edited, ensure_ascii=False, indent=2),
            "diff": applied, "validation": validation, "patch_count": len(applied)}


def _primitive_value(node: Mapping):
    values = node.get("widgets_values")
    if not isinstance(values, list) or not values:
        raise ValueError("PrimitiveNodeの値を解決できません。")
    return values[0]


def _resolve_link(workflow: Mapping, link_id: Any, *, visiting: set[str] | None = None):
    visiting = set(visiting or ())
    key = str(link_id)
    if key in visiting:
        raise ValueError("Workflowのリンクに循環があります。")
    visiting.add(key)
    record = next((item for item in _link_records(workflow) if str(item["id"]) == key), None)
    if not record:
        raise ValueError(f"link {link_id}が見つかりません。")
    by_id = _nodes_by_id(workflow)
    source = by_id.get(record["origin_id"])
    if not source:
        raise ValueError(f"link {link_id}の接続元nodeがありません。")
    if source.get("type") == "PrimitiveNode":
        return _primitive_value(source)
    if source.get("type") == "Reroute":
        input_socket = next((item for item in source.get("inputs", []) if isinstance(item, Mapping) and item.get("link") is not None), None)
        if not input_socket:
            raise ValueError("入力のないRerouteを解決できません。")
        return _resolve_link(workflow, input_socket["link"], visiting=visiting)
    return [record["origin_id"], record["origin_slot"]]


def ui_workflow_to_api_prompt(workflow: Mapping) -> dict:
    """Convert the explicitly supported ComfyUI core UI nodes; reject unknown execution semantics."""
    parsed = _parse_workflow(workflow)
    by_id = _nodes_by_id(parsed)
    prompt = {}
    widget_names = {
        "CheckpointLoaderSimple": ["ckpt_name"],
        "UNETLoader": ["unet_name", "weight_dtype"], "DualCLIPLoader": ["clip_name1", "clip_name2", "type", "device"],
        "CLIPLoader": ["clip_name", "type", "device"], "VAELoader": ["vae_name"],
        "LoraLoader": ["lora_name", "strength_model", "strength_clip"],
        "LoraLoaderModelOnly": ["lora_name", "strength_model"],
        "CLIPTextEncode": ["text"], "FluxGuidance": ["guidance"],
        "EmptyLatentImage": ["width", "height", "batch_size"], "EmptySD3LatentImage": ["width", "height", "batch_size"],
        "KSampler": ["seed", "control_after_generate", "steps", "cfg", "sampler_name", "scheduler", "denoise"],
        "KSamplerAdvanced": ["add_noise", "noise_seed", "control_after_generate", "steps", "cfg", "sampler_name", "scheduler", "start_at_step", "end_at_step", "return_with_leftover_noise"],
        "SaveImage": ["filename_prefix"], "PreviewImage": [], "LoadImage": ["image"], "VAEEncode": [],
        "UpscaleModelLoader": ["model_name"], "ImageUpscaleWithModel": [], "VAEDecode": [],
    }
    for node in parsed["nodes"]:
        if not isinstance(node, dict):
            raise ValueError("Workflowに不正なnodeが含まれています。")
        node_type = node.get("type")
        if node_type in WIDGETLESS_UI_NODES:
            continue
        if node_type not in API_UI_NODE_TYPES or node_type not in widget_names:
            raise ValueError(f"{node_type}はAPI Promptへの変換未対応です。編集内容は保持できますがQueue実行できません。")
        if node.get("mode", 0) not in (0, None):
            raise ValueError(f"無効化またはバイパス中の{node_type}があるため、Queue実行できません。")
        values = node.get("widgets_values") if isinstance(node.get("widgets_values"), list) else []
        inputs = {name: value for name, value in zip(widget_names[node_type], values)
                  if name not in API_OMIT_WIDGETS and name not in {"seed_control"}}
        for socket in node.get("inputs", []) if isinstance(node.get("inputs"), list) else []:
            if not isinstance(socket, Mapping) or socket.get("link") is None:
                continue
            widget = socket.get("widget")
            input_name = widget.get("name") if isinstance(widget, Mapping) else socket.get("name")
            if not isinstance(input_name, str):
                raise ValueError(f"{node_type}の入力名を判定できません。")
            inputs[input_name] = _resolve_link(parsed, socket["link"])
        inputs = {key: value for key, value in inputs.items() if key not in API_OMIT_WIDGETS and key != "seed_control"}
        prompt[str(node["id"])] = {"class_type": node_type, "inputs": inputs}
    if not prompt:
        raise ValueError("ComfyUI APIへ送信できる実行ノードがありません。")
    for node_id, node in prompt.items():
        for name, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], (int, str)):
                if str(value[0]) not in prompt:
                    raise ValueError(f"{node_id}.{name}の接続先node {value[0]}を変換できません。")
    return prompt
