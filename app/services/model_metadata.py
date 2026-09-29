"""Conservative family classification and compatibility for ComfyUI assets."""

from collections.abc import Mapping
import re

from app.services.model_classifier import classify_model

FAMILIES = {"sd15", "sdxl", "flux", "unknown"}


def _metadata_values(metadata: Mapping) -> dict[str, str]:
    values = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float)):
            values[str(key).lower()] = str(value).strip()
    nested = metadata.get("__metadata__")
    if isinstance(nested, Mapping):
        for key, value in nested.items():
            if isinstance(value, (str, int, float)):
                values[str(key).lower()] = str(value).strip()
    return values


def metadata_role(metadata: Mapping) -> str | None:
    values = _metadata_values(metadata)
    architecture = values.get("modelspec.architecture", "").lower()
    network = values.get("ss_network_module", "").lower()
    if architecture.endswith("/lora") or ".lora" in network or "lora" in network:
        return "lora"
    if architecture or values.get("ss_base_model_version") or values.get("base_model"):
        return "checkpoint"
    return None


def classify_asset(name: str, metadata: Mapping | None = None, manual: Mapping | None = None) -> dict:
    """Return family, optional Pony variant, confidence, source and evidence.

    Explicit user classification wins over inference. Metadata can be high
    confidence only when it names a recognized base family; filenames remain
    low confidence and never block a queue by themselves.
    """
    if manual:
        family = manual.get("family", "unknown")
        if family not in FAMILIES:
            family = "unknown"
        variant = manual.get("variant") if family == "sdxl" else None
        return {
            "family": family, "variant": variant, "confidence": "high",
            "source": "manual", "evidence": ["ユーザーが手動分類"],
        }

    values = _metadata_values(metadata or {})
    candidates = [
        values.get("modelspec.architecture", ""), values.get("modelspec.title", ""),
        values.get("modelspec.description", ""), values.get("ss_base_model_version", ""),
        values.get("base_model", ""), values.get("base_model_version", ""),
        values.get("ss_sd_model_name", ""), values.get("ss_output_name", ""),
    ]
    text = " ".join(candidates).lower()
    filename_pony = "pony" in name.lower()
    if text.strip():
        if "pony" in text:
            return {"family": "sdxl", "variant": "pony", "confidence": "high", "source": "metadata", "variant_confidence": "high", "variant_source": "metadata", "evidence": ["safetensors metadataにPonyの識別情報"]}
        if any(token in text for token in ("flux", "black-forest-labs")):
            return {"family": "flux", "variant": None, "confidence": "high", "source": "metadata", "evidence": ["safetensors metadataにFluxの識別情報"]}
        if any(token in text for token in ("stable-diffusion-xl", "sdxl", "sd_xl", "sd-xl")):
            evidence = ["safetensors metadataにSDXLの識別情報"]
            if filename_pony:
                evidence.append("Pony variantはファイル名からの候補")
            return {"family": "sdxl", "variant": "pony" if filename_pony else None, "confidence": "high", "source": "metadata",
                    "variant_confidence": "low" if filename_pony else None, "variant_source": "filename" if filename_pony else None, "evidence": evidence}
        if any(token in text for token in ("stable-diffusion-v1-5", "stable-diffusion-v1.5", "sd-v1-5", "sd15", "sd 1.5")):
            return {"family": "sd15", "variant": None, "confidence": "high", "source": "metadata", "evidence": ["safetensors metadataにSD1.5の識別情報"]}

    family = classify_model(name)
    normalized_name = name.lower().replace("_", "-")
    if family == "unknown" and re.search(r"(?:^|[._-])v1[._-]?5(?:[._-]|$)", name.lower()):
        family = "sd15"
    variant = "pony" if family == "sdxl" and "pony" in normalized_name else None
    if family in FAMILIES and family != "unknown":
        return {"family": family, "variant": variant, "confidence": "low", "source": "filename", "evidence": ["ファイル名からの候補"]}
    return {"family": "unknown", "variant": None, "confidence": "unknown", "source": "unknown", "evidence": []}


def classify_compatibility(checkpoint: Mapping, lora: Mapping) -> dict:
    checkpoint_class = checkpoint.get("classification") or {
        "family": checkpoint.get("family", "unknown"), "variant": checkpoint.get("variant"),
        "confidence": "unknown", "source": "unknown",
    }
    lora_class = lora.get("classification") or {
        "family": lora.get("family", "unknown"), "variant": lora.get("variant"),
        "confidence": "unknown", "source": "unknown",
    }
    checkpoint_family = checkpoint_class.get("family", "unknown")
    lora_family = lora_class.get("family", "unknown")
    base = {"checkpoint": checkpoint_class, "lora": lora_class}
    if "unknown" in (checkpoint_family, lora_family):
        return {**base, "status": "unknown", "message": "ベースモデルを判定できないため、互換性を確認してください。"}

    reliable_checkpoint = checkpoint_class.get("confidence") == "high" or checkpoint_class.get("source") == "manual"
    reliable_lora = lora_class.get("confidence") == "high" or lora_class.get("source") == "manual"
    if checkpoint_family != lora_family:
        if reliable_checkpoint and reliable_lora:
            return {**base, "status": "incompatible", "message": "CheckpointとLoRAのベースモデル系統が異なります。"}
        return {**base, "status": "compatible_with_warning", "message": "ファイル名等の候補分類では系統が異なります。分類を確認してください。"}

    if checkpoint_family == "sdxl":
        checkpoint_variant = checkpoint_class.get("variant")
        lora_variant = lora_class.get("variant")
        if checkpoint_variant != lora_variant:
            return {**base, "status": "compatible_with_warning", "message": "Ponyと一般SDXLの組み合わせです。動作を確認してください。"}
    if reliable_checkpoint and reliable_lora:
        return {**base, "status": "compatible", "message": "分類されたベースモデル系統が一致しています。"}
    return {**base, "status": "compatible_with_warning", "message": "系統は一致していますが、分類の確度が十分ではありません。"}
