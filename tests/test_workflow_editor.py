import copy
import json
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.schemas.workflow import WorkflowBuildRequest
from app.services.workflow_builder import build_definition, to_ui_workflow
from app.services.workflow_editor import (
    apply_workflow_patches,
    editable_manifest,
    ui_workflow_to_api_prompt,
    validate_edited_workflow,
)


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "comfy"


def asset(name, kind, family="sdxl", variant=None):
    classification = {"family": family, "variant": variant, "confidence": "high", "source": "metadata"}
    return {"name": name, "comfy_name": name, "type": kind, "family": family,
            "variant": variant, "classification": classification}


def standard_inventory(*, lora_family="sdxl", lora_variant=None):
    return [
        asset("base.safetensors", "checkpoint"),
        asset("replacement.safetensors", "checkpoint"),
        asset("sd15.safetensors", "checkpoint", "sd15"),
        asset("style.safetensors", "lora", lora_family, lora_variant),
        asset("style-sdxl.safetensors", "lora", "sdxl"),
        asset("style-sd15.safetensors", "lora", "sd15"),
        asset("4x-model.pth", "upscale_model", "unknown"),
    ]


def catalog(*node_types):
    result = {node_type: {"python_module": "comfy.core", "input": {"required": {}}} for node_type in node_types}
    for node_type in ("KSampler", "KSamplerAdvanced"):
        if node_type in result:
            result[node_type]["input"]["required"] = {
                "sampler_name": [["euler", "dpmpp_2m", "heun"], {}],
                "scheduler": [["normal", "karras"], {}],
            }
    return result


def sdxl_workflow(*, lora=True, generation_type="txt2img", seed=42):
    request = WorkflowBuildRequest(
        model="base.safetensors", prompt="portrait", negative_prompt="blur", seed=seed,
        width=1024, height=1024, steps=20, cfg=6.5, sampler="euler", scheduler="normal",
        lora="style.safetensors" if lora else None, lora_weight=0.8, generation_type=generation_type,
        denoise=0.45,
    )
    input_image = "source.png" if generation_type == "img2img" else None
    return to_ui_workflow(build_definition(request, "sdxl", input_image))


class WorkflowEditorTests(unittest.TestCase):
    def test_manifest_uses_parent_sampler_values_and_hides_txt2img_denoise(self):
        workflow = sdxl_workflow()
        node_catalog = catalog(*(node["type"] for node in workflow["nodes"]))
        manifest = editable_manifest(workflow, inventory=standard_inventory(), node_catalog=node_catalog)
        fields = {(item["node_id"], item["field"]): item for item in manifest["fields"]}
        self.assertIn(("5", "seed"), fields)
        self.assertIn(("5", "steps"), fields)
        self.assertEqual(fields[("5", "sampler")]["options"], ["euler", "dpmpp_2m", "heun"])
        self.assertEqual(fields[("5", "scheduler")]["options"], ["normal", "karras"])
        self.assertNotIn(("5", "denoise"), fields)
        self.assertIn(("4", "width"), fields)
        self.assertIn(("4", "height"), fields)

    def test_img2img_manifest_exposes_denoise(self):
        workflow = sdxl_workflow(generation_type="img2img")
        manifest = editable_manifest(workflow, inventory=standard_inventory(), node_catalog=catalog("KSampler"))
        self.assertTrue(any(field["field"] == "denoise" for field in manifest["fields"]))

    def test_patch_keeps_original_and_unedited_metadata(self):
        workflow = sdxl_workflow()
        workflow["groups"] = [{"title": "preserve this group", "bounding": [10, 20, 300, 400], "color": "#abc"}]
        workflow["config"] = {"keep": {"revision": 3}}
        workflow["extra"] = {"opaque": [1, {"name": "retained"}]}
        workflow["nodes"].append({"id": 900, "type": "UnknownCustomNode", "pos": [9, 8], "size": [7, 6],
                                  "widgets_values": [{"opaque": True}], "properties": {"opaque": "keep"},
                                  "inputs": [], "outputs": [], "flags": {"collapsed": True}})
        original_snapshot = copy.deepcopy(workflow)
        original_json = json.dumps(workflow, ensure_ascii=False)
        request = [
            {"node_id": "1", "field": "checkpoint", "expected_old": "base.safetensors", "new_value": "replacement.safetensors"},
            {"node_id": "5", "field": "steps", "expected_old": "20", "new_value": "30"},
            {"node_id": "4", "field": "width", "expected_old": "1024", "new_value": "832"},
            {"node_id": "4", "field": "height", "expected_old": "1024", "new_value": "1216"},
            {"node_id": "5", "field": "cfg", "expected_old": "6.5", "new_value": "7.25"},
            {"node_id": "5", "field": "sampler", "expected_old": "euler", "new_value": "dpmpp_2m"},
            {"node_id": "5", "field": "scheduler", "expected_old": "normal", "new_value": "karras"},
        ]
        result = apply_workflow_patches(original_json, request, inventory=standard_inventory(),
                                        node_catalog=catalog(*(node["type"] for node in workflow["nodes"])))
        self.assertEqual(workflow, original_snapshot)
        edited = json.loads(result["workflow_json"])
        self.assertEqual(edited["groups"], original_snapshot["groups"])
        self.assertEqual(edited["config"], original_snapshot["config"])
        self.assertEqual(edited["extra"], original_snapshot["extra"])
        unknown = next(node for node in edited["nodes"] if node["id"] == 900)
        self.assertEqual(unknown, original_snapshot["nodes"][-1])
        checkpoint = next(node for node in edited["nodes"] if node["id"] == 1)
        sampler = next(node for node in edited["nodes"] if node["id"] == 5)
        latent = next(node for node in edited["nodes"] if node["id"] == 4)
        self.assertEqual(checkpoint["widgets_values"][0], "replacement.safetensors")
        self.assertEqual(sampler["widgets_values"][2:7], [30, 7.25, "dpmpp_2m", "karras", 1.0])
        self.assertEqual(latent["widgets_values"][:2], [832, 1216])
        self.assertTrue(result["validation"]["can_save"])
        self.assertEqual(result["patch_count"], 7)

    def test_lora_replace_weight_and_compatibility_block(self):
        workflow = sdxl_workflow()
        request = [
            {"node_id": "8", "field": "lora", "expected_old": "style.safetensors", "new_value": "style-sd15.safetensors"},
            {"node_id": "8", "field": "strength_model", "expected_old": "0.8", "new_value": "0.65"},
            {"node_id": "8", "field": "strength_clip", "expected_old": "0.8", "new_value": "0.4"},
        ]
        result = apply_workflow_patches(workflow, request, inventory=standard_inventory(),
                                        node_catalog=catalog(*(node["type"] for node in workflow["nodes"])))
        self.assertFalse(result["validation"]["can_save"])
        self.assertTrue(any("互換性がありません" in item for item in result["validation"]["blockers"]))
        lora = next(node for node in json.loads(result["workflow_json"])["nodes"] if node["type"] == "LoraLoader")
        self.assertEqual(lora["widgets_values"], ["style-sd15.safetensors", 0.65, 0.4])

    def test_lora_removal_bypasses_loader_and_keeps_links_consistent(self):
        workflow = sdxl_workflow()
        manifest = editable_manifest(workflow, inventory=standard_inventory(), node_catalog=catalog("LoraLoader"))
        field = next(field for field in manifest["fields"] if field["field"] == "remove_lora")
        result = apply_workflow_patches(workflow, [{"node_id": field["node_id"], "field": "remove_lora",
                                                   "expected_old": field["value"], "new_value": True}],
                                        inventory=standard_inventory(), node_catalog=catalog(*(node["type"] for node in workflow["nodes"])))
        edited = json.loads(result["workflow_json"])
        self.assertFalse(any(node["type"] == "LoraLoader" for node in edited["nodes"]))
        self.assertEqual(result["validation"]["blockers"], [])
        links = edited["links"]
        link_by_id = {link[0]: link for link in links}
        self.assertEqual(link_by_id[1][1:3], [1, 1])
        self.assertEqual(link_by_id[3][1:3], [1, 0])
        checkpoint = next(node for node in edited["nodes"] if node["id"] == 1)
        self.assertEqual(checkpoint["outputs"][0]["links"], [3])
        self.assertEqual(checkpoint["outputs"][1]["links"], [1, 2])

    def test_seed_large_integer_round_trips_without_precision_loss(self):
        workflow = sdxl_workflow(lora=False, seed=9_223_372_036_854_775_808)
        manifest = editable_manifest(workflow, inventory=standard_inventory(), node_catalog=catalog("KSampler"))
        field = next(item for item in manifest["fields"] if item["field"] == "seed")
        self.assertEqual(field["value"], "9223372036854775808")
        result = apply_workflow_patches(workflow, [{"node_id": field["node_id"], "field": "seed",
                                                   "expected_old": field["value"], "new_value": "18446744073709551615"}],
                                        inventory=standard_inventory(), node_catalog=catalog("KSampler"))
        self.assertIn('18446744073709551615', result["workflow_json"])
        self.assertEqual(json.loads(result["workflow_json"])["nodes"][4]["widgets_values"][0], 18_446_744_073_709_551_615)

    def test_stale_expected_value_and_invalid_values_are_rejected(self):
        workflow = sdxl_workflow()
        with self.assertRaisesRegex(ValueError, "編集開始時から変わっています"):
            apply_workflow_patches(workflow, [{"node_id": "5", "field": "steps", "expected_old": "19", "new_value": "30"}],
                                   inventory=standard_inventory(), node_catalog=catalog("KSampler"))
        with self.assertRaisesRegex(ValueError, "許容範囲"):
            apply_workflow_patches(workflow, [{"node_id": "5", "field": "denoise", "expected_old": "1.0", "new_value": "1.5"}],
                                   inventory=standard_inventory(), node_catalog=catalog("KSampler"))
        with self.assertRaisesRegex(ValueError, "利用できないSampler"):
            apply_workflow_patches(workflow, [{"node_id": "5", "field": "sampler", "expected_old": "euler", "new_value": "unknown_sampler"}],
                                   inventory=standard_inventory(), node_catalog=catalog("KSampler"))

    def test_missing_models_block_saving_and_incompatible_checkpoint_family_blocks(self):
        workflow = sdxl_workflow(lora=False)
        missing = validate_edited_workflow(workflow, inventory=[], node_catalog=catalog("KSampler"))
        self.assertFalse(missing["can_save"])
        result = apply_workflow_patches(workflow, [{"node_id": "1", "field": "checkpoint", "expected_old": "base.safetensors", "new_value": "sd15.safetensors"}],
                                        inventory=standard_inventory(), node_catalog=catalog(*(node["type"] for node in workflow["nodes"])))
        self.assertFalse(result["validation"]["can_save"])
        self.assertTrue(any("モデル系統が変わる" in item for item in result["validation"]["blockers"]))

    def test_unknown_nodes_are_preserved_but_queue_conversion_is_blocked(self):
        workflow = sdxl_workflow(lora=False)
        custom = {"id": 501, "type": "ImpactPackFaceDetailer", "pos": [5, 6], "size": [100, 80],
                  "widgets_values": ["leave alone"], "inputs": [], "outputs": [], "properties": {"opaque": [1, 2]}}
        workflow["nodes"].append(custom)
        result = apply_workflow_patches(workflow, [{"node_id": "5", "field": "steps", "expected_old": "20", "new_value": "21"}],
                                        inventory=standard_inventory(), node_catalog=catalog(*(node["type"] for node in workflow["nodes"] if node["type"] != custom["type"])) )
        self.assertTrue(result["validation"]["can_save"])
        edited = json.loads(result["workflow_json"])
        self.assertIn(custom, edited["nodes"])
        with self.assertRaisesRegex(ValueError, "変換未対応"):
            ui_workflow_to_api_prompt(edited)

    def test_official_full_workflow_fixture_preserves_groups_extra_and_node_metadata(self):
        workflow = json.loads((FIXTURES / "comfy_org_sdxl_simple_full.json").read_text(encoding="utf-8"))
        self.assertEqual(len(workflow["nodes"]), 25)
        self.assertEqual(len(workflow["links"]), 23)
        inventory = [asset("sd_xl_base_1.0.safetensors", "checkpoint"),
                     asset("sd_xl_refiner_1.0.safetensors", "checkpoint"),
                     asset("replacement.safetensors", "checkpoint")]
        node_catalog = catalog("EmptyLatentImage", "CheckpointLoaderSimple", "CLIPTextEncode", "KSamplerAdvanced", "VAEDecode", "SaveImage")
        manifest = editable_manifest(workflow, inventory=inventory, node_catalog=node_catalog)
        self.assertTrue(manifest["editable"])
        steps = next(field for field in manifest["fields"] if field["field"] == "steps")
        self.assertEqual(steps["value"], "25")
        result = apply_workflow_patches(workflow, [
            {"node_id": "4", "field": "checkpoint", "expected_old": "sd_xl_base_1.0.safetensors", "new_value": "replacement.safetensors"},
            {"node_id": steps["node_id"], "field": "steps", "expected_old": "25", "new_value": "30"},
        ], inventory=inventory, node_catalog=node_catalog)
        edited = json.loads(result["workflow_json"])
        self.assertEqual(edited["groups"], workflow["groups"])
        self.assertEqual(edited["extra"], workflow["extra"])
        self.assertEqual(edited["config"], workflow["config"])
        self.assertEqual(edited["revision"], workflow["revision"])
        note_before = next(node for node in workflow["nodes"] if node["type"] == "Note")
        note_after = next(node for node in edited["nodes"] if node["id"] == note_before["id"])
        self.assertEqual(note_after, note_before)
        checkpoint = next(node for node in edited["nodes"] if node["id"] == 4)
        self.assertEqual(checkpoint["properties"]["models"][0]["name"], "sd_xl_base_1.0.safetensors")
        primitive = next(node for node in edited["nodes"] if node["type"] == "PrimitiveNode" and node.get("title") == "steps")
        self.assertEqual(primitive["widgets_values"][0], 30)
        prompt = ui_workflow_to_api_prompt(edited)
        samplers = [node for node in prompt.values() if node["class_type"] == "KSamplerAdvanced"]
        self.assertEqual([node["inputs"]["steps"] for node in samplers], [30, 30])
        self.assertEqual(len(prompt), 11)

    def test_api_prompt_is_analysis_only_for_editor(self):
        from app.services.workflow_builder import to_api_prompt
        prompt = to_api_prompt(build_definition(WorkflowBuildRequest(model="base.safetensors", prompt="a tree", seed=12), "sdxl"))
        with self.assertRaisesRegex(ValueError, "UI Workflow JSONが必要"):
            editable_manifest(prompt, inventory=standard_inventory(), node_catalog=catalog("KSampler"))

    def test_prepare_apply_and_explicit_queue_endpoints(self):
        workflow = sdxl_workflow(lora=False)
        original = json.dumps(workflow, ensure_ascii=False)
        inv = standard_inventory()
        node_catalog = catalog(*(node["type"] for node in workflow["nodes"]))
        with patch("app.main._workflow_edit_catalog", new=AsyncMock(return_value=(True, inv, node_catalog))), \
             patch("app.main.client") as client_factory:
            client_factory.return_value.queue = AsyncMock(return_value="prompt-edit-1")
            client_factory.return_value.status = AsyncMock(return_value={"online": True})
            client_factory.return_value.object_info_catalog = AsyncMock(return_value=node_catalog)
            client = TestClient(app)
            prepared = client.post("/api/workflow/edit/prepare", json={"workflow_json": original})
            self.assertEqual(prepared.status_code, 200)
            self.assertTrue(prepared.json()["editable"])
            result = client.post("/api/workflow/edit/apply", json={"workflow_json": original, "patches": [
                {"node_id": "5", "field": "steps", "expected_old": "20", "new_value": "30"},
            ]})
            self.assertEqual(result.status_code, 200)
            edited = json.loads(result.json()["workflow_json"])
            sampler = next(node for node in edited["nodes"] if node["type"] == "KSampler")
            self.assertEqual(sampler["widgets_values"][2], 30)
            client_factory.return_value.queue.assert_not_awaited()
            blocked = client.post("/api/workflow/edit/queue", json={"workflow_json": original, "patches": [
                {"node_id": "5", "field": "sampler", "expected_old": "euler", "new_value": "invalid"},
            ]})
            self.assertEqual(blocked.status_code, 400)
            client_factory.return_value.queue.assert_not_awaited()
            queued = client.post("/api/workflow/edit/queue", json={"workflow_json": original, "patches": [
                {"node_id": "5", "field": "steps", "expected_old": "20", "new_value": "30"},
            ]})
        self.assertEqual(queued.status_code, 200)
        self.assertEqual(queued.json()["prompt_id"], "prompt-edit-1")
        self.assertEqual(queued.json()["status"], "QUEUED")
        client_factory.return_value.queue.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
