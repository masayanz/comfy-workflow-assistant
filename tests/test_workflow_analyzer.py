import json
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.schemas.workflow import WorkflowBuildRequest
from app.services.workflow_analyzer import analyze_workflow_json
from app.services.workflow_builder import build_definition, to_api_prompt, to_ui_workflow


ROOT = Path(__file__).resolve().parent


def inventory(*, lora_family="sdxl"):
    return [
        {"name": "base.safetensors", "comfy_name": "base.safetensors", "type": "checkpoint", "family": "sdxl",
         "classification": {"family": "sdxl", "variant": None, "confidence": "high", "source": "metadata"}},
        {"name": "style.safetensors", "comfy_name": "style.safetensors", "type": "lora", "family": lora_family,
         "classification": {"family": lora_family, "variant": None, "confidence": "high", "source": "metadata"}},
        {"name": "4x-model.pth", "comfy_name": "4x-model.pth", "type": "upscale_model", "family": "unknown"},
    ]


def node_catalog(*node_names):
    return {name: {"python_module": "comfy.core"} for name in node_names}


class WorkflowAnalyzerTests(unittest.TestCase):
    # Reduced external UI graph following the node/link encoding in Comfy-Org's SDXL template.
    def test_reads_comfy_org_style_external_ui_workflow_and_extracts_parameters(self):
        fixture = ROOT / "fixtures" / "comfy" / "external_sdxl_workflow.json"
        catalog = node_catalog("EmptyLatentImage", "CheckpointLoaderSimple", "CLIPTextEncode", "KSamplerAdvanced", "VAEDecode", "SaveImage")
        result = analyze_workflow_json(fixture.read_bytes(), inventory=inventory(), node_catalog=catalog)

        self.assertTrue(result["valid"])
        self.assertEqual(result["workflow_type"], "comfyui_ui_workflow")
        self.assertEqual(result["generation_type"], "txt2img")
        self.assertEqual((result["node_count"], result["link_count"]), (7, 9))
        self.assertEqual(result["models"][0]["name"], "sd_xl_base_1.0.safetensors")
        self.assertEqual(result["models"][0]["status"], "missing")
        self.assertTrue(result["summary"]["title"].startswith("SDXL候補"))
        self.assertEqual(result["parameters"]["latents"][0]["width"], 1024)
        sampler = result["parameters"]["samplers"][0]
        self.assertEqual(sampler["steps"], 25)
        self.assertEqual(sampler["cfg"], 8)
        self.assertEqual(sampler["sampler_name"], "euler")
        self.assertEqual(sampler["seed"], 721897303308196)
        self.assertEqual(result["parameters"]["prompts"][0]["text"], "sunset over a mountain lake")
        self.assertEqual(result["missing_models"][0]["type"], "checkpoint")
        self.assertEqual(result["missing_nodes"], [])
        self.assertTrue(any("sd_xl_base_1.0.safetensors" in warning for warning in result["warnings"]))

    def test_reads_app_generated_sdxl_lora_workflow_and_checks_compatibility(self):
        request = WorkflowBuildRequest(model="base.safetensors", prompt="portrait", negative_prompt="blur", seed=42,
                                       width=1024, height=1024, steps=30, cfg=7, lora="style.safetensors", lora_weight=0.75)
        ui_workflow = to_ui_workflow(build_definition(request, "sdxl"))
        catalog = node_catalog(*(node["type"] for node in ui_workflow["nodes"]))
        result = analyze_workflow_json(ui_workflow, inventory=inventory(lora_family="sd15"), node_catalog=catalog)

        self.assertEqual(result["generation_type"], "txt2img")
        self.assertEqual({item["type"] for item in result["models"]}, {"checkpoint", "lora"})
        self.assertEqual(result["compatibility"][0]["status"], "incompatible")
        self.assertEqual(result["parameters"]["samplers"][0]["seed"], 42)
        self.assertEqual(result["parameters"]["samplers"][0]["steps"], 30)
        self.assertEqual(result["parameters"]["samplers"][0]["cfg"], 7)
        self.assertEqual(result["parameters"]["latents"][0], {"node_id": "4", "width": 1024, "height": 1024, "batch_size": 1})
        self.assertEqual(result["parameters"]["loras"][0]["strength_model"], 0.75)
        self.assertEqual(result["parameters"]["loras"][0]["strength_clip"], 0.75)
        self.assertTrue(any("系統不一致" in warning for warning in result["warnings"]))

    def test_reports_unknown_custom_node_and_preserves_other_analysis(self):
        workflow = {
            "nodes": [
                {"id": 1, "type": "CheckpointLoaderSimple", "inputs": [], "outputs": [], "properties": {}, "widgets_values": ["base.safetensors"]},
                {"id": 2, "type": "ImpactPackFaceDetailer", "inputs": [], "outputs": [],
                 "properties": {"cnr_id": "comfyui-impact-pack"}, "widgets_values": []},
            ], "links": [], "version": 0.4,
        }
        result = analyze_workflow_json(workflow, inventory=inventory(), node_catalog={"CheckpointLoaderSimple": {"python_module": "comfy.core"}})
        self.assertTrue(result["valid"])
        self.assertEqual(result["node_count"], 2)
        self.assertEqual(result["custom_nodes"][0]["status"], "missing")
        self.assertEqual(result["unknown_nodes"][0]["type"], "ImpactPackFaceDetailer")
        self.assertEqual(result["models"][0]["status"], "found")

    def test_api_prompt_is_detected_without_executing_it(self):
        definition = build_definition(WorkflowBuildRequest(model="base.safetensors", prompt="a tree", seed=7), "sdxl")
        prompt = to_api_prompt(definition)
        result = analyze_workflow_json(prompt, inventory=inventory(), node_catalog=node_catalog(*(node["class_type"] for node in prompt.values())))
        self.assertEqual(result["workflow_type"], "comfyui_api_prompt")
        self.assertTrue(result["analysis_only"])
        self.assertEqual(result["generation_type"], "txt2img")
        self.assertEqual(result["parameters"]["samplers"][0]["seed"], 7)

    def test_reads_all_existing_app_workflow_families_and_modes(self):
        cases = [
            (WorkflowBuildRequest(model="base.safetensors", prompt="portrait", seed=3), "sdxl", None, "txt2img"),
            (WorkflowBuildRequest(model="base.safetensors", prompt="portrait", lora="style.safetensors", seed=4), "sdxl", None, "txt2img"),
            (WorkflowBuildRequest(model="base.safetensors", prompt="portrait", generation_type="img2img", seed=5), "sdxl", "source.png", "img2img"),
            (WorkflowBuildRequest(model="base.safetensors", prompt="portrait", generation_type="img2img", lora="style.safetensors", seed=6), "sdxl", "source.png", "img2img"),
            (WorkflowBuildRequest(generation_type="upscale", upscale_model="4x-model.pth"), "upscale", "source.png", "upscale"),
            (WorkflowBuildRequest(profile_id="flux", prompt="a mountain lake", diffusion_model="flux1-dev.safetensors",
                                  clip_name1="clip_l.safetensors", clip_name2="t5xxl.safetensors", vae_model="ae.safetensors", seed=7), "flux", None, "txt2img"),
        ]
        for request, family, input_image, expected_kind in cases:
            with self.subTest(family=family, generation_type=request.generation_type, lora=request.lora):
                ui = to_ui_workflow(build_definition(request, family, input_image))
                result = analyze_workflow_json(ui)
                self.assertTrue(result["valid"])
                self.assertEqual(result["generation_type"], expected_kind)
                self.assertGreater(result["node_count"], 0)
                self.assertIsNotNone(result["parameters"])
                if expected_kind == "upscale":
                    self.assertEqual(result["models"][0]["type"], "upscale_model")
                if family == "flux":
                    self.assertEqual({item["type"] for item in result["models"]}, {"diffusion_model", "text_encoder", "vae"})
                    self.assertIn("Flux", result["summary"]["title"])

    def test_workflow_properties_can_supply_model_assets(self):
        workflow = {"nodes": [{"id": 1, "type": "CustomModelNode", "inputs": [], "outputs": [], "widgets_values": [],
                               "properties": {"models": [{"name": "extra.safetensors", "directory": "loras"}]}}],
                    "links": []}
        result = analyze_workflow_json(workflow, inventory=inventory())
        self.assertEqual(result["models"][0]["type"], "lora")
        self.assertEqual(result["models"][0]["status"], "missing")

    def test_reports_broken_links_but_keeps_partial_analysis(self):
        workflow = {"nodes": [{"id": 1, "type": "EmptyLatentImage", "inputs": [], "outputs": [], "widgets_values": [4096, 4096, 1]}],
                    "links": [[1, 1, 0, 999, 0, "LATENT"]]}
        result = analyze_workflow_json(workflow, inventory=[], node_catalog={"EmptyLatentImage": {}})
        self.assertTrue(result["valid"])
        self.assertEqual(result["generation_type"], "txt2img")
        self.assertTrue(any("存在しないノード" in warning for warning in result["warnings"]))
        self.assertTrue(result["vram_warnings"])

    def test_reports_input_output_link_reference_mismatches(self):
        workflow = {"nodes": [
            {"id": 1, "type": "EmptyLatentImage", "inputs": [], "outputs": [{"name": "LATENT", "type": "LATENT", "links": [4]}], "widgets_values": [512, 512, 1]},
            {"id": 2, "type": "KSampler", "inputs": [{"name": "latent_image", "type": "LATENT", "link": 4}], "outputs": [], "widgets_values": []},
        ], "links": [[5, 1, 0, 2, 0, "LATENT"]]}
        result = analyze_workflow_json(workflow, inventory=[], node_catalog={})
        self.assertTrue(result["valid"])
        self.assertGreaterEqual(len(result["warnings"]), 5)
        self.assertIn("link 5", str(result["warnings"]))

    def test_marks_inventory_as_unknown_when_parent_unavailable(self):
        prompt = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "base.safetensors"}}}
        result = analyze_workflow_json(prompt, inventory=None, node_catalog=None)
        self.assertEqual(result["models"][0]["status"], "unknown")
        self.assertTrue(any("在庫を取得できず" in warning for warning in result["warnings"]))
        self.assertEqual(result["nodes"][0]["availability"], "unknown")

    def test_classifies_unknown_and_invalid_json(self):
        result = analyze_workflow_json({"hello": "world"})
        self.assertFalse(result["valid"])
        self.assertEqual(result["workflow_type"], "unknown")
        with self.assertRaisesRegex(ValueError, "JSONを読み込めません"):
            analyze_workflow_json(b"{broken")
        with self.assertRaisesRegex(ValueError, "ルート"):
            analyze_workflow_json([])

    def test_endpoint_accepts_json_upload_and_never_queues(self):
        payload = json.dumps({"nodes": [{"id": 1, "type": "CheckpointLoaderSimple", "inputs": [], "outputs": [],
                                           "properties": {}, "widgets_values": ["base.safetensors"]}], "links": [], "version": 0.4})
        with patch("app.main.client") as factory, patch("app.main.parent_model_inventory", new=AsyncMock(return_value=inventory())):
            factory.return_value.status = AsyncMock(return_value={"online": True, "url": "http://comfy"})
            factory.return_value.object_info_catalog = AsyncMock(return_value=node_catalog("CheckpointLoaderSimple"))
            factory.return_value.queue = AsyncMock(return_value="should-not-queue")
            response = TestClient(app).post("/api/workflow/import", files={"file": ("sample.workflow.json", payload, "application/json")})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["workflow_type"], "comfyui_ui_workflow")
        self.assertTrue(response.json()["analysis_only"])
        factory.return_value.queue.assert_not_awaited()

    def test_endpoint_rejects_broken_json_and_non_json_files(self):
        client = TestClient(app)
        broken = client.post("/api/workflow/import", files={"file": ("workflow.json", "{broken", "application/json")})
        wrong_type = client.post("/api/workflow/import", files={"file": ("workflow.txt", "{}", "text/plain")})
        self.assertEqual(broken.status_code, 400)
        self.assertIn("JSONを読み込めません", broken.json()["detail"])
        self.assertEqual(wrong_type.status_code, 415)


if __name__ == "__main__":
    unittest.main()
