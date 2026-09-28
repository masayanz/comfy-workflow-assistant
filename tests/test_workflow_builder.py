import unittest

from app.schemas.workflow import WorkflowBuildRequest
from app.services.workflow_builder import build_definition, build_workflow, to_api_prompt, to_ui_workflow, validate_workflow, validate_ui_workflow


class WorkflowBuilderTests(unittest.TestCase):
    def test_builds_api_workflow_with_requested_values(self):
        workflow = build_workflow(WorkflowBuildRequest(
            model="nested/model.safetensors", prompt="portrait", negative_prompt="blur",
            width=832, height=1216, steps=32, cfg=5.5, sampler="dpmpp_2m", seed=123,
        ))
        self.assertEqual(workflow["1"]["inputs"]["ckpt_name"], "nested/model.safetensors")
        self.assertEqual(workflow["2"]["inputs"]["text"], "portrait")
        self.assertEqual(workflow["3"]["inputs"]["text"], "blur")
        self.assertEqual(workflow["4"]["inputs"]["height"], 1216)
        self.assertEqual(workflow["5"]["inputs"]["seed"], 123)
        self.assertEqual(workflow["5"]["inputs"]["sampler_name"], "dpmpp_2m")

    def test_unknown_family_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "SD1.5"):
            build_workflow(WorkflowBuildRequest(model="model.safetensors", prompt="x"), "unknown")

    def test_sd15_uses_sd15_template_and_recommended_defaults(self):
        workflow = build_workflow(WorkflowBuildRequest(
            model="v1-5-pruned.safetensors", prompt="portrait", width=512, height=512, steps=25, cfg=7.0,
        ), "sd15")
        self.assertEqual(workflow["1"]["inputs"]["ckpt_name"], "v1-5-pruned.safetensors")
        self.assertEqual(workflow["4"]["inputs"]["width"], 512)
        self.assertEqual(workflow["5"]["inputs"]["steps"], 25)
        self.assertEqual(workflow["5"]["inputs"]["cfg"], 7.0)

    def test_optional_lora_is_connected_to_sampler_and_encoders(self):
        workflow = build_workflow(WorkflowBuildRequest(model="base.safetensors", prompt="portrait", lora="style.safetensors", lora_weight=0.7))
        self.assertEqual(workflow["8"]["class_type"], "LoraLoader")
        self.assertEqual(workflow["8"]["inputs"]["strength_model"], 0.7)
        self.assertEqual(workflow["5"]["inputs"]["model"], ["8", 0])
        self.assertEqual(workflow["2"]["inputs"]["clip"], ["8", 1])

    def test_validator_rejects_missing_connection_target(self):
        import json
        from app.services.workflow_builder import ROOT
        workflow = json.loads((ROOT / "workflows" / "sdxl" / "txt2img.json").read_text(encoding="utf-8"))
        workflow["2"]["inputs"]["clip"] = ["missing", 0]
        with self.assertRaisesRegex(ValueError, "存在しない接続先"):
            validate_workflow(workflow)

    def test_ui_workflow_contains_canvas_layout_links_and_generation_values(self):
        request = WorkflowBuildRequest(
            model="pony.safetensors", prompt="portrait", negative_prompt="blur", width=832, height=1216,
            steps=31, cfg=5.25, sampler="dpmpp_2m", seed=42,
        )
        definition = build_definition(request, "sdxl")
        api_prompt = to_api_prompt(definition)
        ui_workflow = to_ui_workflow(definition)

        self.assertEqual(ui_workflow["version"], 0.4)
        self.assertEqual({node["id"] for node in ui_workflow["nodes"]}, {1, 2, 3, 4, 5, 6, 7})
        self.assertEqual(ui_workflow["last_node_id"], 7)
        self.assertEqual(ui_workflow["last_link_id"], len(ui_workflow["links"]))
        self.assertEqual(ui_workflow["nodes"][0]["widgets_values"], ["pony.safetensors"])
        self.assertEqual(ui_workflow["nodes"][1]["widgets_values"], ["portrait"])
        self.assertEqual(ui_workflow["nodes"][2]["widgets_values"], ["blur"])
        self.assertEqual(ui_workflow["nodes"][3]["widgets_values"], [832, 1216, 1])
        self.assertEqual(ui_workflow["nodes"][4]["widgets_values"], [42, "fixed", 31, 5.25, "dpmpp_2m", "normal", 1.0])
        self.assertEqual(api_prompt["1"]["inputs"]["ckpt_name"], ui_workflow["nodes"][0]["widgets_values"][0])
        self.assertEqual(api_prompt["4"]["inputs"]["width"], ui_workflow["nodes"][3]["widgets_values"][0])
        self.assertEqual(api_prompt["5"]["inputs"]["seed"], ui_workflow["nodes"][4]["widgets_values"][0])
        validate_ui_workflow(ui_workflow)

    def test_ui_workflow_lora_links_model_and_clip_before_sampling(self):
        definition = build_definition(WorkflowBuildRequest(
            model="pony.safetensors", prompt="portrait", lora="style.safetensors", lora_weight=0.65,
        ), "sdxl")
        workflow = to_ui_workflow(definition)
        nodes = {node["id"]: node for node in workflow["nodes"]}

        self.assertEqual(nodes[8]["type"], "LoraLoader")
        self.assertEqual(nodes[8]["widgets_values"], ["style.safetensors", 0.65, 0.65])
        self.assertLess(nodes[8]["order"], nodes[2]["order"])
        self.assertLess(nodes[8]["order"], nodes[5]["order"])
        self.assertEqual(nodes[5]["inputs"][0]["link"], next(link[0] for link in workflow["links"] if link[1:5] == [8, 0, 5, 0]))
        self.assertEqual(nodes[2]["inputs"][0]["link"], next(link[0] for link in workflow["links"] if link[1:5] == [8, 1, 2, 0]))
        validate_ui_workflow(workflow)

    def test_ui_workflow_is_limited_to_sdxl_mvp(self):
        definition = build_definition(WorkflowBuildRequest(model="v1-5.safetensors", prompt="portrait"), "sd15")
        with self.assertRaisesRegex(ValueError, "現在SDXL"):
            to_ui_workflow(definition)


if __name__ == "__main__":
    unittest.main()
