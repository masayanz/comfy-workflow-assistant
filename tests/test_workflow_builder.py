import unittest

from app.schemas.workflow import WorkflowBuildRequest
from app.services.workflow_builder import Connection, build_definition, build_workflow, to_api_prompt, to_ui_workflow, validate_definition, validate_workflow, validate_ui_workflow


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

    def test_unknown_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "モデルProfile"):
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

    def test_ui_workflow_supports_sd15_with_profile_defaults(self):
        definition = build_definition(WorkflowBuildRequest(model="v1-5.safetensors", prompt="portrait"), "sd15")
        api_prompt = to_api_prompt(definition)
        workflow = to_ui_workflow(definition)
        nodes = {node["id"]: node for node in workflow["nodes"]}
        self.assertEqual(nodes[4]["widgets_values"], [512, 512, 1])
        self.assertEqual(nodes[5]["widgets_values"], [api_prompt["5"]["inputs"]["seed"], "fixed", 25, 7.0, "euler", "normal", 1.0])
        self.assertEqual(api_prompt["5"]["inputs"]["steps"], 25)
        validate_ui_workflow(workflow)

    def test_profile_defaults_are_used_for_omitted_api_settings(self):
        workflow = build_workflow(WorkflowBuildRequest(model="v1-5.safetensors", prompt="portrait"), "sd15")
        self.assertEqual(workflow["4"]["inputs"]["width"], 512)
        self.assertEqual(workflow["4"]["inputs"]["height"], 512)
        self.assertEqual(workflow["5"]["inputs"]["steps"], 25)
        self.assertEqual(workflow["5"]["inputs"]["cfg"], 7.0)
        self.assertEqual(workflow["5"]["inputs"]["scheduler"], "normal")

    def test_sd15_lora_uses_profile_capability_for_api_and_ui_graphs(self):
        request = WorkflowBuildRequest(model="v1-5.safetensors", prompt="portrait", lora="style.safetensors", lora_weight=0.65)
        definition = build_definition(request, "sd15")
        api_prompt = to_api_prompt(definition)
        ui_workflow = to_ui_workflow(definition)
        nodes = {node["id"]: node for node in ui_workflow["nodes"]}
        self.assertEqual(api_prompt["8"]["inputs"]["strength_clip"], 0.65)
        self.assertEqual(nodes[8]["widgets_values"], ["style.safetensors", 0.65, 0.65])
        self.assertEqual(nodes[5]["inputs"][0]["link"], next(link[0] for link in ui_workflow["links"] if link[1:5] == [8, 0, 5, 0]))

    def test_flux_split_model_api_and_ui_workflows_share_settings_and_connections(self):
        request = WorkflowBuildRequest(
            profile_id="flux", diffusion_model="flux1-dev.safetensors", clip_name1="clip_l.safetensors",
            clip_name2="t5xxl_fp8.safetensors", vae_model="ae.safetensors", prompt="a mountain lake",
            width=1024, height=768, steps=20, cfg=1, guidance=3.5, sampler="euler", scheduler="simple", seed=1234,
        )
        definition = build_definition(request, "flux")
        api_prompt = to_api_prompt(definition)
        ui_workflow = to_ui_workflow(definition)
        nodes = {node["id"]: node for node in ui_workflow["nodes"]}

        self.assertEqual(api_prompt["1"]["class_type"], "UNETLoader")
        self.assertEqual(api_prompt["1"]["inputs"]["unet_name"], "flux1-dev.safetensors")
        self.assertEqual(api_prompt["2"]["inputs"]["clip_name1"], "clip_l.safetensors")
        self.assertEqual(api_prompt["2"]["inputs"]["clip_name2"], "t5xxl_fp8.safetensors")
        self.assertEqual(api_prompt["2"]["inputs"]["type"], "flux")
        self.assertEqual(api_prompt["5"]["inputs"]["guidance"], 3.5)
        self.assertEqual(api_prompt["6"]["inputs"], {"width": 1024, "height": 768, "batch_size": 1})
        self.assertEqual(api_prompt["7"]["inputs"]["seed"], 1234)
        self.assertEqual(api_prompt["7"]["inputs"]["scheduler"], "simple")
        self.assertEqual(nodes[1]["widgets_values"], ["flux1-dev.safetensors", "default"])
        self.assertEqual(nodes[2]["widgets_values"], ["clip_l.safetensors", "t5xxl_fp8.safetensors", "flux", "default"])
        self.assertEqual(nodes[4]["widgets_values"], ["a mountain lake"])
        self.assertEqual(nodes[6]["widgets_values"], [1024, 768, 1])
        self.assertEqual(nodes[7]["widgets_values"], [1234, "fixed", 20, 1.0, "euler", "simple", 1.0])
        self.assertEqual(api_prompt["7"]["inputs"]["model"], ["1", 0])
        self.assertEqual(api_prompt["8"]["inputs"]["vae"], ["3", 0])
        validate_ui_workflow(ui_workflow)

    def test_flux_rejects_dimensions_outside_empty_sd3_latent_multiple(self):
        with self.assertRaisesRegex(ValueError, "16の倍数"):
            build_definition(WorkflowBuildRequest(profile_id="flux", prompt="x", width=1000, height=1024), "flux")

    def test_sdxl_img2img_api_and_ui_workflows_use_input_image_and_denoise(self):
        request = WorkflowBuildRequest(
            model="portrait.safetensors", generation_type="img2img", input_image_id="upload-id",
            prompt="soft watercolor", negative_prompt="blurry", steps=20, cfg=6.0,
            sampler="euler", scheduler="normal", seed=987, denoise=0.5,
        )
        definition = build_definition(request, "sdxl", "cwa_test.png")
        api_prompt = to_api_prompt(definition)
        ui = to_ui_workflow(definition)
        nodes = {node["id"]: node for node in ui["nodes"]}
        self.assertEqual(definition.generation_type, "img2img")
        self.assertEqual(definition.input_image, "cwa_test.png")
        self.assertEqual(api_prompt["4"]["class_type"], "LoadImage")
        self.assertEqual(api_prompt["4"]["inputs"]["image"], "cwa_test.png")
        self.assertEqual(api_prompt["5"]["class_type"], "VAEEncode")
        self.assertEqual(api_prompt["5"]["inputs"]["pixels"], ["4", 0])
        self.assertEqual(api_prompt["5"]["inputs"]["vae"], ["1", 2])
        self.assertEqual(api_prompt["6"]["inputs"]["latent_image"], ["5", 0])
        self.assertEqual(api_prompt["6"]["inputs"]["denoise"], 0.5)
        self.assertEqual(api_prompt["2"]["inputs"]["text"], "soft watercolor")
        self.assertEqual(api_prompt["3"]["inputs"]["text"], "blurry")
        self.assertEqual(api_prompt["6"]["inputs"]["seed"], 987)
        self.assertEqual(nodes[4]["widgets_values"], ["cwa_test.png", "image"])
        self.assertEqual(nodes[6]["widgets_values"], [987, "fixed", 20, 6.0, "euler", "normal", 0.5])
        self.assertEqual({node["id"] for node in ui["nodes"]}, set(range(1, 9)))
        validate_ui_workflow(ui)

    def test_sdxl_lora_img2img_routes_checkpoint_model_and_clip_through_lora(self):
        request = WorkflowBuildRequest(
            model="portrait.safetensors", generation_type="img2img", prompt="portrait",
            lora="style.safetensors", lora_weight=0.7, denoise=0.8,
        )
        api = build_workflow(request, "sdxl", "cwa_test.png")
        self.assertEqual(api["9"]["class_type"], "LoraLoader")
        self.assertEqual(api["2"]["inputs"]["clip"], ["9", 1])
        self.assertEqual(api["3"]["inputs"]["clip"], ["9", 1])
        self.assertEqual(api["6"]["inputs"]["model"], ["9", 0])
        self.assertEqual(api["6"]["inputs"]["denoise"], 0.8)
        ui = to_ui_workflow(build_definition(request, "sdxl", "cwa_test.png"))
        self.assertEqual({node["id"] for node in ui["nodes"]}, set(range(1, 10)))
        validate_ui_workflow(ui)

    def test_img2img_requires_a_valid_image_and_supported_profile(self):
        with self.assertRaisesRegex(ValueError, "入力画像が必要"):
            build_definition(WorkflowBuildRequest(model="portrait.safetensors", generation_type="img2img", prompt="x"), "sdxl")
        with self.assertRaisesRegex(ValueError, "img2imgは現在対応していません"):
            build_definition(WorkflowBuildRequest(model="v1-5.safetensors", generation_type="img2img", prompt="x"), "sd15")

    def test_img2img_definition_rejects_wrong_socket_types_and_ui_link_types(self):
        definition = build_definition(
            WorkflowBuildRequest(model="portrait.safetensors", generation_type="img2img", prompt="x"),
            "sdxl", "cwa_test.png",
        )
        definition.nodes[5].inputs["latent_image"] = Connection(1, 2)
        with self.assertRaises(ValueError):
            validate_definition(definition)

        ui = to_ui_workflow(build_definition(
            WorkflowBuildRequest(model="portrait.safetensors", generation_type="img2img", prompt="x"),
            "sdxl", "cwa_test.png",
        ))
        ui["links"][0][5] = "LATENT"
        with self.assertRaises(ValueError):
            validate_ui_workflow(ui)

    def test_upscale_definition_generates_api_and_canvas_graph_without_checkpoint_profile(self):
        request = WorkflowBuildRequest(
            generation_type="upscale", input_image_id="upload-1", upscale_model="4x-UltraSharp.pth",
        )
        definition = build_definition(request, "upscale", "cwa_input.png")
        api = to_api_prompt(definition)
        self.assertEqual(api["1"], {"class_type": "LoadImage", "inputs": {"image": "cwa_input.png"}})
        self.assertEqual(api["2"], {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "4x-UltraSharp.pth"}})
        self.assertEqual(api["3"]["class_type"], "ImageUpscaleWithModel")
        self.assertEqual(api["3"]["inputs"], {"upscale_model": ["2", 0], "image": ["1", 0]})
        self.assertEqual(api["4"], {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": "ComfyWorkflowBuilder_Upscale"}})
        ui = to_ui_workflow(definition)
        nodes = {node["id"]: node for node in ui["nodes"]}
        self.assertEqual(set(nodes), {1, 2, 3, 4})
        self.assertEqual(nodes[1]["widgets_values"], ["cwa_input.png", "image"])
        self.assertEqual(nodes[2]["widgets_values"], ["4x-UltraSharp.pth"])
        self.assertEqual(ui["links"], [
            [1, 2, 0, 3, 0, "UPSCALE_MODEL"],
            [2, 1, 0, 3, 1, "IMAGE"],
            [3, 3, 0, 4, 0, "IMAGE"],
        ])
        validate_ui_workflow(ui)

    def test_upscale_requires_image_and_model_and_rejects_sampler_options(self):
        request = WorkflowBuildRequest(generation_type="upscale", upscale_model="4x.pth")
        with self.assertRaisesRegex(ValueError, "入力画像が必要"):
            build_definition(request, "upscale")
        request = WorkflowBuildRequest(generation_type="upscale", input_image_id="upload-1")
        with self.assertRaisesRegex(ValueError, "Upscale Modelを選択"):
            build_definition(request, "upscale", "cwa_input.png")


if __name__ == "__main__":
    unittest.main()
