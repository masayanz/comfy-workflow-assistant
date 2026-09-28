import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.services.model_profile_service import ModelProfileService


class ModelProfileServiceTests(unittest.TestCase):
    def test_builtin_profiles_include_sd15_and_sdxl_defaults(self):
        profiles = {profile.id: profile for profile in ModelProfileService().list_profiles()}
        self.assertEqual(set(profiles), {"sd15", "sdxl", "flux"})
        self.assertEqual((profiles["sd15"].default_width, profiles["sd15"].default_height), (512, 512))
        self.assertEqual((profiles["sd15"].default_steps, profiles["sd15"].default_cfg), (25, 7.0))
        self.assertEqual((profiles["sdxl"].default_width, profiles["sdxl"].default_height), (1024, 1024))
        self.assertEqual((profiles["sdxl"].default_steps, profiles["sdxl"].default_cfg), (28, 6.0))
        self.assertTrue(profiles["flux"].enabled)
        self.assertEqual((profiles["flux"].default_width, profiles["flux"].default_height), (1024, 1024))
        self.assertEqual((profiles["flux"].default_steps, profiles["flux"].default_cfg), (20, 1.0))
        self.assertEqual(profiles["flux"].default_guidance, 3.5)
        self.assertFalse(profiles["flux"].ui.show_negative_prompt)
        self.assertEqual([component.key for component in profiles["flux"].ui.model_components], ["diffusion_model", "clip_name1", "clip_name2", "vae_model"])

    def test_lookup_returns_profile_and_unknown_profile_is_rejected(self):
        service = ModelProfileService()
        self.assertEqual(service.get_profile("sdxl").name, "Stable Diffusion XL")
        with self.assertRaises(KeyError):
            service.get_profile("unknown")

    def test_invalid_profile_json_reports_filename(self):
        with TemporaryDirectory() as directory:
            Path(directory, "sd15.json").write_text('{"id": "sd15"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "sd15.json"):
                ModelProfileService(directory).list_profiles()

    def test_invalid_capability_relationship_is_rejected(self):
        profile = {
            "id": "sd15", "name": "SD15", "enabled": True,
            "default_width": 512, "default_height": 512, "default_steps": 25,
            "default_cfg": 7, "default_sampler": "euler", "default_scheduler": "normal",
            "supported_generation_types": ["txt2img"], "supports_lora": True,
            "capabilities": {"txt2img": False, "img2img": False, "lora": True, "controlnet": False, "upscale": False},
        }
        with TemporaryDirectory() as directory:
            Path(directory, "sd15.json").write_text(json.dumps(profile), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "txt2img capability"):
                ModelProfileService(directory).list_profiles()


if __name__ == "__main__":
    unittest.main()
