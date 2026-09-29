import unittest

from app.services.model_metadata import classify_asset, classify_compatibility, metadata_role


class ModelMetadataTests(unittest.TestCase):
    def test_metadata_classifies_sd15_sdxl_pony_and_flux(self):
        cases = [
            ({"modelspec.architecture": "stable-diffusion-v1-5/lora"}, "sd15", None),
            ({"ss_base_model_version": "sdxl"}, "sdxl", None),
            ({"modelspec.title": "Pony Diffusion v6"}, "sdxl", "pony"),
            ({"modelspec.architecture": "flux-dev/lora"}, "flux", None),
        ]
        for metadata, family, variant in cases:
            with self.subTest(metadata=metadata):
                result = classify_asset("unhelpful-name.safetensors", metadata)
                self.assertEqual((result["family"], result["variant"]), (family, variant))
                self.assertEqual(result["confidence"], "high")
                self.assertEqual(result["source"], "metadata")

    def test_filename_classification_is_low_confidence(self):
        result = classify_asset("PonyDiffusion_v6.safetensors")
        self.assertEqual(result["family"], "sdxl")
        self.assertEqual(result["variant"], "pony")
        self.assertEqual(result["confidence"], "low")
        self.assertEqual(result["source"], "filename")

    def test_metadata_family_wins_while_filename_only_pony_variant_stays_low_confidence(self):
        result = classify_asset("ponyDiffusionV6XL.safetensors", {"modelspec.architecture": "stable-diffusion-xl-v1-base"})
        self.assertEqual(result["family"], "sdxl")
        self.assertEqual(result["confidence"], "high")
        self.assertEqual(result["variant"], "pony")
        self.assertEqual(result["variant_confidence"], "low")

    def test_v15_filename_hint_is_low_confidence_sd15(self):
        result = classify_asset("JapaneseDollLikeness_v15.safetensors")
        self.assertEqual(result["family"], "sd15")
        self.assertEqual(result["confidence"], "low")

    def test_unknown_metadata_does_not_fail_and_lora_role_is_detected(self):
        metadata = {"ss_network_module": "networks.lora", "custom_field": "opaque"}
        self.assertEqual(metadata_role(metadata), "lora")
        self.assertEqual(classify_asset("opaque.safetensors", metadata)["family"], "unknown")

    def test_clear_mismatch_blocks_but_unknown_and_pony_mix_warn(self):
        checkpoint = {"name": "base", "classification": {"family": "sd15", "confidence": "high", "source": "metadata"}}
        lora = {"name": "style", "classification": {"family": "sdxl", "confidence": "high", "source": "metadata"}}
        self.assertEqual(classify_compatibility(checkpoint, lora)["status"], "incompatible")
        unknown = {"classification": {"family": "unknown", "confidence": "unknown", "source": "unknown"}}
        self.assertEqual(classify_compatibility(checkpoint, unknown)["status"], "unknown")
        pony = {"classification": {"family": "sdxl", "variant": "pony", "confidence": "high", "source": "metadata"}}
        generic_sdxl = {"classification": {"family": "sdxl", "confidence": "high", "source": "metadata"}}
        self.assertEqual(classify_compatibility(pony, generic_sdxl)["status"], "compatible_with_warning")

    def test_matching_metadata_is_compatible_and_low_confidence_mismatch_only_warns(self):
        checkpoint = {"classification": {"family": "sdxl", "confidence": "high", "source": "metadata"}}
        matching = {"classification": {"family": "sdxl", "confidence": "high", "source": "metadata"}}
        weak_mismatch = {"classification": {"family": "sd15", "confidence": "low", "source": "filename"}}
        self.assertEqual(classify_compatibility(checkpoint, matching)["status"], "compatible")
        self.assertEqual(classify_compatibility(checkpoint, weak_mismatch)["status"], "compatible_with_warning")


if __name__ == "__main__":
    unittest.main()
