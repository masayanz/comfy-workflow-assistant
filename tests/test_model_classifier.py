import unittest

from app.services.model_classifier import classify_model


class ModelClassifierTests(unittest.TestCase):
    def test_families_from_name_and_location(self):
        self.assertEqual(classify_model("flux1-dev.safetensors"), "flux")
        self.assertEqual(classify_model("base.safetensors", "C:/models/sdxl"), "sdxl")
        self.assertEqual(classify_model("ponyDiffusionV6XL_v6StartWithThisOne.safetensors"), "sdxl")
        self.assertEqual(classify_model("v1-5-pruned.ckpt"), "sd15")
        self.assertEqual(classify_model("custom-model.safetensors"), "unknown")


if __name__ == "__main__":
    unittest.main()
