import unittest
from pathlib import Path

from app.services.model_scanner import scan_models


class ModelScannerTests(unittest.TestCase):
    def test_scans_model_categories_and_nested_names(self):
        root = Path(__file__).parent / "fixtures" / "comfy"
        records = scan_models(root)
        self.assertEqual(len(records), 6)
        nested = next(item for item in records if item["name"] == "base.safetensors")
        self.assertEqual(nested["comfy_name"], "sdxl/base.safetensors")
        self.assertEqual(nested["family"], "sdxl")
        self.assertGreater(nested["size"], 0)

    def test_missing_root_returns_empty_list(self):
        self.assertEqual(scan_models(None), [])


if __name__ == "__main__":
    unittest.main()
