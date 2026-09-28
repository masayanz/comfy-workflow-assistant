import unittest

from app.services.workflow_builder import validate_workflow


class WorkflowValidatorTests(unittest.TestCase):
    def test_accepts_template(self):
        from app.services.workflow_builder import ROOT
        import json
        workflow = json.loads((ROOT / "workflows" / "sdxl" / "txt2img.json").read_text(encoding="utf-8"))
        validate_workflow(workflow)

    def test_rejects_malformed_node(self):
        from app.services.workflow_builder import ROOT
        import json
        workflow = json.loads((ROOT / "workflows" / "sdxl" / "txt2img.json").read_text(encoding="utf-8"))
        workflow["1"] = {"inputs": {}}
        with self.assertRaisesRegex(ValueError, "形式が正しくありません"):
            validate_workflow(workflow)


if __name__ == "__main__":
    unittest.main()
