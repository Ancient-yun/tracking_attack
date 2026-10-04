"""CPU-only tests for finalizing the resumed campaign's recorded proofs."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
TEST_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "resume_campaign_control", TEST_DIR.parent / "scripts" / "resume_remaining_campaign.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class FinalizationTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="resume_finalization_mock_", dir=TEST_DIR)).resolve()
        self.check_directory()
        self.campaign = {"status": "complete"}
        self.audit = {"status": "passed", "passed": True,
                      "full_campaign_completed": True, "all_frames_completed": True}
        self.analysis = {
            "status": "complete", "all_raw_frames_campaign_completed": True,
            "validation": {"completed_conditions": 48, "errors": [], "missing_conditions": []},
            "artifact_audit": {"validated_for_scope": True},
            "analyzer_source_sha256": wrapper.ANALYZER_SHA256,
            "loss_structure_document": "/workspace/docs/loss_structure.md",
        }
        self.note = "Mock runtime note: GPU usage included other host applications."
        self.plan = {"status": "pending", "final_report_text_updates": {
            "status": "pending", "runtime_note_markdown": self.note}}
        self.campaign_path = self.directory / "campaign_execution.json"
        self.audit_path = self.directory / "artifact_validation.json"
        self.analysis_path = self.directory / "analysis/study_analysis.json"
        self.plan_path = self.directory / "postprocessing_revision.json"
        self.report_path = self.directory / "analysis/study_report.md"
        write_json(self.campaign_path, self.campaign)
        write_json(self.audit_path, self.audit)
        write_json(self.analysis_path, self.analysis)
        write_json(self.plan_path, self.plan)
        self.report_path.write_text(
            "Mock report [loss structure](/workspace/docs/loss_structure.md)\n", encoding="utf-8")

    def check_directory(self):
        if (self.directory.resolve().parent != TEST_DIR
                or not self.directory.name.startswith("resume_finalization_mock_")):
            raise RuntimeError("Refusing access or cleanup outside the checked mock directory")

    def tearDown(self):
        self.check_directory()
        shutil.rmtree(self.directory)

    def assert_rejected_without_writes(self):
        paths = (self.report_path, self.plan_path, self.campaign_path, self.audit_path, self.analysis_path)
        before = {path: path.read_bytes() for path in paths}
        with self.assertRaises((RuntimeError, KeyError)):
            wrapper.finalize(self.directory)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content, str(path))

    def test_full_48_proofs_finalize_only_report_and_plan(self):
        proof_paths = (self.campaign_path, self.audit_path, self.analysis_path)
        proof_bytes = {path: path.read_bytes() for path in proof_paths}
        before_hash = wrapper.digest(self.report_path)
        result = wrapper.finalize(self.directory)
        report = self.report_path.read_text(encoding="utf-8")
        target = os.path.relpath(Path("docs/loss_structure.md").resolve(), self.report_path.parent).replace(os.sep, "/")
        self.assertIn(f"[loss structure]({target})", report)
        self.assertNotIn("](/workspace/docs/loss_structure.md)", report)
        self.assertEqual(report.count(self.note), 1)
        self.assertEqual(result["completed_conditions"], 48)
        self.assertTrue(result["full_campaign_completed"])
        self.assertTrue(result["artifact_audit_passed"])
        self.assertTrue(result["report_text_updates_only"])
        self.assertEqual(result["final_analyzer_sha256"], wrapper.ANALYZER_SHA256)
        self.assertEqual(result["report_sha256"], wrapper.digest(self.report_path))
        plan = json.loads(self.plan_path.read_text(encoding="utf-8"))
        self.assertEqual(plan["status"], "complete")
        self.assertEqual(plan["final_report_text_updates"]["status"], "complete")
        self.assertEqual(plan["final_report_before_text_updates_sha256"], before_hash)
        self.assertEqual(plan["final_report_sha256"], result["report_sha256"])
        for path, content in proof_bytes.items():
            self.assertEqual(path.read_bytes(), content, str(path))

    def test_failed_campaign_rejected_before_report_modification(self):
        self.campaign["status"] = "failed"
        write_json(self.campaign_path, self.campaign)
        self.assert_rejected_without_writes()

    def test_false_full_audit_flag_rejected_before_report_modification(self):
        self.audit["full_campaign_completed"] = False
        write_json(self.audit_path, self.audit)
        self.assert_rejected_without_writes()

    def test_missing_full_audit_flag_rejected_before_report_modification(self):
        self.audit.pop("full_campaign_completed")
        write_json(self.audit_path, self.audit)
        self.assert_rejected_without_writes()

    def test_false_all_frames_analysis_flag_rejected_before_report_modification(self):
        self.analysis["all_raw_frames_campaign_completed"] = False
        write_json(self.analysis_path, self.analysis)
        self.assert_rejected_without_writes()

    def test_missing_all_frames_analysis_flag_rejected_before_report_modification(self):
        self.analysis.pop("all_raw_frames_campaign_completed")
        write_json(self.analysis_path, self.analysis)
        self.assert_rejected_without_writes()

    def test_wrong_analyzer_sha_rejected_before_report_modification(self):
        self.analysis["analyzer_source_sha256"] = "different-analyzer"
        write_json(self.analysis_path, self.analysis)
        self.assert_rejected_without_writes()

    def test_incomplete_condition_count_rejected_before_report_modification(self):
        self.analysis["validation"]["completed_conditions"] = 47
        write_json(self.analysis_path, self.analysis)
        self.assert_rejected_without_writes()

    def test_missing_validation_rejected_before_report_modification(self):
        self.analysis.pop("validation")
        write_json(self.analysis_path, self.analysis)
        self.assert_rejected_without_writes()


if __name__ == "__main__":
    unittest.main(verbosity=2)
