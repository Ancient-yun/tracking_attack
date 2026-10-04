"""CPU-only tests for the external requested-stop control wrapper."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.dont_write_bytecode = True
RUN_DIR = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "requested_stop_control", RUN_DIR.parent / "scripts" / "stop_after_tracking3d.py")
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="stop_wrapper_mock_", dir=RUN_DIR)).resolve()
        if self.directory.parent != RUN_DIR or not self.directory.name.startswith("stop_wrapper_mock_"):
            raise RuntimeError("Mock test path escaped the run directory")
        self.events = []
        self.writes = []
        self.summary_calls = []
        self.summary = {"complete": False, "completed_conditions": 24, "expected_conditions": 48}
        self.entries = [{"dataset": "po_mini" if i % 2 == 0 else "ds_mini", "sequence": f"clip_{i}"}
                        for i in range(8)]
        self.meta = {"signature": "frozen-signature", "manifest": {"entries": self.entries},
                     "config": {"num_frames": 128, "objectives": ["tracking_mse", "tracking_3d",
                                 "reconstruction_3d", "confidence", "joint_training"]}}
        write_json(self.directory / "run.json", self.meta)
        write_json(self.directory / "pause_request.json", {"user_requested": True, "status": "requested"})
        self.record = {"signature": "frozen-signature", "tracking_metrics": {"num_frames": 128},
                       "reconstruction_metrics": {"num_frames": 128},
                       "saved_input_verification": {"passed": True}, "steps": 20}
        for objective in ("clean", "tracking_mse", "tracking_3d"):
            for entry in self.entries:
                record = copy.deepcopy(self.record)
                if objective == "clean":
                    record["steps"] = 0
                write_json(self.path(objective, entry), record)
        self.study = SimpleNamespace(event=self.event, summarize_component_study=self.summarize,
                                     utc_now=lambda: "mock-utc", write_json=self.write)
        self.hook = wrapper.install_boundary_stop(self.study, self.directory)

    def tearDown(self):
        if self.directory.parent != RUN_DIR or not self.directory.name.startswith("stop_wrapper_mock_"):
            raise RuntimeError("Refusing cleanup outside the checked mock directory")
        shutil.rmtree(self.directory)

    def path(self, objective, entry=None):
        entry = entry or self.entries[0]
        return self.directory / "conditions" / objective / entry["dataset"] / entry["sequence"] / "result.json"

    def event(self, directory, kind, **values):
        self.events.append((kind, values))

    def summarize(self, directory, **kwargs):
        self.summary_calls.append((directory, kwargs))
        return self.summary

    def write(self, path, value):
        self.writes.append((path, copy.deepcopy(value)))
        write_json(path, value)

    def complete(self):
        return self.hook(self.directory, "objective_completed", objective="tracking_3d", wall_seconds=123.)

    def assert_failed_pause(self):
        with self.assertRaises(SystemExit) as error:
            self.complete()
        self.assertNotEqual(error.exception.code, 0)
        self.assertFalse((self.directory / "requested_pause.json").exists())
        self.assertFalse(any(kind == "user_requested_pause" for kind, _ in self.events))

    def test_forbidden_objectives_blocked_before_event_publication(self):
        for objective in ("reconstruction_3d", "confidence", "joint_training", "unknown", None):
            with self.subTest(objective=objective):
                before = len(self.events)
                with self.assertRaises(SystemExit) as error:
                    self.hook(self.directory, "objective_started", objective=objective)
                self.assertNotEqual(error.exception.code, 0)
                self.assertEqual(len(self.events), before)

    def test_allowed_objectives_and_iteration_events_preserved(self):
        for objective in ("tracking_mse", "tracking_3d"):
            self.hook(self.directory, "objective_started", objective=objective)
            self.hook(self.directory, "iteration", objective=objective, step=10, loss=2.5)
        self.hook(self.directory, "objective_completed", objective="tracking_mse")
        self.assertEqual(len(self.events), 5)
        self.assertFalse(self.summary_calls)
        self.assertFalse(self.writes)

    def test_tracking_3d_valid_boundary_exits_zero_and_records_partial_24_of_48(self):
        with self.assertRaises(SystemExit) as error:
            self.complete()
        self.assertEqual(error.exception.code, 0)
        self.assertEqual(len(self.summary_calls), 1)
        self.assertEqual(self.summary_calls[0][1], {"verify": False})
        self.assertEqual([kind for kind, _ in self.events], ["objective_completed", "user_requested_pause"])
        marker = json.loads((self.directory / "requested_pause.json").read_text(encoding="utf-8"))
        self.assertEqual(marker["status"], "paused")
        self.assertEqual(marker["completed_conditions"], 24)
        self.assertEqual(marker["original_expected_conditions"], 48)
        self.assertFalse(marker["full_campaign_completed"])
        self.assertTrue(marker["boundary_verified"])
        self.assertEqual(set(marker["not_started_objectives"]),
                         {"reconstruction_3d", "confidence", "joint_training"})
        self.assertIn("not the full 48-condition independent audit", marker["verification_scope"])
        self.assertFalse(self.events[-1][1]["next_objectives_started"])

    def test_bad_signature_rejected(self):
        record = copy.deepcopy(self.record)
        record["signature"] = "changed"
        write_json(self.path("tracking_3d"), record)
        self.assert_failed_pause()

    def test_bad_attack_step_count_rejected(self):
        record = copy.deepcopy(self.record)
        record["steps"] = 19
        write_json(self.path("tracking_3d"), record)
        self.assert_failed_pause()

    def test_bad_frame_count_rejected(self):
        for metrics in ("tracking_metrics", "reconstruction_metrics"):
            with self.subTest(metrics=metrics):
                record = copy.deepcopy(self.record)
                record[metrics]["num_frames"] = 64
                write_json(self.path("tracking_3d"), record)
                self.assert_failed_pause()
        write_json(self.path("tracking_3d"), self.record)

    def test_failed_saved_rgb_replay_rejected(self):
        record = copy.deepcopy(self.record)
        record["saved_input_verification"]["passed"] = False
        write_json(self.path("tracking_3d"), record)
        self.assert_failed_pause()

    def test_future_objective_artifacts_rejected(self):
        (self.directory / "conditions" / "reconstruction_3d").mkdir()
        self.assert_failed_pause()

    def test_incomplete_or_full_summary_rejected(self):
        for summary in ({"complete": False, "completed_conditions": 23},
                        {"complete": True, "completed_conditions": 24}):
            with self.subTest(summary=summary):
                self.summary = summary
                self.assert_failed_pause()

    def test_frozen_roster_or_frame_count_mismatch_rejected(self):
        for mutation in ("roster", "frames"):
            with self.subTest(mutation=mutation):
                meta = copy.deepcopy(self.meta)
                if mutation == "roster":
                    meta["manifest"]["entries"] = self.entries[:-1]
                else:
                    meta["config"]["num_frames"] = 64
                write_json(self.directory / "run.json", meta)
                self.assert_failed_pause()


if __name__ == "__main__":
    unittest.main(verbosity=2)
