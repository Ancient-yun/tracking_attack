"""CPU integration tests for run artifacts and resume, using a synthetic forward.

The upstream metric implementation is real. Only checkpoint provenance, the
dataset decoder, and the large neural model are mocked. No GPU is required.
"""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from st4rtrack_pgd import run_attack as runner
from st4rtrack_pgd.common import VENDOR_ROOT, sha256
from st4rtrack_pgd.evaluate_attack import evaluate_tracks, summarize_run
from st4rtrack_pgd.pgd_video import predict_tracks
from st4rtrack_pgd.worldtrack_adapter import WorldTrackSequence


class SyntheticForward:
    """Small deterministic two-view predictor with fixed query identities."""

    def __init__(self, model, query_xy, *, vendor_root=None, checkpoint=None):
        self.model = model
        self.query_xy = query_xy
        self.vendor_root = Path(vendor_root or VENDOR_ROOT).resolve()
        self.checkpoint = checkpoint
        self.metadata = {"model_class": "SyntheticForward", "vendor_root": str(self.vendor_root),
                         "checkpoint": checkpoint, "frozen_weights": True, "tta": False}

    @classmethod
    def from_checkpoint(cls, checkpoint, query_xy, *, vendor_root=None, device="cpu"):
        if str(device) != "cpu":
            raise AssertionError("Synthetic integration test must not use GPU")
        return cls(torch.nn.Identity(), query_xy, vendor_root=vendor_root, checkpoint=str(checkpoint))

    def pair_fn(self, rgb, t):
        first = rgb[0].mean(dim=(-1, -2))
        other = rgb[t].mean(dim=(-1, -2))
        base = rgb.new_tensor([[0.1, 0.2, 1.1], [0.2, 0.1, 2.2],
                               [0.3, 0.1, 3.3], [0.1, 0.3, 4.4]])
        coeff = rgb.new_tensor([0.4, 0.7, 1., 1.2])[:, None]
        return base + coeff * (0.3 * first + 0.5 * other + first * other * 0.1)

    def official_predict(self, sequence):
        # Ensures the runner preserves an explicitly selected source checkout
        # when it reuses the model for a second sequence.
        if self.vendor_root != Path(sequence.metadata["vendor_root"]).resolve():
            raise AssertionError("Selected vendor root was lost during model reuse")
        return predict_tracks(self.pair_fn, sequence.rgb)


def synthetic_sequence(vendor_root=VENDOR_ROOT):
    rgb = torch.linspace(0.12, 0.84, 2 * 3 * 2 * 3).reshape(2, 3, 2, 3)
    xy = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=torch.float64)
    predictor = SyntheticForward(torch.nn.Identity(), xy)
    clean_tracks = predict_tracks(predictor.pair_fn, rgb)
    gt = clean_tracks * 1.2 + torch.tensor([0.15, -0.13, 0.04])
    gt[1, :, 0] += 0.03
    return WorldTrackSequence(
        rgb=rgb, gt_tracks=gt, valid=torch.ones((2, 4), dtype=torch.bool),
        dynamic=torch.tensor([True, False, True, False]), query_xy=xy,
        metadata={"vendor_root": str(Path(vendor_root).resolve()), "intrinsics": [256., 256., 128., 128.],
                  "num_frames": 2, "num_queries": 4})


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.run_dir = self.root / "run"
        checkpoint = self.root / "checkpoint.pth"
        checkpoint.write_bytes(b"mocked test checkpoint")
        self.config = {"checkpoint": str(checkpoint), "num_frames": 2, "image_size": 512,
                       "device": "cpu", "seed": 271, "methods": ["clean", "noise", "fgsm", "pgd"],
                       "epsilons_255": [4], "steps": 2, "restarts": 1,
                       "step_size_fraction": 0.25, "gradient_mode": "recompute",
                       "verify_official_clean": True, "save_inputs": True, "verify_saved_inputs": True}
        source = self.root / "source.npz"
        source.write_bytes(b"mocked test sequence")
        self.manifest = {"schema_version": 1, "num_frames": 2, "entries": [
            {"dataset": "po_mini", "sequence": "clip_a", "path": str(source), "sha256": sha256(source)}]}
        self.patches = [patch.object(runner, "provenance", return_value={"test_source": "fixed"}),
                        patch.object(runner, "St4RTrackForward", SyntheticForward),
                        patch.object(runner, "load_sequence", return_value=synthetic_sequence())]
        self.mocks = [item.start() for item in self.patches]

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.directory.cleanup()

    def run_test_experiment(self, *, resume=False, vendor_root=VENDOR_ROOT):
        return runner.run_experiment(self.config, self.manifest, self.run_dir, resume=resume,
                                     fail_fast=True, vendor_root=vendor_root)

    def test_end_to_end_artifact_metrics_reloads_and_resume(self):
        summary = self.run_test_experiment()
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["completed_conditions"], 4)
        self.assertEqual(summary["expected_conditions"], 4)
        self.assertEqual(summary["failed_this_invocation"], 0)
        records = list(self.run_dir.glob("sequences/*/*/*/result.json"))
        before = {path: path.read_bytes() for path in records}
        for path in records:
            record = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(record["saved_input_verification"]["passed"])
            self.assertIsNone(record["peak_vram_bytes"])
            for name, digest in record["artifact_sha256"].items():
                self.assertEqual(sha256(path.parent / name), digest)
            rgb = np.load(path.parent / "rgb_float32.npy", allow_pickle=False)
            delta = np.load(path.parent / "delta_float32.npy", allow_pickle=False)
            self.assertEqual(rgb.dtype, np.float32)
            self.assertLessEqual(float(np.max(np.abs(delta))), record["epsilon_255"] / 255 + 1e-7)
            self.assertTrue(np.logical_and(rgb >= 0, rgb <= 1).all())
            norms = json.loads((path.parent / "perturbation_norms.json").read_text())
            self.assertEqual(len(norms), 2)
            self.assertTrue(all(item["linf"] <= record["epsilon_255"] / 255 + 1e-7 for item in norms))
            if record["method"] == "pgd":
                history = json.loads((path.parent / "history.json").read_text())
                gradients = [item["gradient_l2_per_frame"] for item in history if "gradient_l2_per_frame" in item]
                self.assertEqual(len(gradients), 2)
                self.assertTrue(all(len(item) == 2 and min(item) > 0 for item in gradients))
        with (self.run_dir / "sequence_metrics.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 4)
        self.assertTrue(all("apd_drop_all_pp" in row and "epe_increase_all_m" in row for row in rows))
        with patch.object(runner, "attack_video", side_effect=AssertionError("resume must not rerun completed attacks")):
            resumed = self.run_test_experiment(resume=True)
        self.assertTrue(resumed["complete"])
        self.assertEqual(before, {path: path.read_bytes() for path in records})

    def test_resume_refuses_config_changes_or_tampered_artifacts(self):
        self.run_test_experiment()
        self.config["steps"] = 3
        with self.assertRaisesRegex(ValueError, "Resume refused"):
            self.run_test_experiment(resume=True)
        self.config["steps"] = 2
        artifact = self.run_dir / "sequences" / "po_mini" / "clip_a" / "clean" / "rgb_float32.npy"
        with artifact.open("ab") as output:
            output.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "artifact missing or altered"):
            self.run_test_experiment(resume=True)

    def test_dataset_hash_failure_is_recorded_and_run_incomplete(self):
        Path(self.manifest["entries"][0]["path"]).write_bytes(b"changed source")
        summary = runner.run_experiment(self.config, self.manifest, self.run_dir, fail_fast=False)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["completed_conditions"], 0)
        self.assertEqual(summary["failed_this_invocation"], 1)
        failures = (self.run_dir / "failures.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(failures), 1)
        failure = json.loads(failures[0])
        self.assertEqual(failure["stage"], "load_or_clean_parity")
        self.assertIn("Dataset hash mismatch", failure["error"])

    def test_attack_failure_is_recorded_without_silent_exclusion(self):
        actual_attack = runner.attack_video

        def failing_pgd(pair, rgb, gt, valid, config, **kwargs):
            if config.method == "pgd":
                raise FloatingPointError("synthetic nonfinite gradient")
            return actual_attack(pair, rgb, gt, valid, config, **kwargs)

        with patch.object(runner, "attack_video", side_effect=failing_pgd):
            summary = runner.run_experiment(self.config, self.manifest, self.run_dir, fail_fast=False)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["completed_conditions"], 3)
        self.assertEqual(summary["expected_conditions"], 4)
        failure = json.loads((self.run_dir / "failures.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(failure["method"], "pgd")
        self.assertIn("nonfinite gradient", failure["error"])

    def test_selected_vendor_root_is_preserved_across_two_sequences(self):
        vendor = self.root / "alternate_official_checkout"
        (vendor / "dust3r").mkdir(parents=True)
        shutil.copyfile(VENDOR_ROOT / "dust3r" / "track_eval_util.py", vendor / "dust3r" / "track_eval_util.py")
        self.mocks[2].return_value = synthetic_sequence(vendor)
        second = dict(self.manifest["entries"][0], sequence="clip_b")
        self.manifest["entries"].append(second)
        summary = self.run_test_experiment(vendor_root=vendor)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["completed_conditions"], 8)

    def test_duplicate_manifest_identity_is_rejected_before_work(self):
        self.manifest["entries"].append(dict(self.manifest["entries"][0]))
        with self.assertRaises(ValueError):
            self.run_test_experiment()
        self.mocks[2].assert_not_called()

    def test_stale_result_signature_is_rejected_on_re_evaluation(self):
        self.run_test_experiment()
        path = self.run_dir / "sequences" / "po_mini" / "clip_a" / "clean" / "result.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record["signature"] = "different_config_or_source"
        path.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaises((ValueError, RuntimeError)):
            summarize_run(self.run_dir)

    def test_dynamic_metric_reestimates_scale_on_its_fixed_gt_subset(self):
        gt = np.array([[[0., 0., 1.], [0., 0., 3.]],
                       [[0., 0., 2.], [0., 0., 4.]]], dtype=np.float32)
        pred = gt.copy()
        pred[:, 0] *= 2
        pred[:, 1] *= 6
        metrics = evaluate_tracks(pred, gt, np.ones((2, 2), dtype=bool),
                                  np.array([True, False]), [256., 256., 128., 128.])
        self.assertEqual(metrics["scale_dynamic"], 0.5)
        self.assertEqual(metrics["epe_dynamic_m"], 0)
        self.assertEqual(metrics["apd3d_dynamic"], 100)
        self.assertNotEqual(metrics["scale_all"], metrics["scale_dynamic"])
        self.assertGreater(metrics["epe_all_m"], 0)


if __name__ == "__main__":
    unittest.main()
