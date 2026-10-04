from pathlib import Path

import numpy as np
import pytest

from st4rtrack_pgd.evaluate_attack import evaluate_tracks
from st4rtrack_pgd.make_manifest import build_manifest


def test_official_metric_perfect_scaled_trajectory_and_empty_dynamic():
    gt = np.array([[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], [[2.0, 1.0, 4.0], [4.0, 2.0, 6.0]]])
    result = evaluate_tracks(gt * 2, gt, np.ones(gt.shape[:2], dtype=bool), np.array([False, False]), [500, 500, 320, 240])
    assert result["apd3d_all"] == 100
    assert result["epe_all_m"] == 0
    assert result["scale_all"] == .5
    assert result["apd3d_dynamic"] is None


def test_evaluator_rejects_nonfinite_instead_of_excluding_it():
    gt = np.ones((2, 2, 3))
    pred = gt.copy()
    pred[0, 0, 0] = np.nan
    with pytest.raises(FloatingPointError):
        evaluate_tracks(pred, gt, np.ones((2, 2), dtype=bool), np.ones(2, dtype=bool), [500, 500, 320, 240])


def test_manifest_selection_stable_and_insufficient_data_explicit(tmp_path: Path):
    directory = tmp_path / "po_mini"
    directory.mkdir()
    for index in range(8):
        (directory / f"clip_{index}.npz").write_bytes(bytes([index]))
    first = build_manifest(tmp_path, ["po_mini"], 3, 123, 64)
    second = build_manifest(tmp_path, ["po_mini"], 3, 123, 64)
    assert first["entries"] == second["entries"]
    assert first["paper_sequence_identity_verified"] is False
    with pytest.raises(ValueError, match="only 8 available"):
        build_manifest(tmp_path, ["po_mini"], 50, 123, 64)
