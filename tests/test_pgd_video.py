"""CPU checks for the objective, chain-rule replay, and attack artifacts.

Run with ``python -m unittest discover -s tests``; no checkpoint/data needed.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from st4rtrack_pgd.pgd_video import (
    AttackConfig,
    attack_video,
    loss_and_gradient,
    predict_tracks,
    project_rgb,
)
from st4rtrack_pgd.tracking_loss import aligned_tracking_loss, global_median_scale


class ToyPair:
    """A deterministic nonlinear point predictor with both view contributions."""

    @staticmethod
    def points(first, other, t):
        a = first.mean(dim=(-1, -2))
        b = other.mean(dim=(-1, -2))
        base = a.new_tensor([[0.1, 0.2, 1.0], [0.3, 0.1, 2.1],
                             [0.2, 0.4, 3.3], [0.4, 0.2, 4.8]])
        w = a.new_tensor([[0.30, 0.11, 0.18], [0.17, 0.21, 0.12], [0.19, 0.13, 0.16]])
        v = a.new_tensor([[0.13, 0.23, 0.15], [0.25, 0.14, 0.10], [0.10, 0.20, 0.22]])
        coeff = a.new_tensor([0.4, 0.7, 0.9, 1.2])[:, None]
        return base + coeff * (a @ w + b @ v + 0.2 * a * b) + t * 0.03

    def __call__(self, rgb, t):
        return self.points(rgb[0], rgb[t], t)


def fixture(dtype=torch.float64):
    rgb = torch.linspace(0.12, 0.83, 3 * 3 * 2 * 2, dtype=dtype).reshape(3, 3, 2, 2)
    pair = ToyPair()
    tracks = predict_tracks(pair, rgb)
    drift = rgb.new_tensor([0.11, -0.09, 0.07])[None, None, :]
    temporal = torch.arange(3, dtype=dtype)[:, None, None] * rgb.new_tensor([0.02, 0.01, -0.03])
    gt = tracks * 1.3 + drift + temporal
    valid = torch.ones(gt.shape[:-1], dtype=torch.bool)
    return pair, rgb, gt, valid


class TrackingLossTests(unittest.TestCase):
    def test_even_median_matches_upstream_numpy_convention(self):
        pred = torch.tensor([[[0., 0., 1.], [0., 0., 2.], [0., 0., 4.], [0., 0., 8.]]],
                            dtype=torch.float64)
        gt = torch.tensor([[[0., 0., 2.], [0., 0., 3.], [0., 0., 5.], [0., 0., 9.]]],
                          dtype=torch.float64)
        valid = torch.ones((1, 4), dtype=torch.bool)
        expected = np.median(np.linalg.norm(gt.numpy(), axis=-1)) / np.median(np.linalg.norm(pred.numpy(), axis=-1))
        self.assertAlmostEqual(float(global_median_scale(pred, gt, valid)), expected)
        self.assertAlmostEqual(expected, 4.0 / 3.0)

    def test_scale_is_included_in_gradient(self):
        pair, rgb, gt, valid = fixture()
        tracks = predict_tracks(pair, rgb).requires_grad_()
        loss = aligned_tracking_loss(tracks, gt, valid)
        actual = torch.autograd.grad(loss, tracks)[0]
        tracks_2 = tracks.detach().requires_grad_()
        scale = global_median_scale(tracks_2, gt, valid).detach()
        wrong_loss = (tracks_2[valid] * scale - gt[valid]).square().sum(-1).mean()
        omitted = torch.autograd.grad(wrong_loss, tracks_2)[0]
        self.assertGreater(float((actual - omitted).abs().max()), 1e-3)

    def test_fixed_mask_and_invalid_gt_padding(self):
        pair, rgb, gt, valid = fixture()
        tracks = predict_tracks(pair, rgb)
        valid[1, 2] = False
        gt[1, 2] = float("nan")
        loss = aligned_tracking_loss(tracks, gt, valid)
        selected_pred = tracks[valid].reshape(1, -1, 3)
        selected_gt = gt[valid].reshape(1, -1, 3)
        expected = aligned_tracking_loss(selected_pred, selected_gt, torch.ones(selected_pred.shape[:-1], dtype=torch.bool))
        torch.testing.assert_close(loss, expected)
        with self.assertRaisesRegex(ValueError, "selects no points"):
            aligned_tracking_loss(tracks, gt, torch.zeros_like(valid))
        tracks[1, 2] = float("nan")
        with self.assertRaisesRegex(FloatingPointError, "NaN or Inf"):
            aligned_tracking_loss(tracks, gt, valid)


class ReplayGradientTests(unittest.TestCase):
    def test_full_and_recompute_match_on_every_frame(self):
        pair, rgb, gt, valid = fixture()
        full = loss_and_gradient(pair, rgb, gt, valid, mode="full")
        replay = loss_and_gradient(pair, rgb, gt, valid, mode="recompute")
        torch.testing.assert_close(full.tracks, replay.tracks, rtol=0, atol=0)
        torch.testing.assert_close(full.gradient, replay.gradient, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(full.loss, replay.loss, places=14)
        self.assertAlmostEqual(full.scale, replay.scale, places=14)
        self.assertTrue(bool((replay.gradient.flatten(1).norm(dim=1) > 0).all()))
        self.assertFalse(rgb.requires_grad)

    def test_finite_differences_through_common_scale(self):
        pair, rgb, gt, valid = fixture()
        actual = loss_and_gradient(pair, rgb, gt, valid).gradient
        step = 1e-6
        for t, c, y, x in [(0, 0, 0, 0), (0, 2, 1, 1), (1, 1, 1, 0), (2, 2, 0, 1)]:
            plus, minus = rgb.clone(), rgb.clone()
            plus[t, c, y, x] += step
            minus[t, c, y, x] -= step
            numeric = (aligned_tracking_loss(predict_tracks(pair, plus), gt, valid)
                       - aligned_tracking_loss(predict_tracks(pair, minus), gt, valid)) / (2 * step)
            self.assertAlmostEqual(float(actual[t, c, y, x]), float(numeric), delta=2e-8)

    def test_frame_zero_sums_both_self_pair_views_and_all_other_pairs(self):
        pair, rgb, gt, valid = fixture()
        compact = predict_tracks(pair, rgb).requires_grad_()
        upstream = torch.autograd.grad(aligned_tracking_loss(compact, gt, valid), compact)[0]
        expected = torch.zeros_like(rgb)
        self_pair_a, self_pair_b = None, None
        for t in range(rgb.shape[0]):
            # Independent leaves make both roles in (0,0) explicit.
            first = rgb[0].clone().requires_grad_()
            other = rgb[t].clone().requires_grad_()
            points = pair.points(first, other, t)
            grad_first, grad_other = torch.autograd.grad(points, (first, other), upstream[t])
            expected[0] += grad_first
            expected[t] += grad_other
            if t == 0:
                self_pair_a, self_pair_b = grad_first, grad_other
        actual = loss_and_gradient(pair, rgb, gt, valid).gradient
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        self.assertGreater(float(self_pair_a.abs().max()), 0)
        self.assertGreater(float(self_pair_b.abs().max()), 0)
        # An implementation with a single role would have a materially wrong gradient.
        self.assertGreater(float((actual[0] - (expected[0] - self_pair_b)).abs().max()), 1e-6)

    def test_replay_rejects_stateful_forward(self):
        pair, rgb, gt, valid = fixture()

        class Stateful:
            calls = 0

            def __call__(self, video, t):
                self.calls += 1
                return pair(video, t) + self.calls * 0.01

        with self.assertRaisesRegex(RuntimeError, "replay changed"):
            loss_and_gradient(Stateful(), rgb, gt, valid)

    def test_detach_and_nonfinite_failures_are_explicit(self):
        pair, rgb, gt, valid = fixture()
        with self.assertRaisesRegex(RuntimeError, "detached"):
            loss_and_gradient(lambda x, t: pair(x, t).detach(), rgb, gt, valid)
        with self.assertRaisesRegex(FloatingPointError, "frame=1"):
            loss_and_gradient(lambda x, t: pair(x, t) * (float("nan") if t == 1 else 1), rgb, gt, valid)


class AttackTests(unittest.TestCase):
    def test_projection_at_rgb_boundaries(self):
        clean = torch.tensor([0., 0.01, 0.5, 0.99, 1.], dtype=torch.float64)
        proposal = torch.tensor([-1., 1., 2., 0., 2.], dtype=torch.float64)
        result = project_rgb(proposal, clean, 0.05)
        expected = torch.tensor([0., 0.06, 0.55, 0.94, 1.], dtype=torch.float64)
        torch.testing.assert_close(result, expected)
        self.assertLessEqual(float((result - clean).abs().max()), 0.05 + 1e-15)

    def test_all_methods_bounds_and_deterministic_seed(self):
        pair, rgb, gt, valid = fixture()
        for method in ["clean", "noise", "fgsm", "pgd"]:
            with self.subTest(method=method):
                config = AttackConfig(method=method, eps=0.03, steps=4, restarts=2, seed=42)
                one = attack_video(pair, rgb, gt, valid, config)
                two = attack_video(pair, rgb, gt, valid, config)
                torch.testing.assert_close(one.rgb, two.rgb, rtol=0, atol=0)
                torch.testing.assert_close(one.delta, one.rgb - rgb, rtol=0, atol=0)
                self.assertLessEqual(float(one.delta.abs().max()), config.eps + 1e-14)
                self.assertTrue(bool(((one.rgb >= 0) & (one.rgb <= 1)).all()))
                if method == "clean":
                    torch.testing.assert_close(one.rgb, rgb, rtol=0, atol=0)
                if method == "pgd":
                    self.assertGreaterEqual(one.loss, one.history[0]["loss"])

    def test_pgd_keeps_best_iterate_and_restart_including_clean(self):
        pair, rgb, gt, valid = fixture()
        attacked = attack_video(pair, rgb, gt, valid,
                                AttackConfig(eps=0.05, steps=5, restarts=3, step_size=0.02, seed=41))
        self.assertAlmostEqual(attacked.loss, max(entry["loss"] for entry in attacked.history), places=14)
        self.assertEqual(len(attacked.history), 1 + 3 * (5 + 1))
        gradient_records = [entry for entry in attacked.history if "gradient_l2_per_frame" in entry]
        self.assertEqual(len(gradient_records), 15)
        self.assertTrue(all(len(entry["gradient_l2_per_frame"]) == rgb.shape[0] for entry in gradient_records))

    def test_optional_progress_callback_receives_each_history_state(self):
        pair, rgb, gt, valid = fixture()
        received = []
        attacked = attack_video(pair, rgb, gt, valid, AttackConfig(steps=2, restarts=2),
                                progress_callback=received.append)
        self.assertEqual(received, attacked.history)
        self.assertEqual(len(received), 1 + 2 * 3)
        self.assertEqual(received[0]["restart"], -1)
        self.assertEqual(received[-1]["restart"], 1)
        self.assertEqual(received[-1]["step"], 2)

    def test_fgsm_takes_eps_sign_step_from_clean(self):
        pair, rgb, gt, valid = fixture()
        derivative = loss_and_gradient(pair, rgb, gt, valid)
        eps = 0.03
        expected = project_rgb(rgb + eps * derivative.gradient.sign(), rgb, eps)
        attacked = attack_video(pair, rgb, gt, valid,
                                AttackConfig(method="fgsm", eps=eps, steps=50, restarts=8, random_start=True))
        self.assertEqual(len(attacked.history), 3)
        # On this smooth toy objective the eps step improves the incumbent.
        self.assertGreater(attacked.loss, derivative.loss)
        torch.testing.assert_close(attacked.rgb, expected, rtol=0, atol=0)

    def test_zero_epsilon_and_float_saved_input_reload(self):
        pair, rgb, gt, valid = fixture(torch.float32)
        zero = attack_video(pair, rgb, gt, valid, AttackConfig(eps=0))
        torch.testing.assert_close(zero.rgb, rgb, rtol=0, atol=0)
        attacked = attack_video(pair, rgb, gt, valid, AttackConfig(eps=4 / 255, steps=3, seed=19))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "float_attack.npz"
            np.savez_compressed(path, rgb=attacked.rgb.numpy(), delta=attacked.delta.numpy())
            with np.load(path, allow_pickle=False) as saved:
                reloaded = torch.from_numpy(saved["rgb"].copy())
            torch.testing.assert_close(reloaded, attacked.rgb, rtol=0, atol=0)
            tracks = predict_tracks(pair, reloaded)
            torch.testing.assert_close(tracks, attacked.tracks, rtol=0, atol=0)
            self.assertEqual(float(aligned_tracking_loss(tracks, gt, valid)), attacked.loss)

    def test_fgsm_returns_the_actual_step_even_when_overshoot_lowers_loss(self):
        rgb = torch.full((1, 3, 1, 1), 0.2, dtype=torch.float64)
        gt = torch.tensor([[[1., 0., 1.], [0., 0., 2.]]], dtype=torch.float64)
        valid = torch.ones((1, 2), dtype=torch.bool)

        def periodic_pair(video, t):
            x = 1 + 0.5 * torch.sin(2 * torch.pi * video.mean())
            zero = x * 0
            return torch.stack([torch.stack([x, zero, zero + 1]),
                                torch.stack([zero, zero, zero + 2])])

        derivative = loss_and_gradient(periodic_pair, rgb, gt, valid)
        self.assertTrue(bool((derivative.gradient > 0).all()))
        attacked = attack_video(periodic_pair, rgb, gt, valid, AttackConfig(method="fgsm", eps=0.3))
        expected = project_rgb(rgb + 0.3 * derivative.gradient.sign(), rgb, 0.3)
        torch.testing.assert_close(attacked.rgb, expected, rtol=0, atol=0)
        self.assertLess(attacked.loss, derivative.loss)

    def test_invalid_configuration_and_rgb_fail_early(self):
        for kwargs in [{"eps": float("nan")}, {"eps": -1}, {"steps": 0}, {"restarts": 0},
                       {"step_size": float("inf")}, {"method": "unknown"}, {"gradient_mode": "unknown"}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                AttackConfig(**kwargs)
        pair, rgb, gt, valid = fixture()
        rgb[0, 0, 0, 0] = -0.01
        with self.assertRaisesRegex(ValueError, "outside"):
            attack_video(pair, rgb, gt, valid)


if __name__ == "__main__":
    unittest.main()
