"""Depersin-Lecue estimator and its covering SDP."""

import unittest

import numpy as np

from depersin_lecue import SDP_ETA, CoveringSDP, depersin_lecue_estimation


class DepersinLecueTests(unittest.TestCase):
    def test_covering_bounds_are_feasible_and_close(self):
        z = np.random.default_rng(19).normal(size=(6, 3))
        result = CoveringSDP(z, 1e-5).solve(.3)
        self.assertTrue(result["converged"])
        self.assertLessEqual(result["lower"], result["upper"])
        self.assertLessEqual(result["upper"], (1 + SDP_ETA) * result["lower"])
        self.assertGreaterEqual(np.linalg.eigvalsh(result["matrix"]).min(), -1e-12)

    def test_zero_residual_covering_has_known_value(self):
        # With z = 0 only the slack covers: y_i = 1/(0.9K), so the value is K/(0.9K) = 10/9.
        result = CoveringSDP(np.zeros((4, 2)), 1e-5).solve(1.)
        self.assertAlmostEqual(result["upper"], 10 / 9, places=6)
        self.assertAlmostEqual(result["lower"], 10 / 9, places=6)

    def test_degenerate_inputs(self):
        data = np.full((11, 3), 2.5)
        estimate, info = depersin_lecue_estimation(data, K=5)
        np.testing.assert_allclose(estimate, 2.5)
        self.assertTrue(info["converged"])
        self.assertEqual(info["termination"], "zero_median_radius")
        for K in (0, 12, 2.5, True):
            with self.subTest(K=K), self.assertRaises(ValueError):
                depersin_lecue_estimation(data, K=K)

    def test_estimate_moves_toward_the_inlier_mean(self):
        rng = np.random.default_rng(3)
        data = rng.normal(size=(105, 3)) + [1., -1., 0.]
        data[:5] = 40.
        estimate, info = depersin_lecue_estimation(data, K=21, seed=0)
        self.assertGreater(info["sdp_calls"], 0)
        self.assertLess(np.linalg.norm(estimate - [1., -1., 0.]), 1.)


if __name__ == "__main__":
    unittest.main()
