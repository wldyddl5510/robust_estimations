"""Closed-form baselines."""

import unittest

import numpy as np

from baselines import coordinate_mom_estimation, geometric_mom_estimation, sample_mean_ht_estimation
from utils import mom_blocks


class BaselineTests(unittest.TestCase):
    def test_sample_mean_ht_uses_magnitude_and_stable_ties(self):
        data = np.array([[6., -10., 6., 0.], [4., -8., 4., 2.]])
        estimate, info = sample_mean_ht_estimation(data, 2)
        np.testing.assert_array_equal(estimate, [5., -9., 0., 0.])
        self.assertTrue(info["converged"])

    def test_coordinate_mom_thresholds_the_block_median(self):
        data = np.random.default_rng(1).standard_t(5, size=(60, 4)) + [3, 0, -1, 0]
        estimate, info = coordinate_mom_estimation(data, 2, .05, seed=2)
        center = np.median(mom_blocks(data, 2, .05, seed=2), axis=0)
        top = np.argsort(-np.abs(center))[:2]
        np.testing.assert_array_equal(estimate[top], center[top])
        self.assertEqual(np.count_nonzero(estimate), 2)
        self.assertEqual(info["K"], 7)

    def test_geometric_median_then_hard_threshold(self):
        data = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        estimate, info = geometric_mom_estimation(data, 1, 0, delta=0.5, seed=0, C=1)
        np.testing.assert_allclose(estimate, [0, 1 / np.sqrt(3)], atol=1e-7)
        self.assertEqual(info["K"], 3)


if __name__ == "__main__":
    unittest.main()
