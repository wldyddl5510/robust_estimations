"""Shared building blocks: the support cutting plane, MoM blocks, and hard thresholding."""

import unittest

import numpy as np

from utils import block_means, cutting_plane, mom_blocks, top_s


class CuttingPlaneTests(unittest.TestCase):
    @staticmethod
    def solve_support(support):
        weights = np.array([1.0, 4.0, 2.0])
        # For F(mu)=sum_j weights_j*|1-mu_j| and |mu_j| <= z_j,
        # H(z)=sum_j weights_j*(1-z_j) on [0,1]^d.
        value = weights @ (1 - support)
        return support, value - 0.05, value + 0.1, -weights

    def test_approximate_oracle_bounds(self):
        initial = np.zeros(3)
        estimate, info = cutting_plane(1, initial, initial, 7, self.solve_support, tol=0.2)
        np.testing.assert_array_equal(estimate, [0, 1, 0])
        np.testing.assert_array_equal(initial, np.zeros(3))
        self.assertTrue(info["converged"])
        self.assertEqual(info["iterations"], 2)
        self.assertAlmostEqual(info["objective"], 3.1)
        self.assertAlmostEqual(info["lower_bound"], 2.95)
        self.assertAlmostEqual(info["gap"], 0.15)

    def test_lower_bound_is_not_used_as_incumbent(self):
        _, info = cutting_plane(1, np.zeros(3), np.zeros(3), 7, self.solve_support, tol=0.1, max_iter=2)
        self.assertFalse(info["converged"])
        self.assertGreater(info["gap"], info["tol"])

    def test_relative_gap_and_incumbent_support(self):
        estimate, info = cutting_plane(1, np.zeros(3), np.zeros(3), 7, self.solve_support, tol=0.01, rtol=0.05)
        self.assertTrue(info["converged"])
        self.assertGreater(info["gap"], info["tol"])
        self.assertLessEqual(info["gap"], 0.05 * info["objective"])
        np.testing.assert_array_equal(info["support"], estimate)

    def test_support_retains_selected_zero_coordinate(self):
        def oracle(z):
            return np.zeros(2), 1 - z[0], 1 - z[0], np.array([-1., 0.])
        _, info = cutting_plane(1, np.zeros(2), np.zeros(2), 1, oracle, tol=1e-5)
        self.assertEqual(info["support"], [1, 0])

    def test_rejects_invalid_limits(self):
        for kwargs in ({"tol": 0}, {"tol": np.nan}, {"tol": np.inf}, {"tol": 1e-5, "max_iter": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                cutting_plane(1, np.zeros(3), np.zeros(3), 7, self.solve_support, **kwargs)


class BlockTests(unittest.TestCase):
    def test_block_rule_and_balanced_partition(self):
        data = np.arange(1000., dtype=float).reshape(100, 10)
        # Full support uses complexity d: K = oddceil(2 * max(10, 1, log 20)) = 21.
        blocks = mom_blocks(data, 10, .01, seed=4)
        groups = np.array_split(np.random.default_rng(4).permutation(100), 21)
        np.testing.assert_array_equal(blocks, [data[group].mean(axis=0) for group in groups])
        np.testing.assert_array_equal(blocks, block_means(data, 21, 4))
        self.assertEqual({len(group) for group in groups}, {4, 5})
        # s < d: K = oddceil(2 * 2*log(5e)) = 11.
        self.assertEqual(len(mom_blocks(data, 2, .01, seed=4)), 11)
        # C scales K, and epsilon*n can dominate.
        self.assertEqual(len(mom_blocks(data, 10, .01, C=3, seed=4)), 31)
        self.assertEqual(len(mom_blocks(data, 2, .2, seed=4)), 41)

    def test_invalid_settings(self):
        data = np.ones((100, 10))
        for kwargs in ({"s": 0}, {"s": 11}, {"s": True}, {"s": 1.5}, {"epsilon": .5},
                       {"delta": 0}, {"delta": 1}, {"C": 0}, {"C": np.inf}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                mom_blocks(data, **{"s": 2, "epsilon": .01, **kwargs})
        for bad in (np.ones(5), np.ones((0, 2)), np.full((4, 2), np.nan)):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                mom_blocks(bad, 1, 0)
        with self.assertRaisesRegex(ValueError, "exceeds n"):
            mom_blocks(np.ones((2, 2)), 1, 0)


class TopSTests(unittest.TestCase):
    def test_magnitude_with_stable_ties(self):
        np.testing.assert_array_equal(top_s(np.array([5., -9., 5., 0.]), 2), [1, 1, 0, 0])
        np.testing.assert_array_equal(top_s(np.zeros(3), 2), [1, 1, 0])


if __name__ == "__main__":
    unittest.main()
