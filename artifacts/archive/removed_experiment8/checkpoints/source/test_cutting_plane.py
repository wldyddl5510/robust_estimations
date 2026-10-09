"""Check the shared outer loop with an approximate fixed-support oracle."""

import unittest

import numpy as np

from utils import cutting_plane


class CuttingPlaneTests(unittest.TestCase):
    def solve_support(self, support):
        weights = np.array([1.0, 4.0, 2.0])
        # For F(mu)=sum_j weights_j*|1-mu_j| and |mu_j| <= z_j,
        # H(z)=sum_j weights_j*(1-z_j) on [0,1]^d.
        value = weights @ (1 - support)
        return support, value - 0.05, value + 0.1, -weights

    def test_approximate_oracle_bounds(self):
        initial = np.zeros(3)
        estimate, info = cutting_plane(
            1, initial, initial, 7, self.solve_support, tol=0.2,
        )
        np.testing.assert_array_equal(estimate, [0, 1, 0])
        np.testing.assert_array_equal(initial, np.zeros(3))
        self.assertTrue(info["converged"])
        self.assertEqual(info["iterations"], 2)
        self.assertAlmostEqual(info["objective"], 3.1)
        self.assertAlmostEqual(info["lower_bound"], 2.95)
        self.assertAlmostEqual(info["gap"], 0.15)

    def test_lower_bound_is_not_used_as_incumbent(self):
        _, info = cutting_plane(
            1, np.zeros(3), np.zeros(3), 7, self.solve_support,
            tol=0.1, max_iter=2,
        )
        self.assertFalse(info["converged"])
        self.assertGreater(info["gap"], info["tol"])

    def test_relative_gap_and_incumbent_support(self):
        estimate, info = cutting_plane(
            1, np.zeros(3), np.zeros(3), 7, self.solve_support,
            tol=0.01, rtol=0.05,
        )
        self.assertTrue(info["converged"])
        self.assertGreater(info["gap"], info["tol"])
        self.assertAlmostEqual(info["gap_tolerance"], 0.155)
        np.testing.assert_array_equal(info["support"], estimate)

    def test_support_retains_selected_zero_coordinate(self):
        def oracle(z):
            return np.zeros(2), 1 - z[0], 1 - z[0], np.array([-1., 0.])
        _, info = cutting_plane(1, np.zeros(2), np.zeros(2), 1, oracle, tol=1e-5)
        self.assertEqual(info["support"], [1, 0])


if __name__ == "__main__":
    unittest.main()
