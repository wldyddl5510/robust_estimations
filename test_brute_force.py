"""Small correctness checks; run with python -m unittest test_brute_force."""

from itertools import combinations
import unittest

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import cKDTree

from brute_force import brute_force_estimation, discretized_net


class BruteForceTests(unittest.TestCase):
    def test_net_covering(self):
        d, s, radius = 5, 2, 0.25
        net = discretized_net(d, s, radius)
        self.assertTrue(np.all(np.linalg.norm(net, axis=1) <= 1 + 1e-12))
        self.assertTrue(np.all(np.count_nonzero(net, axis=1) <= 2 * s))
        rng = np.random.default_rng(7)
        points = np.zeros((500, d))
        for i in range(len(points)):
            support = rng.choice(d, size=rng.integers(1, 2 * s + 1), replace=False)
            v = rng.normal(size=len(support))
            v /= np.linalg.norm(v)
            points[i, support] = v * (1 if i % 2 else rng.random())
        distances, _ = cKDTree(net).query(points)
        self.assertLessEqual(distances.max(), radius + 1e-12)
        np.testing.assert_array_equal(net, discretized_net(d, s, radius))
        np.testing.assert_array_equal(cKDTree(net).query(np.eye(d))[0], 0)

    def test_against_all_supports(self):
        for d, s in [(3, 1), (4, 2), (3, 3)]:
            with self.subTest(d=d, s=s):
                data = np.random.default_rng(9).standard_t(5, size=(80, d))
                data[:, :s] += 2
                original = data.copy()
                estimate, info = brute_force_estimation(
                    data, s, 0.05, 10 / 3, seed=4, tol=1e-6,
                )
                np.testing.assert_array_equal(data, original)
                self.assertTrue(info["converged"])
                self.assertLessEqual(np.count_nonzero(estimate), s)
                self.assertLessEqual(info["gap"], info["tol"])
                self.assertEqual(info["K"], 9)
                self.assertGreater(info["runtime"], 0)

                blocks = np.array_split(np.random.default_rng(4).permutation(80), info["K"])
                means = np.array([data[block].mean(axis=0) for block in blocks])
                net = discretized_net(d, s)
                medians = np.median(means @ net.T, axis=0)
                optimum = np.max(np.abs(medians))  # Empty support.
                for size in range(1, s + 1):
                    for support in combinations(range(d), size):
                        directions = net[:, support]
                        A = np.column_stack((
                            np.vstack((directions, -directions)), -np.ones(2 * len(net)),
                        ))
                        # No M bounds: also checks that the baseline's M is valid.
                        result = linprog(
                            np.r_[np.zeros(size), 1], A_ub=A,
                            b_ub=np.r_[medians, -medians],
                            bounds=[(None, None)] * size + [(0, None)], method="highs",
                        )
                        self.assertTrue(result.success)
                        optimum = min(optimum, result.fun)
                self.assertLessEqual(info["lower_bound"], optimum + 1e-7)
                self.assertLessEqual(abs(info["objective"] - optimum), info["tol"] + 1e-7)
                self.assertAlmostEqual(
                    info["objective"], np.max(np.abs(medians - net @ estimate)), places=8,
                )

    def test_constant_data_and_limits(self):
        truth = np.array([3.0, 0.0, -2.0])
        estimate, info = brute_force_estimation(np.tile(truth, (40, 1)), 2, 0, 2, seed=1)
        np.testing.assert_array_equal(estimate, truth)
        self.assertTrue(info["converged"])
        self.assertEqual(info["objective"], 0)
        data = np.random.default_rng(9).standard_t(5, size=(80, 3)) + [2, 0, 0]
        _, info = brute_force_estimation(data, 1, 0.05, 10 / 3, seed=4, tol=1e-6, max_iter=1)
        self.assertFalse(info["converged"])
        with self.assertRaisesRegex(ValueError, "max_points"):
            discretized_net(100, 5, max_points=100)
        with self.assertRaisesRegex(ValueError, "exceeds n"):
            brute_force_estimation(np.ones((3, 2)), 1, 0, 2)


if __name__ == "__main__":
    unittest.main()
