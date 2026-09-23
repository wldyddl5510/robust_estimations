"""Small correctness checks; run with python -m unittest test_brute_force."""

from itertools import combinations, product
import unittest

import gurobipy as gp
import numpy as np
from scipy.spatial import cKDTree

from brute_force import _support_extremes, brute_force_estimation, discretized_net, solve_fixed_support_lp


class BruteForceTests(unittest.TestCase):
    def test_net_covering(self):
        d, s = 5, 2
        rng = np.random.default_rng(7)
        points = np.zeros((500, d))
        for i in range(len(points)):
            support = rng.choice(d, size=rng.integers(1, 2 * s + 1), replace=False)
            v = rng.normal(size=len(support))
            v /= np.linalg.norm(v)
            points[i, support] = v * (1 if i % 2 else rng.random())
        for radius in (0.25, 1):
            with self.subTest(radius=radius):
                net = discretized_net(d, s, radius)
                self.assertTrue(np.all(np.linalg.norm(net, axis=1) <= 1 + 1e-12))
                self.assertTrue(np.all(np.count_nonzero(net, axis=1) <= 2 * s))
                distances, _ = cKDTree(net).query(points)
                self.assertLessEqual(distances.max(), radius + 1e-12)
                np.testing.assert_array_equal(net, discretized_net(d, s, radius))
                np.testing.assert_array_equal(cKDTree(net).query(np.vstack((np.eye(d), -np.eye(d))))[0], 0)

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
                self.assertEqual(info["K"], 5)
                self.assertGreater(info["runtime"], 0)

                blocks = np.array_split(np.random.default_rng(4).permutation(80), info["K"])
                means = np.array([data[block].mean(axis=0) for block in blocks])
                net = discretized_net(d, s)
                medians = np.median(means @ net.T, axis=0)
                optimum = np.max(np.abs(medians))  # Empty support.
                with gp.Env(empty=True) as env:
                    env.setParam("OutputFlag", 0)
                    env.setParam("Threads", 1)
                    env.start()
                    for size in range(1, s + 1):
                        for support in combinations(range(d), size):
                            # Only active variables, without M bounds or cuts.
                            with gp.Model(env=env) as model:
                                model.Params.FeasibilityTol = 1e-9
                                model.Params.OptimalityTol = 1e-9
                                mu = model.addMVar(size, lb=-gp.GRB.INFINITY)
                                r = model.addVar(lb=0)
                                residual = net[:, support] @ mu - medians
                                model.addConstr(residual <= r)
                                model.addConstr(-residual <= r)
                                model.setObjective(r)
                                model.optimize()
                                self.assertEqual(model.Status, gp.GRB.OPTIMAL)
                                optimum = min(optimum, model.ObjVal)
                self.assertLessEqual(info["lower_bound"], optimum + 1e-7)
                self.assertLessEqual(abs(info["objective"] - optimum), info["tol"] + 1e-7)
                self.assertAlmostEqual(
                    info["objective"], np.max(np.abs(medians - net @ estimate)), places=8,
                )

    def test_lp_duals_against_distance_to_box(self):
        target = np.array([2.0, -3.0])
        M = np.array([4.0, 5.0])
        net = np.vstack((np.eye(2), -np.eye(2)))

        def exact(z):
            return np.max(np.maximum(np.abs(target) - M * z, 0))

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            for support in (np.zeros(2), np.array([0.2, 0.1]), np.ones(2)):
                mu, value, gradient = solve_fixed_support_lp(net, net @ target, support, M, env=env)
                self.assertAlmostEqual(value, exact(support), places=8)
                self.assertLessEqual(np.max(np.abs(mu - target)), value + 1e-8)
                self.assertTrue(np.all(np.abs(mu) <= M * support + 1e-9))
                if np.all(support > 0) and value > 1e-6:
                    np.testing.assert_allclose(gradient, [0, -5], atol=1e-8)
                for z in product((0, 0.25, 0.5, 1), repeat=2):
                    z = np.array(z)
                    self.assertLessEqual(value + gradient @ (z - support), exact(z) + 1e-8)

    def test_support_projection_reduction(self):
        net = np.array([(x, y, 0) for x in (-1, -0.5, 0, 0.5, 1)
                        for y in (-1, 0, 1)], dtype=float)
        medians = np.random.default_rng(8).normal(size=len(net))
        support = np.array([1, 0, 0])
        reduced_net, reduced_medians = _support_extremes(net, medians, support)
        self.assertEqual(len(reduced_net), 10)
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            _, full_value, _ = solve_fixed_support_lp(net, medians, support, np.ones(3) * 4, env=env)
            estimate, reduced_value, gradient = solve_fixed_support_lp(
                reduced_net, reduced_medians, support, np.ones(3) * 4, env=env,
            )
            self.assertAlmostEqual(reduced_value, full_value, places=8)
            self.assertAlmostEqual(np.max(np.abs(medians - net @ estimate)), full_value, places=8)
            for z in product((0, 0.5, 1), repeat=3):
                _, value, _ = solve_fixed_support_lp(net, medians, np.array(z), np.ones(3) * 4, env=env)
                self.assertLessEqual(reduced_value + gradient @ (np.array(z) - support), value + 1e-8)

    def test_constant_data_and_limits(self):
        truth = np.array([3.0, 0.0, -2.0])
        estimate, info = brute_force_estimation(np.tile(truth, (40, 1)), 2, 0, 2, seed=1)
        np.testing.assert_array_equal(estimate, truth)
        self.assertTrue(info["converged"])
        self.assertEqual(info["objective"], 0)
        self.assertEqual(info["C"], 1)
        self.assertEqual(info["K"], 3)
        self.assertAlmostEqual(info["tol"], np.sqrt(3 * 2 / 40) / 100)
        data = np.random.default_rng(9).standard_t(5, size=(80, 3)) + [2, 0, 0]
        _, info = brute_force_estimation(data, 1, 0.05, 10 / 3, seed=4, tol=1e-6, max_iter=1)
        self.assertFalse(info["converged"])
        with self.assertRaisesRegex(ValueError, "max_points"):
            discretized_net(100, 5, max_points=100)
        with self.assertRaisesRegex(ValueError, "exceeds n"):
            brute_force_estimation(np.ones((2, 2)), 1, 0, 2)


if __name__ == "__main__":
    unittest.main()
