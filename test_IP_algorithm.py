"""Compare Algorithm 1 with exhaustive (S,B) and mean-support enumeration."""

from itertools import combinations, product
import unittest

import gurobipy as gp
import numpy as np

from IP_algorithm import ip_estimation, separation_oracle, solve_restricted_socp
from utils import mom_initialization


class IPAlgorithmTests(unittest.TestCase):
    def test_oracle_against_segment_distances(self):
        means = np.array([[-3, 1, 0.1, 2], [0, 2, 2.5, -0.2], [2, -3, -1, 4]])
        mu = np.array([0.2, -0.3, 0, 0.4])
        # With K=3 each B has two points: distance to its convex hull is
        # the distance to a line segment, computed without an optimizer.
        exact = 0.0
        for S in combinations(range(4), 2):
            for B in combinations(range(3), 2):
                a, b = (means - mu)[np.ix_(B, S)]
                edge = b - a
                weight = np.clip(-(a @ edge) / (edge @ edge), 0, 1)
                exact = max(exact, np.linalg.norm(a + weight * edge))
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            lower, upper, (S, B) = separation_oracle(means, mu, 1, 1e-4, env=env)
            self.assertLessEqual(lower, exact + 1e-7)
            self.assertGreaterEqual(upper, exact - 1e-7)
            self.assertLessEqual(upper - lower, 1e-4)
            self.assertTrue(1 <= len(S) <= 2)
            self.assertEqual(len(B), 2)
            # A negative median must still give a positive absolute objective.
            lower, upper, _ = separation_oracle(
                np.array([[-4.0], [-3.0], [-2.0], [1.0], [5.0]]), np.zeros(1), 1, 1e-4, env=env,
            )
            self.assertAlmostEqual(lower, 2.0, places=6)
            self.assertAlmostEqual(upper, 2.0, places=6)

    @staticmethod
    def full_optimum(means, s):
        K, d = means.shape
        optimum = np.inf
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.setParam("Threads", 1)
            env.start()
            with gp.Model(env=env) as model:
                model.Params.NonConvex = 0
                model.Params.BarQCPConvTol = 1e-9
                mu = model.addMVar(d, lb=-gp.GRB.INFINITY)
                r = model.addVar(lb=0)
                for S in combinations(range(d), min(2 * s, d)):
                    for B in combinations(range(K), (K + 1) // 2):
                        alpha = model.addMVar(len(B), lb=0)
                        residual = model.addMVar(len(S), lb=-gp.GRB.INFINITY)
                        model.addConstr(alpha.sum() == 1)
                        model.addConstr(residual == mu[list(S)] - means[np.ix_(B, S)].T @ alpha)
                        model.addConstr(residual @ residual <= r * r)
                model.setObjective(r)
                for size in range(s + 1):
                    for support in combinations(range(d), size):
                        # No M bound and all (S,B) pairs; no cutting-plane loop.
                        bounds = np.zeros(d)
                        bounds[list(support)] = gp.GRB.INFINITY
                        mu.LB, mu.UB = -bounds, bounds
                        model.optimize()
                        if model.Status != gp.GRB.OPTIMAL:
                            raise AssertionError(model.Status)
                        optimum = min(optimum, model.ObjVal)
        return optimum

    def test_socp_duals_against_distance_to_box(self):
        target = np.array([2.0, -3.0])
        M = np.array([4.0, 5.0])
        means = np.tile(target, (3, 1))
        pairs = [((0, 1), (0, 1))]

        def exact(z):
            return np.linalg.norm(np.maximum(np.abs(target) - M * z, 0))

        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.start()
            for support in (np.zeros(2), np.array([0.2, 0.1]), np.ones(2)):
                mu, value, gradient = solve_restricted_socp(means, support, M, pairs, env=env)
                self.assertAlmostEqual(value, exact(support), places=6)
                self.assertLessEqual(np.linalg.norm(mu - target), value + 1e-6)
                self.assertTrue(np.all(np.abs(mu) <= M * support + 1e-9))
                if np.all(support > 0) and value > 1e-6:
                    expected = -M * np.maximum(np.abs(target) - M * support, 0) / exact(support)
                    np.testing.assert_allclose(gradient, expected, atol=1e-5)
                for z in product((0, 0.25, 0.5, 1), repeat=2):
                    z = np.array(z)
                    self.assertLessEqual(value + gradient @ (z - support), exact(z) + 1e-6)

    def test_against_full_enumeration(self):
        for d, s in ((3, 1), (3, 2)):
            with self.subTest(d=d, s=s):
                data = np.random.default_rng(2).standard_t(5, size=(35, d))
                data[:, :s] += 1
                original = data.copy()
                estimate, info = ip_estimation(
                    data, s, 0, 10 / 3, delta=0.4, tol=0.01, seed=4,
                )
                means, _, _, _, _ = mom_initialization(data, s, 0, 10 / 3, delta=0.4, seed=4)
                optimum = self.full_optimum(means, s)
                np.testing.assert_array_equal(data, original)
                self.assertTrue(info["converged"])
                self.assertLessEqual(np.count_nonzero(estimate), s)
                self.assertLessEqual(info["lower_bound"], optimum + 1e-6)
                self.assertGreaterEqual(info["objective"], optimum - 1e-6)
                self.assertLessEqual(info["objective"] - optimum, info["tol"] + 1e-6)
                self.assertLessEqual(info["gap"], info["tol"])
                self.assertEqual(info["oracle_calls"], info["inner_iterations"] + 1)
                self.assertLessEqual(info["constraints"], info["inner_iterations"] + 1)
                self.assertEqual(info["K"], len(means))
                self.assertGreater(info["runtime"], 0)

    def test_constant_data_and_iteration_limits(self):
        truth = np.array([3.0, 0.0, -2.0])
        estimate, info = ip_estimation(np.tile(truth, (35, 1)), 2, 0, 2, seed=1)
        np.testing.assert_allclose(estimate, truth, atol=1e-8)
        self.assertTrue(info["converged"])
        self.assertAlmostEqual(info["objective"], 0, places=7)
        data = np.random.default_rng(2).standard_t(5, size=(35, 3)) + [1, 0, 0]
        _, info = ip_estimation(data, 1, 0, 10 / 3, delta=0.4, tol=0.01, seed=4, max_iter=1)
        self.assertFalse(info["converged"])
        with self.assertRaisesRegex(RuntimeError, "max_inner_iter"):
            ip_estimation(data, 1, 0, 10 / 3, delta=0.4, tol=0.01, seed=4, max_inner_iter=1)


if __name__ == "__main__":
    unittest.main()
