"""Compare Algorithm 1 with exhaustive (S,B) and mean-support enumeration."""

from itertools import combinations
import unittest

import cvxpy as cp
import gurobipy as gp
import numpy as np

from IP_algorithm import ip_estimation, separation_oracle
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
        mu = cp.Variable(d)
        r = cp.Variable(nonneg=True)
        constraints = []
        for S in combinations(range(d), min(2 * s, d)):
            for B in combinations(range(K), (K + 1) // 2):
                alpha = cp.Variable(len(B), nonneg=True)
                constraints += [cp.sum(alpha) == 1,
                                cp.SOC(r, means[np.ix_(B, S)].T @ alpha - mu[list(S)])]
        optimum = np.inf
        for size in range(s + 1):
            for support in combinations(range(d), size):
                inactive = sorted(set(range(d)) - set(support))
                zeros = [mu[inactive] == 0] if inactive else []
                # No M bound and all direction/block pairs: independent reference.
                problem = cp.Problem(cp.Minimize(r), constraints + zeros)
                problem.solve(solver=cp.CLARABEL, tol_gap_abs=1e-9, tol_feas=1e-9, tol_gap_rel=1e-9)
                if problem.status != cp.OPTIMAL:
                    raise AssertionError(problem.status)
                optimum = min(optimum, problem.value)
        return optimum

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
