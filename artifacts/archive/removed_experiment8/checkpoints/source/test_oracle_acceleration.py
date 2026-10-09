"""Check faster separation against analytic optima and certificate invariants."""

from itertools import combinations
import unittest
from unittest.mock import patch

import gurobipy as gp
import numpy as np

import IP_algorithm as ip


def segment_supremum(means, mu, s):
    """Independent Eq. (13) reference for K=3: distances to line segments."""
    residuals = np.asarray(means) - mu
    d = residuals.shape[1]
    best = 0.0
    for support in combinations(range(d), min(2 * s, d)):
        for blocks in combinations(range(3), 2):
            a, b = residuals[np.ix_(blocks, support)]
            edge = b - a
            squared_length = edge @ edge
            weight = np.clip(-(a @ edge) / squared_length, 0, 1) if squared_length else 0.0
            best = max(best, float(np.linalg.norm(a + weight * edge)))
    return best


class OracleAccelerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gp.Env(empty=True)
        cls.env.setParam("OutputFlag", 0)
        cls.env.setParam("Threads", 1)
        cls.env.start()

    @classmethod
    def tearDownClass(cls):
        cls.env.dispose()

    def test_formulations_match_analytic_sparse_and_dense_optima(self):
        means = np.array([[-3., 1., 0.1, 2.], [0., 2., 2.5, -0.2], [2., -3., -1., 4.]])
        mu = np.array([0.2, -0.3, 0., 0.4])
        for s in (1, 2):
            exact = segment_supremum(means, mu, s)
            for formulation, perspective in (("bigm", False), ("indicator", False),
                                             ("bigm", True), ("indicator", True)):
                with self.subTest(s=s, formulation=formulation, perspective=perspective):
                    lower, upper, (support, blocks) = ip.separation_oracle(
                        means, mu, s, 1e-4, formulation=formulation,
                        perspective=perspective, env=self.env,
                    )
                    self.assertLessEqual(lower, exact + 1e-6)
                    self.assertGreaterEqual(upper, exact - 1e-6)
                    self.assertLessEqual(upper - lower, 1e-4)
                    self.assertTrue(1 <= len(support) <= min(2 * s, means.shape[1]))
                    self.assertEqual(len(blocks), 2)

    def test_heuristic_witness_and_deterministic_bound_on_degenerate_data(self):
        cases = (
            (np.zeros((3, 4)), np.zeros(4), 1),
            (np.tile([2., -3., 1., 0.], (3, 1)), np.zeros(4), 1),
            (np.array([[-4.], [-3.], [2.]]), np.zeros(1), 1),
            (np.array([[1., 2., -1., 0.], [2., -1., 0., 1.],
                       [1000., -3000., 2000., 4000.]]), np.array([0.2, 0., -0.1, 0.]), 1),
            (np.array([[1., 2., -1.], [2., -1., 0.], [-3., 1., 4.]]), np.zeros(3), 2),
        )
        for means, mu, s in cases:
            with self.subTest(means=means.tolist(), s=s):
                original_means, original_mu = means.copy(), mu.copy()
                exact = segment_supremum(means, mu, s)
                bound = ip.deterministic_upper_bound(means, mu, s)
                lower, (support, blocks), direction = ip.heuristic_separation(
                    means, mu, s, env=self.env,
                )
                self.assertTrue(np.isfinite(direction).all())
                self.assertLessEqual(np.linalg.norm(direction), 1 + 1e-9)
                self.assertLessEqual(np.count_nonzero(direction), min(2 * s, means.shape[1]))
                self.assertTrue(set(np.flatnonzero(direction)).issubset(support))
                self.assertEqual(len(blocks), 2)
                self.assertEqual(len(set(blocks)), 2)
                self.assertTrue(1 <= len(support) <= min(2 * s, means.shape[1]))
                projections = (means - mu) @ direction
                self.assertAlmostEqual(lower, float(np.median(projections)), places=7)
                self.assertAlmostEqual(lower, float(np.min(projections[list(blocks)])), places=7)
                self.assertGreaterEqual(lower, -1e-9)
                self.assertLessEqual(lower, exact + 1e-6)
                self.assertGreaterEqual(bound, exact - 1e-7)
                # Reuse the heuristic witness as a MIP start; it cannot change
                # the problem's optimum, including ties and the zero optimum.
                exact_lower, exact_upper, _ = ip.separation_oracle(
                    means, mu, s, 1e-4, warm_start=direction, env=self.env,
                )
                self.assertLessEqual(exact_lower, exact + 1e-6)
                self.assertGreaterEqual(exact_upper, exact - 1e-6)
                self.assertLessEqual(exact_upper - exact_lower, 1e-4)
                np.testing.assert_array_equal(means, original_means)
                np.testing.assert_array_equal(mu, original_mu)

    def test_incomplete_constraint_set_still_gives_global_support_cut(self):
        means = np.array([[-3., 1., 0.1], [0., 2., 2.5], [2., -3., -1.]])
        M = np.full(3, 8.)
        source = np.zeros(3)
        restricted_pairs = [((0, 1), (0, 1))]
        _, lower, gradient = ip.solve_restricted_socp(
            means, source, M, restricted_pairs, env=self.env,
        )
        # The set really is incomplete at the point where its dual cut is made.
        self.assertGreater(segment_supremum(means, np.zeros(3), 1), lower + 0.01)
        all_pairs = [(support, blocks) for support in combinations(range(3), 2)
                     for blocks in combinations(range(3), 2)]
        for support in (np.zeros(3), *np.eye(3)):
            _, full_value, _ = ip.solve_restricted_socp(
                means, support, M, all_pairs, env=self.env,
            )
            self.assertLessEqual(lower + gradient @ (support - source), full_value + 1e-6)

    def test_stalled_heuristic_cannot_certify_or_replace_upper_bound(self):
        means = np.array([[2.], [3.], [4.]])
        # At the fixed mean zero, this first pair has value 2 but F(0)=3.
        # A deliberately poor feasible heuristic always reports the zero vector.
        first_pair = ((0,), (0, 1))
        for certify in (False, True):
            with self.subTest(certify=certify):
                stats = {key: 0 for key in ip._oracle_stats("hybrid")}
                pairs = [first_pair]
                with patch.object(ip, "heuristic_separation",
                                  return_value=(0., first_pair, np.zeros(1))), \
                     patch.object(ip, "separation_oracle", wraps=ip.separation_oracle) as exact:
                    _, lower, upper, _ = ip.constraint_generation(
                        means, np.zeros(1), np.array([10.]), pairs, 1,
                        1e-4, 1e-3, stats, certify=certify,
                        oracle_strategy="hybrid", env=self.env,
                    )
                self.assertAlmostEqual(upper, 3., places=6)
                if certify:
                    self.assertGreaterEqual(exact.call_count, 1)
                    self.assertLessEqual(upper - lower, 1e-3)
                    self.assertGreater(len(pairs), 1)
                else:
                    exact.assert_not_called()
                    self.assertAlmostEqual(lower, 2., places=6)
                    self.assertGreater(upper - lower, 1e-3)
                    self.assertEqual(stats["deferred_supports"], 1)


if __name__ == "__main__":
    unittest.main()
