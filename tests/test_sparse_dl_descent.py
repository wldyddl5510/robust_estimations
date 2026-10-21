"""Sparse Depersin-Lecue descent: certified SDP bounds, the support branch-and-bound, and the estimator."""

from itertools import combinations
import unittest

import cvxpy as cp
import numpy as np
from scipy.optimize import linprog

from data import adversarial_sparse_contamination, sparse_t_dist_data_generation
from sparse_dl_descent import game_sdp, sparse_bound, sparse_dl_descent_estimation, support_search, trimmed_average


class SparseDLDescentTests(unittest.TestCase):
    def test_trimmed_average_is_the_capped_simplex_minimum(self):
        values = np.random.default_rng(0).exponential(size=11)
        for keep in (11, 9.9, 8.03, 1.):
            with self.subTest(keep=keep):
                lp = linprog(values, A_eq=np.ones((1, 11)), b_eq=[1], bounds=(0, 1 / keep), method="highs")
                self.assertAlmostEqual(trimmed_average(values, keep), lp.fun, places=10)

    def test_game_sdp_bounds_bracket_an_independent_solve(self):
        r = np.random.default_rng(1).normal(size=(15, 4)) * [2., 1., .5, .2]
        keep = .73 * 15
        lower, upper, M, w = game_sdp(r, keep)
        weights, t = cp.Variable(15), cp.Variable()
        reference = cp.Problem(cp.Minimize(t), [t * np.eye(4) - r.T @ cp.diag(weights) @ r >> 0, weights >= 0,
                                                weights <= 1 / keep, cp.sum(weights) == 1]).solve(solver="SCS", eps=1e-9)
        self.assertLessEqual(lower, upper)
        self.assertLess(upper - lower, 1e-6 * upper)
        self.assertAlmostEqual(upper, reference, delta=1e-5 * upper)
        self.assertAlmostEqual(np.trace(M), 1.)
        self.assertGreaterEqual(np.linalg.eigvalsh(M)[0], -1e-12)
        self.assertAlmostEqual(w.sum(), 1.)
        self.assertLessEqual(w.max(), 1 / keep * (1 + 1e-12))

    def test_sparse_bound_dominates_every_support(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(9, 7))
        C = X.T @ X
        I, A, q = (1, 4), (0, 1, 2, 4, 5, 6), 4
        bound = sparse_bound(C, I, A, q)
        for rest in combinations([j for j in A if j not in I], q - len(I)):
            S = list(I + rest)
            self.assertLessEqual(np.linalg.eigvalsh(C[np.ix_(S, S)])[-1], bound + 1e-12)

    def test_support_search_matches_enumeration(self):
        rng = np.random.default_rng(3)
        r = rng.normal(size=(15, 6)) * rng.uniform(.2, 2, size=6)
        keep = .73 * 15
        for q in (2, 3, 6):
            with self.subTest(q=q):
                values = [game_sdp(r[:, S], keep)[:2] for S in combinations(range(6), q)]
                game = support_search(r, q, keep, .95)
                self.assertTrue(game["certified"])
                self.assertLessEqual(game["lower"], max(upper for _, upper in values) + 1e-9)
                self.assertGreaterEqual(game["upper"], max(lower for lower, _ in values) - 1e-9)
                self.assertEqual(len(game["support"]), q)
                exact = support_search(r, q, keep, 1 - 1e-7)
                self.assertAlmostEqual(exact["lower"], max(lower for lower, _ in values), delta=1e-6)

    def test_estimate_is_sparse_and_accurate(self):
        np.random.seed(4)
        data, truth = sparse_t_dist_data_generation(400, loc=3, nu=5, d=8, s=2)
        data = adversarial_sparse_contamination(data, epsilon=.02, s=2)
        estimate, info = sparse_dl_descent_estimation(data, 2, .02, seed=0)
        self.assertTrue(info["converged"])
        self.assertLessEqual(np.count_nonzero(estimate), 2)
        self.assertEqual(info["objective"], min(h["upper"] for h in info["history"]))
        self.assertLess(np.linalg.norm(estimate - truth), .5)


if __name__ == "__main__":
    unittest.main()
