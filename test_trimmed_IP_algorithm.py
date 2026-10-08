"""Validate Algorithm 2 against independently enumerated trimmed-mean geometry."""

from itertools import combinations, product
import unittest
from unittest.mock import patch

import gurobipy as gp
import numpy as np

import trimmed_IP_algorithm as trimmed


def subset_mean_vertices(points, q):
    """Vertices of the capped-simplex image, built without alpha upper bounds."""
    return np.array([points[list(subset)].mean(axis=0)
                     for subset in combinations(range(len(points)), q)])


def hull_distance(vertices):
    """Distance from zero to a small convex hull using exhaustive active faces.

    Caratheodory's theorem permits at most d+1 active vertices. Equality-
    constrained least squares on each face is independent of the MIP/SOCP
    formulations under test, and includes singleton and degenerate faces.
    """
    best = np.inf
    for size in range(1, min(len(vertices), vertices.shape[1] + 1) + 1):
        for indices in combinations(range(len(vertices)), size):
            face = vertices[list(indices)]
            system = np.block([[face @ face.T, np.ones((size, 1))],
                               [np.ones((1, size)), np.zeros((1, 1))]])
            rhs = np.r_[np.zeros(size), 1.0]
            solution = np.linalg.lstsq(system, rhs, rcond=None)[0]
            weights = solution[:size]
            if (np.linalg.norm(system @ solution - rhs) <= 1e-8
                    and np.min(weights) >= -1e-9):
                best = min(best, float(np.linalg.norm(weights @ face)))
    if not np.isfinite(best):
        raise AssertionError("No convex-hull face found")
    return best


def exhaustive_oracle(data, mu, s, k):
    """Reference max_(S,B) dist(mu_S, conv(q-subset means of X_(B,S)))."""
    n, d = data.shape
    q = n - 2 * k
    value = 0.0
    for support in combinations(range(d), min(2 * s, d)):
        for rows in combinations(range(n), n - k):
            vertices = subset_mean_vertices((data - mu)[np.ix_(rows, support)], q)
            value = max(value, hull_distance(vertices))
    return value


class TrimmedInitializationTests(unittest.TestCase):
    def test_trimmed_mean_even_n_axes_and_no_trimming(self):
        values = np.array([[100., -20., 6., 1.], [0., 10., 3., 2.],
                           [2., 4., 1., 3.], [4., 6., 2., 4.]])
        original = values.copy()
        np.testing.assert_allclose(trimmed.trimmed_mean(values, 1),
                                   np.sort(values, axis=0)[1:-1].mean(axis=0))
        np.testing.assert_allclose(trimmed.trimmed_mean(values, 1, axis=1),
                                   np.sort(values, axis=1)[:, 1:-1].mean(axis=1))
        np.testing.assert_allclose(trimmed.trimmed_mean(values, 0), values.mean(axis=0))
        self.assertEqual(trimmed.trimmed_mean([9., 1., 1., 1., -8.], 2), 1.)
        np.testing.assert_array_equal(values, original)

    def test_initialization_formula_support_and_default_tolerance(self):
        data = np.arange(90., dtype=float).reshape(30, 3) - [0., 40., 60.]
        for epsilon, delta, expected_k in ((0., .05, 4), (.1, .9, 5), (.1, .05, 7)):
            with self.subTest(epsilon=epsilon, delta=delta):
                center, support, initial_mu, k, tol = trimmed.trimmed_initialization(
                    data, 1, epsilon, delta,
                )
                expected_center = np.sort(data, axis=0)[expected_k:-expected_k].mean(axis=0)
                self.assertEqual(k, expected_k)
                self.assertEqual(tol, 1e-5)
                np.testing.assert_allclose(center, expected_center)
                self.assertEqual(support.sum(), 1)
                self.assertEqual(np.argmax(support), np.argmax(np.abs(expected_center)))
                np.testing.assert_allclose(initial_mu, center * support)

    def test_invalid_k_is_asserted_instead_of_clipped(self):
        for n, epsilon, delta in ((100, .4, .05), (8, 0., .05), (5, 0., 1e-8),
                                  (5, 0., 1e-310)):
            with self.subTest(n=n, epsilon=epsilon, delta=delta):
                with self.assertRaises(AssertionError):
                    trimmed.trimmed_initialization(np.ones((n, 2)), 1, epsilon, delta)
                with self.assertRaises(AssertionError):
                    trimmed.trimmed_ip_estimation(np.ones((n, 2)), 1, epsilon, delta)


class TrimmedIPAlgorithmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gp.Env(empty=True)
        cls.env.setParam("OutputFlag", 0)
        cls.env.setParam("Threads", 1)
        cls.env.start()

    @classmethod
    def tearDownClass(cls):
        cls.env.dispose()

    def assert_oracle_certificate(self, data, mu, s, k, exact, **kwargs):
        lower, upper, (support, rows) = trimmed.separation_oracle(
            data, mu, s, k, 1e-4, env=self.env, **kwargs,
        )
        self.assertLessEqual(lower, exact + 2e-6)
        self.assertGreaterEqual(upper, exact - 2e-6)
        self.assertLessEqual(upper - lower, 1e-4)
        self.assertTrue(1 <= len(support) <= min(2 * s, data.shape[1]))
        self.assertEqual(len(rows), len(data) - k)
        self.assertEqual(len(set(rows)), len(rows))
        self.assertTrue(set(support).issubset(range(data.shape[1])))
        self.assertTrue(set(rows).issubset(range(len(data))))

    def test_formulations_match_independent_sparse_and_dense_geometry(self):
        data = np.array([[-3., 1., .1, 2.], [0., 2., 2.5, -.2],
                         [2., -3., -1., 4.], [1., 4., .5, -2.],
                         [-1., -.5, 3., 1.]])
        mu = np.array([.2, -.3, 0., .4])
        for s in (1, 2):
            exact = exhaustive_oracle(data, mu, s, 1)
            for formulation, perspective in product(("bigm", "indicator"), (False, True)):
                with self.subTest(s=s, formulation=formulation, perspective=perspective):
                    self.assert_oracle_certificate(
                        data, mu, s, 1, exact, formulation=formulation,
                        perspective=perspective,
                    )

    def test_theta_must_exceed_objective_bound(self):
        # U_det = 3 but the optimal positive direction needs theta = 5.
        # Incorrectly bounding theta by U_det changes the optimum from 1/3.
        data = np.array([[-2.], [-2.], [-2.], [5.], [5.]])
        self.assertAlmostEqual(trimmed.deterministic_upper_bound(data, np.zeros(1), 1, 1), 3.)
        for formulation in ("bigm", "indicator"):
            with self.subTest(formulation=formulation):
                self.assert_oracle_certificate(
                    data, np.zeros(1), 1, 1, 1. / 3., formulation=formulation,
                    warm_start=np.ones(1),
                )

    def test_one_dimensional_sign_ties_even_n_and_k_zero(self):
        for values, k in (([-8., -3., -2., 1., 5.], 1),
                          ([-8., -3., 2., 5.], 1),
                          ([1., 1., 1., 1., 9.], 1),
                          ([-4., -3., -2., 1.], 0),
                          ([0., 0., 0., 0.], 1)):
            data = np.array(values)[:, None]
            exact = abs(float(np.mean(np.sort(values)[k:len(values) - k])))
            for formulation in ("bigm", "indicator"):
                with self.subTest(values=values, k=k, formulation=formulation):
                    self.assert_oracle_certificate(
                        data, np.zeros(1), 1, k, exact, formulation=formulation,
                    )
        # In higher dimensions k=0 reduces to the sparse norm of the mean.
        data = np.array([[3., -1., 2.], [1., -3., 4.], [2., -2., 3.], [6., 2., -1.]])
        mu = np.array([.5, 0., -.25])
        exact = np.linalg.norm(np.sort(np.abs(data.mean(axis=0) - mu))[-2:])
        self.assert_oracle_certificate(data, mu, 1, 0, exact)

    def test_heuristic_witness_and_deterministic_bound(self):
        cases = (
            (np.zeros((4, 3)), np.zeros(3), 1, 1),
            (np.tile([2., -3., 1.], (5, 1)), np.zeros(3), 1, 1),
            (np.array([[-4.], [-3.], [1.], [2.], [20.]]), np.zeros(1), 1, 1),
            (np.array([[1., 2., -1.], [2., -1., 0.], [-3., 1., 4.],
                       [4., 0., -2.], [100., -150., 200.]]), np.array([.2, 0., -.1]), 1, 1),
            (np.array([[1., 2., -1.], [2., -1., 0.], [-3., 1., 4.], [4., 0., -2.]]),
             np.zeros(3), 2, 0),
        )
        for data, mu, s, k in cases:
            with self.subTest(data=data.tolist(), s=s, k=k):
                original_data, original_mu = data.copy(), mu.copy()
                exact = exhaustive_oracle(data, mu, s, k)
                upper = trimmed.deterministic_upper_bound(data, mu, s, k)
                lower, (support, rows), direction = trimmed.heuristic_separation(
                    data, mu, s, k, env=self.env,
                )
                self.assertTrue(np.isfinite(direction).all())
                self.assertLessEqual(np.linalg.norm(direction), 1 + 1e-9)
                self.assertLessEqual(np.count_nonzero(direction), min(2 * s, data.shape[1]))
                self.assertTrue(set(np.flatnonzero(direction)).issubset(support))
                self.assertEqual(len(rows), len(data) - k)
                self.assertEqual(len(set(rows)), len(rows))
                projections = (data - mu) @ direction
                q = len(data) - 2 * k
                direct_value = np.sort(projections)[k:len(data) - k].mean()
                witness_value = np.sort(projections[list(rows)])[:q].mean()
                self.assertAlmostEqual(lower, direct_value, places=7)
                self.assertAlmostEqual(lower, witness_value, places=7)
                self.assertGreaterEqual(lower, -1e-9)
                self.assertLessEqual(lower, exact + 2e-6)
                self.assertGreaterEqual(upper, exact - 2e-6)
                self.assert_oracle_certificate(data, mu, s, k, exact, warm_start=direction)
                np.testing.assert_array_equal(data, original_data)
                np.testing.assert_array_equal(mu, original_mu)

    def test_decision_threshold_certificates(self):
        data = np.array([[2.], [3.], [4.], [5.], [10.]])
        exact = 4.
        lower, upper, _ = trimmed.separation_oracle(
            data, np.zeros(1), 1, 1, 1e-4, decision_threshold=exact / 2, env=self.env,
        )
        self.assertGreater(lower, exact / 2)
        self.assertLessEqual(lower, exact + 1e-6)
        self.assertGreaterEqual(upper, exact - 1e-6)
        lower, upper, _ = trimmed.separation_oracle(
            data, np.zeros(1), 1, 1, 1e-4, decision_threshold=exact + 1, env=self.env,
        )
        self.assertGreaterEqual(upper, exact - 1e-6)
        self.assertLessEqual(upper, exact + 1)
        self.assertLessEqual(lower, exact + 1e-6)

    def test_socp_caps_and_dual_support_cuts(self):
        # B=(0,1,2,3), q=3 gives the interval [3,6], not the uncapped
        # convex hull [0,9]. The support box is [-10*z, 10*z].
        data = np.array([[0.], [0.], [9.], [9.], [100.]])
        pairs = [((0,), (0, 1, 2, 3))]
        for z in (0., .1, .2, 1.):
            with self.subTest(z=z):
                candidate, value, gradient = trimmed.solve_restricted_socp(
                    data, np.array([z]), np.array([10.]), pairs, 1, env=self.env,
                )
                exact = max(3. - 10. * z, 0.)
                self.assertAlmostEqual(value, exact, places=6)
                self.assertLessEqual(abs(candidate[0]), 10. * z + 1e-9)
                self.assertLessEqual(max(3. - candidate[0], candidate[0] - 6., 0.), value + 1e-6)
                if 0 < z < .3:
                    np.testing.assert_allclose(gradient, [-10.], atol=1e-5)
                for target in (0., .05, .2, .4, 1.):
                    self.assertLessEqual(value + gradient[0] * (target - z),
                                         max(3. - 10. * target, 0.) + 1e-6)

    def full_optimum(self, data, s, k):
        """Enumerate all pairs/supports using convex hulls of subset means.

        This model has no cap constraints, box bound, separation oracle,
        cutting planes, or initialization from the implementation.
        """
        n, d = data.shape
        q = n - 2 * k
        optimum = np.inf
        with gp.Model(env=self.env) as model:
            model.Params.OutputFlag = 0
            model.Params.NonConvex = 0
            model.Params.BarQCPConvTol = 1e-9
            mu = model.addMVar(d, lb=-gp.GRB.INFINITY)
            radius = model.addVar(lb=0)
            for support in combinations(range(d), min(2 * s, d)):
                for rows in combinations(range(n), n - k):
                    vertices = subset_mean_vertices(data[np.ix_(rows, support)], q)
                    weights = model.addMVar(len(vertices), lb=0)
                    residual = model.addMVar(len(support), lb=-gp.GRB.INFINITY)
                    model.addConstr(weights.sum() == 1)
                    model.addConstr(residual == mu[list(support)] - vertices.T @ weights)
                    model.addConstr(residual @ residual <= radius * radius)
            model.setObjective(radius)
            for size in range(s + 1):
                for support in combinations(range(d), size):
                    bounds = np.zeros(d)
                    bounds[list(support)] = gp.GRB.INFINITY
                    mu.LB, mu.UB = -bounds, bounds
                    model.optimize()
                    self.assertEqual(model.Status, gp.GRB.OPTIMAL)
                    optimum = min(optimum, model.ObjVal)
        return optimum

    def test_estimator_against_full_support_and_pair_enumeration(self):
        data = np.array([[-3., 1., .1], [0., 2., 2.5], [2., -3., -1.],
                         [5., 0., -2.], [-1., 4., 3.]])
        for s in (1, 2):
            optimum = self.full_optimum(data, s, 1)
            for strategy in ("hybrid", "exact"):
                with self.subTest(s=s, strategy=strategy):
                    original = data.copy()
                    estimate, info = trimmed.trimmed_ip_estimation(
                        data, s, 0., delta=.9, tol=.001, oracle_strategy=strategy,
                    )
                    actual = exhaustive_oracle(data, estimate, s, 1)
                    np.testing.assert_array_equal(data, original)
                    self.assertTrue(info["converged"])
                    self.assertLessEqual(np.count_nonzero(estimate), s)
                    self.assertLessEqual(info["lower_bound"], optimum + 2e-6)
                    self.assertGreaterEqual(info["objective"], actual - 2e-6)
                    self.assertGreaterEqual(actual, optimum - 2e-6)
                    self.assertLessEqual(info["objective"] - optimum, info["tol"] + 2e-6)
                    self.assertLessEqual(info["gap"], info["tol"])
                    self.assertEqual(info["k"], 1)
                    self.assertEqual(info["n"], len(data))
                    self.assertEqual(info["effective_n"], len(data) - 2)
                    self.assertGreater(info["runtime"], 0)

    def test_constant_data_defaults_and_iteration_limits(self):
        truth = np.array([3., 0., -2.])
        data = np.tile(truth, (9, 1))
        for tol in (1e-5, None):
            with self.subTest(tol=tol):
                estimate, info = trimmed.trimmed_ip_estimation(data, 2, 0., tol=tol)
                np.testing.assert_allclose(estimate, truth, atol=1e-8)
                self.assertTrue(info["converged"])
                self.assertAlmostEqual(info["objective"], 0., places=7)
                self.assertEqual(info["tol"], 1e-5)
                self.assertEqual(info["k"], 4)
                self.assertEqual(info["effective_n"], 1)
        data = np.array([[-3., 1., .1], [0., 2., 2.5], [2., -3., -1.],
                         [5., 0., -2.], [-1., 4., 3.]])
        _, info = trimmed.trimmed_ip_estimation(data, 1, 0., delta=.9, tol=.001, max_iter=1)
        self.assertFalse(info["converged"])
        with self.assertRaisesRegex(RuntimeError, "max_inner_iter"):
            trimmed.trimmed_ip_estimation(
                data, 1, 0., delta=.9, tol=.001, max_inner_iter=1, oracle_strategy="exact",
            )

    def test_stalled_heuristic_falls_back_to_certified_oracle(self):
        data = np.array([[-3., 1., .1], [0., 2., 2.5], [2., -3., -1.],
                         [5., 0., -2.], [-1., 4., 3.]])
        # A feasible zero direction supplies no useful lower bound. Revisited
        # supports must still get exact separation before declaring convergence.
        with patch.object(trimmed, "heuristic_separation",
                          return_value=(0., ((0,), (0, 1, 2, 3)), np.zeros(3))), \
             patch.object(trimmed, "separation_oracle", wraps=trimmed.separation_oracle) as exact:
            estimate, info = trimmed.trimmed_ip_estimation(data, 1, 0., delta=.9, tol=.001)
        optimum = self.full_optimum(data, 1, 1)
        self.assertTrue(info["converged"])
        self.assertGreater(exact.call_count, 0)
        self.assertGreater(info["deferred_supports"], 0)
        self.assertLessEqual(info["objective"] - optimum, info["tol"] + 2e-6)
        self.assertGreaterEqual(info["objective"], exhaustive_oracle(data, estimate, 1, 1) - 2e-6)

    def test_default_tolerance_on_nonconstant_data(self):
        # Regression: an apparently optimal QCP had noisy cone duals and could
        # trigger duplicate-cut failure at the default absolute tolerance.
        data = np.random.default_rng(12).normal(size=(7, 3)) + [1., 0., 0.]
        for strategy, formulation in (("exact", "indicator"), ("hybrid", "bigm")):
            with self.subTest(strategy=strategy, formulation=formulation):
                estimate, info = trimmed.trimmed_ip_estimation(
                    data, 1, 0., delta=.9, oracle_strategy=strategy,
                    oracle_formulation=formulation,
                )
                actual = exhaustive_oracle(data, estimate, 1, 1)
                self.assertTrue(info["converged"])
                self.assertLessEqual(info["gap"], 1e-5)
                self.assertGreaterEqual(info["objective"], actual - 1e-7)

    def test_explicit_dual_recovery_against_interval_geometry(self):
        data = np.array([[0.], [0.], [9.], [9.], [100.]])
        pairs, M = [((0,), (0, 1, 2, 3))], np.array([10.])
        support = np.array([.1])
        # The capped hull is [3,6], and the support box is [-1,1].
        lower, gradient = trimmed._explicit_dual_cut(
            data, support, M, pairs, 3, 2., 1e-6, self.env,
        )
        self.assertAlmostEqual(lower, 2., places=6)
        for z in (0., .1, .2, .5, 1.):
            self.assertLessEqual(lower + gradient @ (np.array([z]) - support),
                                 max(0., 3. - 10 * z) + 1e-7)


if __name__ == "__main__":
    unittest.main()
