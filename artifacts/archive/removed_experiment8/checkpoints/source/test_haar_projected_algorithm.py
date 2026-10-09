"""Haar projections and sparse aggregation against exhaustive independent references."""

from itertools import combinations
import unittest
from unittest.mock import patch

import gurobipy as gp
import numpy as np
from scipy.optimize import lsq_linear

from IP_algorithm import dense_estimation, deterministic_upper_bound
from haar_aggregation import aggregate_haar_estimates, fixed_support_aggregation
from haar_projected_algorithm import haar_projected_estimation, random_haar_projections
from utils import mom_initialization


def supports(d, s):
    return [np.isin(np.arange(d), S).astype(float)
            for k in range(s + 1) for S in combinations(range(d), k)]


def losses(P, y, mu):
    return np.sum((P @ mu - y)**2, axis=1)


def reference_ls(P, y, z, M):
    active = np.flatnonzero(z)
    mu = np.zeros(len(z))
    if active.size:
        A = P.reshape(-1, len(z))[:, active]
        result = lsq_linear(A, y.ravel(), bounds=(-M[active], M[active]),
                            tol=1e-13, max_iter=1000)
        if not result.success:
            raise AssertionError(result.message)
        mu[active] = result.x
    return float(losses(P, y, mu).sum())


def reference_max(P, y, z, M, env):
    """Independent SOCP minimizes the norm epigraph, then squares its value."""
    active = np.asarray(z, dtype=bool)
    with gp.Model(env=env) as model:
        model.Params.FeasibilityTol = 1e-9
        model.Params.BarQCPConvTol = 1e-10
        mu = model.addMVar(len(z), lb=np.where(active, -M, 0),
                          ub=np.where(active, M, 0))
        radius = model.addVar(lb=0)
        for projection, target in zip(P, y):
            residual = model.addMVar(len(target), lb=-gp.GRB.INFINITY)
            model.addConstr(residual == projection @ mu - target)
            model.addConstr(residual @ residual <= radius * radius)
        model.setObjective(radius)
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            raise AssertionError(f"Reference SOCP status: {model.Status}")
        value = radius.X**2
        actual = losses(P, y, np.where(active, mu.X, 0)).max()
        if abs(value - actual) > 2e-6 * max(1, actual):
            raise AssertionError(f"Reference residual gap: {actual - value}")
        return float(value)


class HaarProjectionTests(unittest.TestCase):
    def test_row_orthonormal_reproducible_and_independent(self):
        for d, r, J in ((7, 3, 2), (5, 5, 3), (8, 1, 2)):
            first = random_haar_projections(d, r, J, np.random.default_rng(11))
            repeated = random_haar_projections(d, r, J, np.random.default_rng(11))
            longer = random_haar_projections(d, r, J + 1, np.random.default_rng(11))
            self.assertEqual(first.shape, (J, r, d))
            np.testing.assert_allclose(first @ first.transpose(0, 2, 1),
                                       np.broadcast_to(np.eye(r), (J, r, r)), atol=1e-13)
            np.testing.assert_array_equal(first, repeated)
            np.testing.assert_array_equal(first, longer[:J])
            self.assertGreater(np.linalg.norm(first[0] - first[1]), 0.1)
        # Neither equal coordinate partitions nor full ambient rank is required.
        self.assertEqual(random_haar_projections(11, 3, 2, np.random.default_rng(0)).shape,
                         (2, 3, 11))

    def test_isotropic_projection_moments(self):
        P = random_haar_projections(5, 2, 1000, np.random.default_rng(20))
        np.testing.assert_allclose(P.mean(axis=0), 0, atol=0.065)
        np.testing.assert_allclose(np.mean(P.transpose(0, 2, 1) @ P, axis=0),
                                   (2 / 5) * np.eye(5), atol=0.035)

    def test_invalid_dimensions(self):
        for d, r, J in ((0, 1, 1), (4, 0, 1), (4, 5, 1), (4, 2, 0),
                        (4, 2.5, 2), (4, 2, 1.5)):
            with self.subTest(d=d, r=r, J=J), self.assertRaises(ValueError):
                random_haar_projections(d, r, J, np.random.default_rng(0))


class HaarAggregationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = gp.Env(empty=True)
        cls.env.setParam("OutputFlag", 0)
        cls.env.setParam("Threads", 1)
        cls.env.start()

    @classmethod
    def tearDownClass(cls):
        cls.env.dispose()

    def setUp(self):
        rng = np.random.default_rng(3)
        self.P = random_haar_projections(4, 2, 3, rng)
        self.y = rng.normal(size=(3, 2)) * 2
        self.M = np.array([0.35, 0.8, 1.1, 0.5])
        self.supports = supports(4, 2)

    def test_certificates_and_cuts_on_every_support(self):
        for objective in ("least_squares", "max"):
            with self.subTest(objective=objective):
                refs = [reference_ls(self.P, self.y, z, self.M)
                        if objective == "least_squares" else
                        reference_max(self.P, self.y, z, self.M, self.env)
                        for z in self.supports]
                active_bound = False
                for z, value in zip(self.supports, refs):
                    mu, lower, upper, gradient = fixed_support_aggregation(
                        self.P, self.y, z, self.M, objective=objective,
                        tol=1e-7, rtol=1e-7, env=self.env,
                    )
                    np.testing.assert_array_equal(mu[z == 0], 0)
                    self.assertTrue(np.isfinite(np.r_[mu, lower, upper, gradient]).all())
                    self.assertTrue(np.all(np.abs(mu) <= self.M * z + 1e-8))
                    active_bound |= bool(np.any((z > 0) & np.isclose(np.abs(mu), self.M,
                                                                               atol=1e-5)))
                    actual = losses(self.P, self.y, mu)
                    self.assertAlmostEqual(upper, actual.sum() if objective == "least_squares"
                                           else actual.max(), delta=1e-8 * max(1, upper))
                    self.assertLessEqual(lower, value + 3e-6)
                    self.assertGreaterEqual(upper, value - 3e-6)
                    self.assertLessEqual(upper - lower, max(1e-7, 1e-7 * upper) + 1e-8)
                    for other, other_value in zip(self.supports, refs):
                        self.assertLessEqual(lower + gradient @ (other - z),
                                             other_value + 3e-6)
                self.assertTrue(active_bound, "The fixture must exercise finite active M bounds")
                mu, info = aggregate_haar_estimates(
                    self.P, self.y, 2, self.M, objective=objective,
                    tol=1e-6, rtol=1e-6, env=self.env,
                )
                self.assertTrue(info["converged"])
                self.assertLessEqual(np.count_nonzero(mu), 2)
                self.assertTrue(np.all(np.abs(mu) <= self.M + 1e-8))
                self.assertAlmostEqual(info["objective"], min(refs), delta=3e-5)
                self.assertLessEqual(info["gap"], info["gap_tolerance"] + 1e-8)

    def test_underobserved_and_rank_deficient_system(self):
        P = np.array([[[1., 0., 0., 0.]], [[1., 0., 0., 0.]]])
        y = np.array([[1.], [3.]])
        for objective in ("least_squares", "max"):
            mu, info = aggregate_haar_estimates(P, y, 1, np.full(4, 4.),
                                                objective=objective, env=self.env)
            np.testing.assert_allclose(mu, [2, 0, 0, 0], atol=1e-5)
            self.assertTrue(info["converged"])
            self.assertAlmostEqual(info["objective"], 2 if objective == "least_squares"
                                   else 1, places=6)
            zero, zero_info = aggregate_haar_estimates(P, y * 0, 1, np.full(4, 4.),
                                                      objective=objective, env=self.env)
            np.testing.assert_allclose(zero, 0, atol=1e-9)
            self.assertTrue(zero_info["converged"])
            self.assertAlmostEqual(zero_info["objective"], 0)

    def test_max_tie_refinement_preserves_cap(self):
        P = np.array([[[1., 0., 0.]], [[1., 0., 0.]], [[0., 1., 0.]]])
        y = np.array([[-2.], [2.], [3.]])
        mu, info = aggregate_haar_estimates(P, y, 2, np.full(3, 5.), objective="max",
                                            tol=1e-7, rtol=1e-7, env=self.env)
        np.testing.assert_allclose(mu, [0, 3, 0], atol=1e-5)
        self.assertTrue(info["converged"])
        self.assertAlmostEqual(info["objective"], 4, places=6)
        self.assertLessEqual(losses(P, y, mu).max(), 4 + 1e-6)

    def test_scaled_data_use_mixed_certificate_tolerance(self):
        for objective in ("least_squares", "max"):
            mu, info = aggregate_haar_estimates(self.P, self.y * 30, 2, self.M * 30,
                                                objective=objective, env=self.env)
            self.assertTrue(info["converged"])
            self.assertGreater(info["objective"], 100)
            self.assertLessEqual(info["gap"], info["gap_tolerance"] + 1e-7)
            actual = losses(self.P, self.y * 30, mu)
            self.assertAlmostEqual(info["objective"], actual.sum() if objective == "least_squares"
                                   else actual.max(), delta=1e-6)

    def test_scaled_max_refinement_does_not_stall_at_slack_boundary(self):
        # This large-objective fixture stalled when the feasible line search
        # anchored at an incumbent already on the numerical max-cap boundary.
        rng = np.random.default_rng(103)
        for scale in (1., 1000.):
            P = rng.normal(size=(3, 2, 5))
            y = scale * rng.normal(size=(3, 2))
            M = np.full(5, 2 * scale)
        mu, info = aggregate_haar_estimates(P, y, 2, M, objective="max", env=self.env)
        self.assertTrue(info["converged"])
        self.assertTrue(info["refined"])
        self.assertGreater(info["objective"], 1e6)
        self.assertLessEqual(info["gap"], info["gap_tolerance"])
        self.assertLessEqual(info["objective"], info["objective_before_refinement"]
                             + 1.001 * info["refinement_max_slack"])
        self.assertAlmostEqual(info["objective"], losses(P, y, mu).max(), delta=1e-7)
        self.assertLessEqual(np.count_nonzero(mu), 2)
        self.assertTrue(np.all(np.abs(mu) <= M + 1e-8))

    def test_invalid_inputs(self):
        for kwargs in ({"objective": "median"}, {"tol": 0}, {"rtol": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                aggregate_haar_estimates(self.P, self.y, 2, self.M, env=self.env, **kwargs)
        for M in (self.M[:2], np.array([0., 1., 1., 1.]), np.full(4, np.inf)):
            with self.subTest(M=M), self.assertRaises(ValueError):
                aggregate_haar_estimates(self.P, self.y, 2, M, env=self.env)


class HaarEstimatorTests(unittest.TestCase):
    def test_dense_full_sphere_data_driven_box_and_sparse_recovery(self):
        # r neither divides d nor satisfies Jr >= d; global sparsity still
        # identifies this noiseless mean through the known Haar matrix.
        truth = np.array([2., 0., 0., 0.])
        data = np.tile(truth, (25, 1))
        blocks, center, _, initial, _ = mom_initialization(data, 1, 0, 2, delta=0.4,
                                                          tol=0.01, seed=9, C=1)
        bound = deterministic_upper_bound(blocks, initial, 1)
        expected_box = np.abs(center) + bound
        expected_box += 1e-8 * max(1, expected_box.max())
        for objective in ("least_squares", "max"):
            with self.subTest(objective=objective), patch(
                "haar_projected_algorithm.dense_estimation", wraps=dense_estimation,
            ) as dense:
                mu, info = haar_projected_estimation(
                    data, 1, 0, 2, delta=0.4, r=3, J=1, seed=9, C=1,
                    tol=0.01, objective=objective, aggregation_tol=1e-6,
                )
                self.assertEqual(dense.call_count, 1)
                args, kwargs = dense.call_args
                self.assertEqual(args[2] if len(args) > 2 else kwargs["s"], 3)
            np.testing.assert_allclose(mu, truth, atol=2e-5)
            self.assertTrue(info["converged"])
            self.assertEqual(info["direction_sparsity"], 3)
            self.assertEqual(info["projection_method"], "haar")
            self.assertEqual(info["r"], 3)
            self.assertEqual(info["J"], 1)
            self.assertEqual(info["K"], 3)
            self.assertEqual(info["box_K"], len(blocks))
            self.assertAlmostEqual(info["projection_delta"], 0.4)
            self.assertAlmostEqual(info["box_upper"], bound)
            np.testing.assert_allclose(info["M"], expected_box)
            self.assertTrue(np.all(np.asarray(info["M"]) > 0))
            self.assertLessEqual(np.count_nonzero(mu), 1)
            self.assertLessEqual(info["gap"], info["gap_tolerance"] + 1e-8)

    def test_projected_block_confidence_and_repeatability(self):
        truth = np.array([1., 0., -2., 0., 0.])
        data = np.tile(truth, (35, 1))
        first, info = haar_projected_estimation(data, 2, 0, 3, delta=0.4, r=2, J=3,
                                                C=1, seed=17)
        second, _ = haar_projected_estimation(data, 2, 0, 3, delta=0.4, r=2, J=3,
                                               C=1, seed=17)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first, truth, atol=2e-5)
        self.assertEqual(info["K"], 3)  # oddceil(max(2, log(3 / .4))).
        self.assertAlmostEqual(info["projection_delta"], 0.4 / 3)
        self.assertEqual(info["projection_tol"], 1e-5)
        self.assertEqual(info["aggregation_tol"], 1e-5)
        self.assertEqual(info["aggregation_rtol"], 1e-5)

    def test_noisy_end_to_end_matches_exhaustive_sparse_aggregation(self):
        data = np.random.default_rng(41).standard_t(5, size=(30, 3)) + [2, 0, 0]
        with patch("haar_projected_algorithm.aggregate_haar_estimates",
                   wraps=aggregate_haar_estimates) as aggregate:
            mu, info = haar_projected_estimation(data, 1, 0, 4, delta=0.4,
                                                 r=2, J=2, C=1, seed=8, tol=0.01)
        P, y, s, M = aggregate.call_args.args
        projection_seed = np.random.SeedSequence(8).spawn(2)[0]
        np.testing.assert_array_equal(P, random_haar_projections(3, 2, 2,
                                      np.random.default_rng(projection_seed)))
        optimum = min(reference_ls(P, y, z, M) for z in supports(3, s))
        self.assertTrue(info["converged"])
        self.assertGreater(info["inner_iterations"], 0)
        self.assertLessEqual(np.count_nonzero(mu), s)
        self.assertAlmostEqual(info["objective"], optimum, delta=1e-5)
        self.assertAlmostEqual(info["objective"], losses(P, y, mu).sum(), places=8)
        self.assertLessEqual(info["gap"], info["gap_tolerance"] + 1e-8)

    def test_invalid_estimator_inputs(self):
        data = np.zeros((20, 4))
        for kwargs in ({"r": 5, "J": 1}, {"r": 2, "J": 0},
                       {"r": 2, "J": 2, "objective": "other"},
                       {"r": 2, "J": 2, "aggregation_tol": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                haar_projected_estimation(data, 1, 0, 2, **kwargs)


if __name__ == "__main__":
    unittest.main()
