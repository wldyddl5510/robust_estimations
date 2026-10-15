"""Projected Algorithm 1: projections, sparse LS aggregation, and estimators against references."""

from itertools import combinations
import unittest
from unittest.mock import patch

import gurobipy as gp
import numpy as np
from scipy.optimize import lsq_linear

import projection
from IP_algorithm import dense_estimation, deterministic_upper_bound
from projection import (
    aggregate_coordinates, aggregate_haar, haar_estimation, haar_support_ls, partition_estimation,
    random_coordinate_projections, random_haar_projections,
)
from utils import gurobi_env, mom_blocks, top_s


def supports(d, s):
    return [np.isin(np.arange(d), S).astype(float) for k in range(s + 1) for S in combinations(range(d), k)]


def coordinate_losses(mu, projections, estimates):
    return np.sum((mu[projections] - estimates)**2, axis=1)


def haar_losses(P, y, mu):
    return np.sum((P @ mu - y)**2, axis=1)


def reference_haar_ls(P, y, z, M):
    """Boxed least squares on the active coordinates with SciPy, independent of Gurobi."""
    active = np.flatnonzero(z)
    mu = np.zeros(len(z))
    if active.size:
        result = lsq_linear(P.reshape(-1, len(z))[:, active], y.ravel(), bounds=(-M[active], M[active]),
                            tol=1e-13, max_iter=1000)
        if not result.success:
            raise AssertionError(result.message)
        mu[active] = result.x
    return float(haar_losses(P, y, mu).sum())


class CoordinateAggregationTests(unittest.TestCase):
    def test_random_partition_projections(self):
        for d, r in ((6, 1), (6, 2), (6, 3), (6, 6)):
            P = random_coordinate_projections(d, r, np.random.default_rng(0))
            self.assertEqual(P.shape, (d // r, r))
            np.testing.assert_array_equal(np.sort(P, axis=1), P)
            np.testing.assert_array_equal(np.sort(P.ravel()), np.arange(d))
        np.testing.assert_array_equal(random_coordinate_projections(6, 2, np.random.default_rng(1)),
                                      random_coordinate_projections(6, 2, np.random.default_rng(1)))
        for r in (4, 7):
            with self.subTest(r=r), self.assertRaisesRegex(ValueError, "divides"):
                random_coordinate_projections(6, r, np.random.default_rng(0))

    def test_partition_keeps_the_largest_block_estimates(self):
        rng = np.random.default_rng(0)
        P = random_coordinate_projections(6, 2, rng)
        estimates = rng.normal(size=(3, 2))
        flat = np.zeros(6)
        flat[P.ravel()] = estimates.ravel()
        mu, _ = aggregate_coordinates(P, estimates, 6, 2)
        top = np.argsort(-np.abs(flat))[:2]
        np.testing.assert_allclose(mu[top], flat[top])
        self.assertEqual(np.count_nonzero(mu), 2)

    def test_overlapping_subsets_reach_the_exhaustive_optimum(self):
        rng = np.random.default_rng(3)
        d, s = 4, 2
        P = np.array([np.sort(rng.choice(d, size=2, replace=False)) for _ in range(5)])
        estimates = rng.normal(size=(5, 2)) * 2
        values = []
        for z in supports(d, s):
            # Each selected coordinate is the average of its observed estimates.
            reference = np.zeros(d)
            for i in np.flatnonzero(z):
                observed = estimates[P == i]
                reference[i] = observed.mean() if observed.size else 0.
            values.append(coordinate_losses(reference, P, estimates).sum())
        mu, info = aggregate_coordinates(P, estimates, d, s, tol=1e-8)
        self.assertTrue(info["converged"])
        self.assertLessEqual(np.count_nonzero(mu), s)
        self.assertAlmostEqual(coordinate_losses(mu, P, estimates).sum(), min(values))
        self.assertAlmostEqual(info["objective"], min(values))
        self.assertLessEqual(info["iterations"], 2)

    def test_uncovered_and_zero_estimates_and_iteration_limit(self):
        P, estimates = np.array([[0, 1], [0, 2]]), np.array([[1., 5.], [3., -4.]])
        np.testing.assert_allclose(aggregate_coordinates(P, estimates, 4, 3)[0], [2, 5, -4, 0])
        mu, info = aggregate_coordinates(P, np.zeros_like(estimates), 4, 3)
        np.testing.assert_array_equal(mu, np.zeros(4))
        self.assertTrue(info["converged"])
        self.assertAlmostEqual(info["objective"], 0)
        P, estimates = np.array([[0], [1]]), np.array([[3.], [2.]])
        mu, info = aggregate_coordinates(P, estimates, 2, 1, max_iter=1, tol=1e-8)
        self.assertFalse(info["converged"])
        self.assertEqual(info["iterations"], 1)
        self.assertLessEqual(np.count_nonzero(mu), 1)
        self.assertAlmostEqual(info["objective"], coordinate_losses(mu, P, estimates).sum())


class PartitionEstimatorTests(unittest.TestCase):
    def test_dense_estimation_against_full_enumeration(self):
        data = np.random.default_rng(2).standard_t(5, size=(35, 3)) + [1, 0, -1]
        s = 1
        means = mom_blocks(data, s, 0, delta=0.4, seed=4)
        K, d = means.shape
        with gurobi_env() as env:
            estimate, info = dense_estimation(means, s, 0.01, env)
            with gp.Model(env=env) as model:
                model.Params.BarQCPConvTol = 1e-9
                mu = model.addMVar(d, lb=-gp.GRB.INFINITY)
                radius = model.addVar(lb=0)
                for S in combinations(range(d), 2 * s):
                    for B in combinations(range(K), (K + 1) // 2):
                        alpha = model.addMVar(len(B), lb=0)
                        residual = model.addMVar(len(S), lb=-gp.GRB.INFINITY)
                        model.addConstr(alpha.sum() == 1)
                        model.addConstr(residual == mu[list(S)] - means[np.ix_(B, S)].T @ alpha)
                        model.addConstr(residual @ residual <= radius * radius)
                model.setObjective(radius)
                model.optimize()
                optimum = model.ObjVal
        self.assertEqual(np.count_nonzero(estimate), d)
        self.assertLessEqual(info["lower_bound"], optimum + 1e-6)
        self.assertGreaterEqual(info["objective"], optimum - 1e-6)
        self.assertLessEqual(info["objective"] - info["lower_bound"], 0.01 / 4)

    def test_constant_data_limits_and_inputs(self):
        truth = np.array([3., 0., -2., 0.])
        data = np.tile(truth, (35, 1))
        estimate, info = partition_estimation(data, 2, 0, r=2, seed=1, tol=0.01)
        np.testing.assert_allclose(estimate, truth, atol=1e-6)
        self.assertTrue(info["converged"])
        self.assertAlmostEqual(info["objective"], 0, places=6)
        self.assertEqual((info["J"], info["K"], info["oracle_calls"]), (2, 9, 0))
        _, limited = partition_estimation(data, 2, 0, r=2, seed=1, max_iter=1)
        self.assertFalse(limited["converged"])
        self.assertEqual(limited["iterations"], 1)
        for r in (3, 5):
            with self.subTest(r=r), self.assertRaisesRegex(ValueError, "r must be"):
                partition_estimation(data, 2, 0, r=r, seed=1)

    def test_projection_dimension_below_sparsity_and_reproducibility(self):
        truth = np.array([3., 0., -2., 1.])
        estimate, info = partition_estimation(np.tile(truth, (35, 1)), 3, 0, r=2, seed=1)
        np.testing.assert_allclose(estimate, truth, atol=1e-6)
        self.assertTrue(info["converged"])
        data = np.random.default_rng(5).standard_t(5, size=(40, 6)) + [2, 0, 0, -2, 0, 0]
        np.testing.assert_array_equal(partition_estimation(data, 2, 0, r=2, seed=7, delta=0.4)[0],
                                      partition_estimation(data, 2, 0, r=2, seed=7, delta=0.4)[0])


class HaarProjectionTests(unittest.TestCase):
    def test_row_orthonormal_reproducible_and_independent(self):
        for d, r, J in ((7, 3, 2), (5, 5, 3), (8, 1, 2)):
            first = random_haar_projections(d, r, J, np.random.default_rng(11))
            longer = random_haar_projections(d, r, J + 1, np.random.default_rng(11))
            self.assertEqual(first.shape, (J, r, d))
            np.testing.assert_allclose(first @ first.transpose(0, 2, 1), np.broadcast_to(np.eye(r), (J, r, r)),
                                       atol=1e-13)
            np.testing.assert_array_equal(first, random_haar_projections(d, r, J, np.random.default_rng(11)))
            np.testing.assert_array_equal(first, longer[:J])
            self.assertGreater(np.linalg.norm(first[0] - first[1]), 0.1)
        # Neither equal coordinate partitions nor full ambient rank is required.
        self.assertEqual(random_haar_projections(11, 3, 2, np.random.default_rng(0)).shape, (2, 3, 11))

    def test_isotropic_projection_moments(self):
        P = random_haar_projections(5, 2, 1000, np.random.default_rng(20))
        np.testing.assert_allclose(P.mean(axis=0), 0, atol=0.065)
        np.testing.assert_allclose(np.mean(P.transpose(0, 2, 1) @ P, axis=0), (2 / 5) * np.eye(5), atol=0.035)

    def test_invalid_dimensions(self):
        for d, r, J in ((0, 1, 1), (4, 0, 1), (4, 5, 1), (4, 2, 0), (4, 2.5, 2), (4, 2, 1.5)):
            with self.subTest(d=d, r=r, J=J), self.assertRaises(ValueError):
                random_haar_projections(d, r, J, np.random.default_rng(0))


class HaarAggregationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = gurobi_env()
        cls.env = cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def setUp(self):
        rng = np.random.default_rng(3)
        self.P = random_haar_projections(4, 2, 3, rng)
        self.y = rng.normal(size=(3, 2)) * 2
        self.M = np.array([0.35, 0.8, 1.1, 0.5])

    def test_certificates_and_cuts_on_every_support(self):
        all_supports = supports(4, 2)
        refs = [reference_haar_ls(self.P, self.y, z, self.M) for z in all_supports]
        active_bound = False
        for z, value in zip(all_supports, refs):
            mu, lower, upper, gradient = haar_support_ls(self.P, self.y, z, self.M, tol=1e-7, env=self.env)
            np.testing.assert_array_equal(mu[z == 0], 0)
            self.assertTrue(np.isfinite(np.r_[mu, lower, upper, gradient]).all())
            self.assertTrue(np.all(np.abs(mu) <= self.M * z + 1e-8))
            active_bound |= bool(np.any((z > 0) & np.isclose(np.abs(mu), self.M, atol=1e-5)))
            self.assertAlmostEqual(upper, haar_losses(self.P, self.y, mu).sum(), delta=1e-8 * max(1, upper))
            self.assertLessEqual(lower, value + 3e-6)
            self.assertGreaterEqual(upper, value - 3e-6)
            self.assertLessEqual(upper - lower, max(1e-7, 1e-7 * upper) + 1e-8)
            for other, other_value in zip(all_supports, refs):
                self.assertLessEqual(lower + gradient @ (other - z), other_value + 3e-6)
        self.assertTrue(active_bound, "The fixture must exercise finite active M bounds")
        mu, info = aggregate_haar(self.P, self.y, 2, self.M, tol=1e-6, env=self.env)
        self.assertTrue(info["converged"])
        self.assertLessEqual(np.count_nonzero(mu), 2)
        self.assertTrue(np.all(np.abs(mu) <= self.M + 1e-8))
        self.assertAlmostEqual(info["objective"], min(refs), delta=3e-5)

    def test_underobserved_and_rank_deficient_system(self):
        P, y = np.array([[[1., 0., 0., 0.]], [[1., 0., 0., 0.]]]), np.array([[1.], [3.]])
        mu, info = aggregate_haar(P, y, 1, np.full(4, 4.), env=self.env)
        np.testing.assert_allclose(mu, [2, 0, 0, 0], atol=1e-5)
        self.assertTrue(info["converged"])
        self.assertAlmostEqual(info["objective"], 2, places=6)
        zero, zero_info = aggregate_haar(P, y * 0, 1, np.full(4, 4.), env=self.env)
        np.testing.assert_allclose(zero, 0, atol=1e-9)
        self.assertTrue(zero_info["converged"])
        self.assertAlmostEqual(zero_info["objective"], 0)

    def test_scaled_data_use_mixed_certificate_tolerance(self):
        mu, info = aggregate_haar(self.P, self.y * 30, 2, self.M * 30, env=self.env)
        self.assertTrue(info["converged"])
        self.assertGreater(info["objective"], 100)
        self.assertLessEqual(info["gap"], max(1e-5, 1e-5 * info["objective"]) + 1e-7)
        self.assertAlmostEqual(info["objective"], haar_losses(self.P, self.y * 30, mu).sum(), delta=1e-6)


class HaarEstimatorTests(unittest.TestCase):
    def test_dense_full_sphere_data_driven_box_and_sparse_recovery(self):
        # r neither divides d nor satisfies Jr >= d; global sparsity still identifies this noiseless mean.
        truth = np.array([2., 0., 0., 0.])
        data = np.tile(truth, (25, 1))
        blocks = mom_blocks(data, 1, 0, delta=0.4, seed=9, C=1)
        center = np.median(blocks, axis=0)
        expected_box = np.abs(center) + deterministic_upper_bound(blocks, center * top_s(center, 1), 1)
        expected_box += 1e-8 * max(1, expected_box.max())
        with patch.object(projection, "dense_estimation", wraps=dense_estimation) as dense, \
                patch.object(projection, "aggregate_haar", wraps=aggregate_haar) as aggregate:
            mu, info = haar_estimation(data, 1, 0, delta=0.4, r=3, J=1, seed=9, C=1, tol=0.01)
        self.assertEqual(dense.call_count, 1)
        self.assertEqual(dense.call_args.args[1], 3)  # full sphere V_r
        np.testing.assert_allclose(aggregate.call_args.args[3], expected_box)
        np.testing.assert_allclose(mu, truth, atol=2e-5)
        self.assertTrue(info["converged"])
        self.assertEqual((info["r"], info["J"], info["K"], info["box_K"]), (3, 1, 3, len(blocks)))
        self.assertLessEqual(np.count_nonzero(mu), 1)

    def test_projected_block_confidence_and_repeatability(self):
        truth = np.array([1., 0., -2., 0., 0.])
        data = np.tile(truth, (35, 1))
        first, info = haar_estimation(data, 2, 0, delta=0.4, r=2, J=3, C=1, seed=17)
        second, _ = haar_estimation(data, 2, 0, delta=0.4, r=2, J=3, C=1, seed=17)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first, truth, atol=2e-5)
        self.assertEqual(info["K"], 3)  # oddceil(max(2, log(3 / .4)))

    def test_noisy_end_to_end_matches_exhaustive_sparse_aggregation(self):
        data = np.random.default_rng(41).standard_t(5, size=(30, 3)) + [2, 0, 0]
        with patch.object(projection, "aggregate_haar", wraps=aggregate_haar) as aggregate:
            mu, info = haar_estimation(data, 1, 0, delta=0.4, r=2, J=2, C=1, seed=8, tol=0.01)
        P, y, s, M = aggregate.call_args.args
        projection_seed = np.random.SeedSequence(8).spawn(2)[0]
        np.testing.assert_array_equal(P, random_haar_projections(3, 2, 2, np.random.default_rng(projection_seed)))
        optimum = min(reference_haar_ls(P, y, z, M) for z in supports(3, s))
        self.assertTrue(info["converged"])
        self.assertGreater(info["inner_iterations"], 0)
        self.assertLessEqual(np.count_nonzero(mu), s)
        self.assertAlmostEqual(info["objective"], optimum, delta=1e-5)
        self.assertAlmostEqual(info["objective"], haar_losses(P, y, mu).sum(), places=8)

    def test_invalid_estimator_inputs(self):
        data = np.zeros((20, 4))
        for kwargs in ({"r": 5, "J": 1}, {"r": 2, "J": 0}, {"r": 2, "J": 2, "tol": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                haar_estimation(data, 1, 0, **kwargs)


if __name__ == "__main__":
    unittest.main()
