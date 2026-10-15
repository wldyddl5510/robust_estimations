"""Finite direction sets (covering net and random bank) against exhaustive support LPs."""

from itertools import combinations, product
import unittest

import gurobipy as gp
import numpy as np
from scipy.spatial import cKDTree

from finite_directions import (
    NetSizeError, _support_extremes, brute_force_estimation, discretized_net, random_mom_estimation,
    random_trimmed_estimation, sample_sparse_directions, solve_fixed_support_lp,
)
from utils import gurobi_env


def exhaustive_optimum(directions, targets, s):
    """min over supports of max_j |targets_j - directions_j @ mu|, using only active variables."""
    d = directions.shape[1]
    best_value, best_mu = float(np.max(np.abs(targets))), np.zeros(d)
    with gurobi_env() as env:
        for size in range(1, s + 1):
            for support in combinations(range(d), size):
                with gp.Model(env=env) as model:
                    model.Params.FeasibilityTol = 1e-9
                    model.Params.OptimalityTol = 1e-9
                    active = model.addMVar(size, lb=-gp.GRB.INFINITY)
                    radius = model.addVar(lb=0)
                    residual = directions[:, support] @ active - targets
                    model.addConstr(residual <= radius)
                    model.addConstr(-residual <= radius)
                    model.setObjective(radius)
                    model.optimize()
                    if model.Status != gp.GRB.OPTIMAL:
                        raise AssertionError(model.Status)
                    if model.ObjVal < best_value:
                        best_value, best_mu = float(model.ObjVal), np.zeros(d)
                        best_mu[list(support)] = active.X
    return best_value, best_mu


class NetTests(unittest.TestCase):
    def test_net_covering(self):
        d, s = 5, 2
        rng = np.random.default_rng(7)
        points = np.zeros((500, d))
        for i in range(len(points)):
            support = rng.choice(d, size=rng.integers(1, 2 * s + 1), replace=False)
            v = rng.normal(size=len(support))
            points[i, support] = v / np.linalg.norm(v) * (1 if i % 2 else rng.random())
        for radius in (0.25, 1):
            with self.subTest(radius=radius):
                net = discretized_net(d, s, radius)
                self.assertTrue(np.all(np.linalg.norm(net, axis=1) <= 1 + 1e-12))
                self.assertTrue(np.all(np.count_nonzero(net, axis=1) <= 2 * s))
                self.assertLessEqual(cKDTree(net).query(points)[0].max(), radius + 1e-12)
                np.testing.assert_array_equal(net, discretized_net(d, s, radius))
                np.testing.assert_array_equal(cKDTree(net).query(np.vstack((np.eye(d), -np.eye(d))))[0], 0)
        with self.assertRaisesRegex(NetSizeError, "max_points"):
            discretized_net(100, 5, max_points=100)

    def test_lp_duals_against_distance_to_box(self):
        target, M = np.array([2.0, -3.0]), np.array([4.0, 5.0])
        net = np.vstack((np.eye(2), -np.eye(2)))

        def exact(z):
            return np.max(np.maximum(np.abs(target) - M * z, 0))

        with gurobi_env() as env:
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
        net = np.array([(x, y, 0) for x in (-1, -0.5, 0, 0.5, 1) for y in (-1, 0, 1)], dtype=float)
        medians = np.random.default_rng(8).normal(size=len(net))
        support = np.array([1, 0, 0])
        reduced_net, reduced_medians = _support_extremes(net, medians, support)
        self.assertEqual(len(reduced_net), 10)
        with gurobi_env() as env:
            _, full_value, _ = solve_fixed_support_lp(net, medians, support, np.ones(3) * 4, env=env)
            estimate, reduced_value, gradient = solve_fixed_support_lp(
                reduced_net, reduced_medians, support, np.ones(3) * 4, env=env)
            self.assertAlmostEqual(reduced_value, full_value, places=8)
            self.assertAlmostEqual(np.max(np.abs(medians - net @ estimate)), full_value, places=8)
            for z in product((0, 0.5, 1), repeat=3):
                _, value, _ = solve_fixed_support_lp(net, medians, np.array(z), np.ones(3) * 4, env=env)
                self.assertLessEqual(reduced_value + gradient @ (np.array(z) - support), value + 1e-8)

    def test_brute_force_against_all_supports(self):
        for d, s in [(3, 1), (4, 2), (3, 3)]:
            with self.subTest(d=d, s=s):
                data = np.random.default_rng(9).standard_t(5, size=(80, d))
                data[:, :s] += 2
                original = data.copy()
                estimate, info = brute_force_estimation(data, s, 0.05, seed=4, tol=1e-6, C=1)
                np.testing.assert_array_equal(data, original)
                self.assertTrue(info["converged"])
                self.assertEqual(info["K"], 5)
                blocks = np.array_split(np.random.default_rng(4).permutation(80), info["K"])
                means = np.array([data[block].mean(axis=0) for block in blocks])
                net = discretized_net(d, s)
                medians = np.median(means @ net.T, axis=0)
                optimum, _ = exhaustive_optimum(net, medians, s)
                self.assertLessEqual(np.count_nonzero(estimate), s)
                self.assertLessEqual(info["gap"], info["tol"])
                self.assertLessEqual(info["lower_bound"], optimum + 1e-7)
                self.assertLessEqual(abs(info["objective"] - optimum), info["tol"] + 1e-7)
                self.assertAlmostEqual(info["objective"], np.max(np.abs(medians - net @ estimate)), places=8)

    def test_brute_force_constant_data_and_limits(self):
        truth = np.array([3.0, 0.0, -2.0])
        estimate, info = brute_force_estimation(np.tile(truth, (40, 1)), 2, 0, seed=1)
        np.testing.assert_array_equal(estimate, truth)
        self.assertTrue(info["converged"])
        self.assertEqual((info["objective"], info["C"], info["K"], info["tol"]), (0, 3, 9, 1e-5))
        data = np.random.default_rng(9).standard_t(5, size=(80, 3)) + [2, 0, 0]
        _, info = brute_force_estimation(data, 1, 0.05, seed=4, tol=1e-6, max_iter=1)
        self.assertFalse(info["converged"])
        with self.assertRaisesRegex(ValueError, "exceeds n"):
            brute_force_estimation(np.ones((2, 2)), 1, 0)


class RandomDirectionTests(unittest.TestCase):
    def assert_optimum(self, estimate, info, directions, targets, s):
        optimum, unrestricted_mu = exhaustive_optimum(directions, targets, s)
        d = len(estimate)
        self.assertTrue(info["converged"])
        self.assertLessEqual(np.count_nonzero(estimate), s)
        self.assertLessEqual(info["lower_bound"], optimum + 1e-7)
        self.assertGreaterEqual(info["objective"], optimum - 1e-7)
        self.assertLessEqual(info["objective"] - optimum, info["tol"] + 1e-7)
        self.assertLessEqual(info["gap"], info["tol"])
        self.assertAlmostEqual(info["objective"], np.max(np.abs(targets - directions @ estimate)), places=8)
        # The coordinate directions keep the box bound valid even for the unrestricted optimum.
        center = targets[:d]
        initial_mu = np.zeros(d)
        top = np.argsort(-np.abs(center))[:s]
        initial_mu[top] = center[top]
        initial_value = np.max(np.abs(targets - directions @ initial_mu))
        self.assertTrue(np.all(np.abs(unrestricted_mu) <= np.abs(center) + initial_value + 1e-7))
        self.assertEqual(info["tilde_J"], len(directions) - d)
        self.assertGreater(info["runtime"], 0)

    def test_sampling_geometry_and_reproducibility(self):
        for d, s in ((7, 2), (3, 2), (1, 1)):
            with self.subTest(d=d, s=s):
                bank = sample_sparse_directions(d, s, 37, seed=8)
                self.assertEqual(bank.shape, (d + 37, d))
                np.testing.assert_array_equal(bank[:d], np.eye(d))
                np.testing.assert_allclose(np.linalg.norm(bank[d:], axis=1), 1, atol=1e-14)
                np.testing.assert_array_equal(np.count_nonzero(bank[d:], axis=1), np.full(37, min(2 * s, d)))
                np.testing.assert_array_equal(bank, sample_sparse_directions(d, s, 37, seed=8))
                np.testing.assert_array_equal(bank[:d + 9], sample_sparse_directions(d, s, 9, seed=8))
                self.assertFalse(np.array_equal(bank[d:], sample_sparse_directions(d, s, 37, seed=9)[d:]))
        for count in (0, -1, 1.5, True, np.bool_(True)):
            with self.subTest(tilde_J=count), self.assertRaisesRegex(ValueError, "tilde_J"):
                sample_sparse_directions(3, 1, count)

    def test_mom_against_all_unbounded_supports_and_raw_projections(self):
        for d, s, expected_k in ((3, 1, 5), (4, 2, 7), (3, 3, 7)):
            with self.subTest(d=d, s=s):
                data = np.random.default_rng(19).standard_t(4, size=(41, d))
                data[:, :s] += 1.5
                data[:3] += np.arange(d) * 7
                original = data.copy()
                estimate, info = random_mom_estimation(data, s, .05, delta=.3, tilde_J=19, seed=4, C=2, tol=1e-6)
                directions = sample_sparse_directions(d, s, 19, seed=4)
                blocks = np.array_split(np.random.default_rng(4).permutation(len(data)), info["K"])
                # Project the raw observations, then form block means and medians.
                projected = data @ directions.T
                targets = np.median([projected[block].mean(axis=0) for block in blocks], axis=0)
                self.assert_optimum(estimate, info, directions, targets, s)
                np.testing.assert_array_equal(data, original)
                self.assertEqual((info["C"], info["K"]), (2, expected_k))

    def test_trimmed_against_all_unbounded_supports_and_raw_projections(self):
        data = np.array([[-3., 1., .1], [0., 2., 2.5], [2., -3., -1.], [5., 0., -2.],
                         [-1., 4., 3.], [20., -40., 6.], [-15., 3., 14.]])
        original = data.copy()
        for s in (1, 2, 3):
            with self.subTest(s=s):
                estimate, info = random_trimmed_estimation(data, s, 0, delta=.9, tilde_J=23, seed=7, tol=1e-6)
                directions = sample_sparse_directions(3, s, 23, seed=7)
                k = 2 if s == 1 else 1  # ceil(log(2*C(3, min(2s, 3))/.9))
                targets = np.sort(data @ directions.T, axis=0)[k:len(data) - k].mean(axis=0)
                self.assert_optimum(estimate, info, directions, targets, s)
                self.assertEqual((info["k"], info["effective_n"]), (k, len(data) - 2 * k))
        np.testing.assert_array_equal(data, original)

    def test_constant_data_and_sample_size_limits(self):
        truth = np.array([3., 0., -2.])
        data = np.tile(truth, (35, 1))
        for estimator in (random_mom_estimation, random_trimmed_estimation):
            with self.subTest(estimator=estimator.__name__):
                estimate, info = estimator(data, 2, 0, tilde_J=11, seed=1)
                np.testing.assert_allclose(estimate, truth, atol=1e-8)
                self.assertTrue(info["converged"])
                self.assertAlmostEqual(info["objective"], 0, places=8)
                self.assertEqual(info["tol"], 1e-5)
        with self.assertRaisesRegex(ValueError, "exceeds n"):
            random_mom_estimation(np.ones((2, 2)), 1, 0, tilde_J=3)
        with self.assertRaises(AssertionError):
            random_trimmed_estimation(np.ones((8, 2)), 1, 0, tilde_J=3)

    def test_iteration_limit_retains_valid_sampled_objective_and_bounds(self):
        data = np.random.default_rng(19).standard_t(4, size=(41, 4)) + [1.5, .5, 0, 0]
        for estimator in (random_mom_estimation, random_trimmed_estimation):
            with self.subTest(estimator=estimator.__name__):
                estimate, info = estimator(data, 1, 0, tilde_J=19, seed=4, tol=1e-7, max_iter=1)
                directions = sample_sparse_directions(4, 1, 19, seed=4)
                projected = data @ directions.T
                if estimator is random_mom_estimation:
                    blocks = np.array_split(np.random.default_rng(4).permutation(len(data)), info["K"])
                    targets = np.median([projected[block].mean(axis=0) for block in blocks], axis=0)
                else:
                    targets = np.sort(projected, axis=0)[info["k"]:len(data) - info["k"]].mean(axis=0)
                optimum, _ = exhaustive_optimum(directions, targets, 1)
                self.assertEqual(info["iterations"], 1)
                self.assertFalse(info["converged"])
                self.assertGreater(info["gap"], info["tol"])
                self.assertLessEqual(info["lower_bound"], optimum + 1e-7)
                self.assertGreaterEqual(info["objective"], optimum - 1e-7)
                self.assertAlmostEqual(info["objective"], np.max(np.abs(targets - directions @ estimate)), places=8)


if __name__ == "__main__":
    unittest.main()
