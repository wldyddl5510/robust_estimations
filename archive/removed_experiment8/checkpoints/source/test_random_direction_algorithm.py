"""Check sampled objectives against unbounded, exhaustive support LPs."""

from itertools import combinations
import unittest

import gurobipy as gp
import numpy as np

from random_direction_algorithm import (
    random_mom_estimation,
    random_trimmed_estimation,
    sample_sparse_directions,
)


class RandomDirectionTests(unittest.TestCase):
    @staticmethod
    def exhaustive_optimum(directions, targets, s):
        """Use only active variables, with no M bounds or support cuts."""
        d = directions.shape[1]
        best_value, best_mu = float(np.max(np.abs(targets))), np.zeros(d)
        with gp.Env(empty=True) as env:
            env.setParam("OutputFlag", 0)
            env.setParam("Threads", 1)
            env.start()
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
                            best_value = float(model.ObjVal)
                            best_mu = np.zeros(d)
                            best_mu[list(support)] = active.X
        return best_value, best_mu

    def assert_optimum(self, estimate, info, directions, targets, s):
        optimum, unrestricted_mu = self.exhaustive_optimum(directions, targets, s)
        d = len(estimate)
        self.assertTrue(info["converged"])
        self.assertLessEqual(np.count_nonzero(estimate), s)
        self.assertLessEqual(info["lower_bound"], optimum + 1e-7)
        self.assertGreaterEqual(info["objective"], optimum - 1e-7)
        self.assertLessEqual(info["objective"] - optimum, info["tol"] + 1e-7)
        self.assertLessEqual(info["gap"], info["tol"])
        self.assertAlmostEqual(info["objective"],
                               np.max(np.abs(targets - directions @ estimate)), places=8)

        # Axes make the level-set bound valid even for the unrestricted optimum.
        center = targets[:d]
        initial_mu = np.zeros(d)
        initial_support = np.argsort(-np.abs(center))[:s]
        initial_mu[initial_support] = center[initial_support]
        initial_value = np.max(np.abs(targets - directions @ initial_mu))
        self.assertTrue(np.all(np.abs(unrestricted_mu) <= np.abs(center) + initial_value + 1e-7))
        self.assertEqual(info["coordinate_directions"], d)
        self.assertEqual(info["num_directions"], len(directions))
        self.assertEqual(info["tilde_J"], len(directions) - d)
        self.assertEqual(info["direction_sparsity"], min(2 * s, d))
        self.assertEqual(info["objective_scope"], "sampled_directions_plus_coordinates")
        self.assertGreater(info["runtime"], 0)

    def test_sampling_geometry_and_reproducibility(self):
        for d, s in ((7, 2), (3, 2), (1, 1)):
            with self.subTest(d=d, s=s):
                bank = sample_sparse_directions(d, s, 37, seed=8)
                self.assertEqual(bank.shape, (d + 37, d))
                np.testing.assert_array_equal(bank[:d], np.eye(d))
                np.testing.assert_allclose(np.linalg.norm(bank[d:], axis=1), 1, atol=1e-14)
                np.testing.assert_array_equal(np.count_nonzero(bank[d:], axis=1),
                                              np.full(37, min(2 * s, d)))
                np.testing.assert_array_equal(bank, sample_sparse_directions(d, s, 37, seed=8))
                np.testing.assert_array_equal(bank[:d + 9],
                                              sample_sparse_directions(d, s, 9, seed=8))
                self.assertFalse(np.array_equal(bank[d:],
                                                sample_sparse_directions(d, s, 37, seed=9)[d:]))

    def test_mom_against_all_unbounded_supports_and_raw_projections(self):
        for d, s in ((3, 1), (4, 2), (3, 3)):
            with self.subTest(d=d, s=s):
                data = np.random.default_rng(19).standard_t(4, size=(41, d))
                data[:, :s] += 1.5
                data[:3] += np.arange(d) * 7
                original = data.copy()
                estimate, info = random_mom_estimation(
                    data, s, .05, 4, delta=.3, tilde_J=19, seed=4,
                    direction_seed=11, C=2, tol=1e-6, batch_size=3,
                )
                directions = sample_sparse_directions(d, s, 19, seed=11)
                blocks = np.array_split(np.random.default_rng(4).permutation(len(data)), info["K"])
                # Project raw observations, then form block means and medians.
                projected = data @ directions.T
                targets = np.median([projected[block].mean(axis=0) for block in blocks], axis=0)
                self.assert_optimum(estimate, info, directions, targets, s)
                np.testing.assert_array_equal(data, original)
                self.assertEqual(info["C"], 2)
                self.assertEqual(info["K"], 5)
                self.assertEqual(info["seed"], 4)
                self.assertEqual(info["direction_seed"], 11)

    def test_trimmed_against_all_unbounded_supports_and_raw_projections(self):
        data = np.array([[-3., 1., .1], [0., 2., 2.5], [2., -3., -1.],
                         [5., 0., -2.], [-1., 4., 3.], [20., -40., 6.],
                         [-15., 3., 14.]])
        original = data.copy()
        for s in (1, 2, 3):
            with self.subTest(s=s):
                estimate, info = random_trimmed_estimation(
                    data, s, 0, delta=.9, tilde_J=23, seed=7,
                    direction_seed=13, tol=1e-6, batch_size=2,
                )
                directions = sample_sparse_directions(3, s, 23, seed=13)
                targets = np.sort(data @ directions.T, axis=0)[1:-1].mean(axis=0)
                self.assert_optimum(estimate, info, directions, targets, s)
                self.assertEqual(info["k"], 1)
                self.assertEqual(info["effective_n"], len(data) - 2)
                self.assertEqual(info["seed"], 7)
                self.assertEqual(info["direction_seed"], 13)
        np.testing.assert_array_equal(data, original)

    def test_seed_fallback_and_projection_batching_do_not_change_result(self):
        data = np.random.default_rng(23).standard_t(5, size=(40, 4)) + [2, -.5, 0, 0]
        for estimator, args in ((random_mom_estimation, (data, 2, 0, 4)),
                                (random_trimmed_estimation, (data, 2, 0))):
            with self.subTest(estimator=estimator.__name__):
                first, first_info = estimator(*args, tilde_J=17, seed=5)
                second, second_info = estimator(
                    *args, tilde_J=17, seed=5, direction_seed=5, batch_size=1,
                )
                third, third_info = estimator(*args, tilde_J=17, seed=5, batch_size=1000)
                np.testing.assert_allclose(first, second, atol=1e-8)
                np.testing.assert_allclose(first, third, atol=1e-8)
                self.assertAlmostEqual(first_info["objective"], second_info["objective"], places=8)
                self.assertAlmostEqual(first_info["objective"], third_info["objective"], places=8)
                self.assertTrue(first_info["converged"])

    def test_constant_data_and_sample_size_limits(self):
        truth = np.array([3., 0., -2.])
        data = np.tile(truth, (35, 1))
        for estimator, args in ((random_mom_estimation, (data, 2, 0, 2)),
                                (random_trimmed_estimation, (data, 2, 0))):
            for tol in (None, 1e-5):
                with self.subTest(estimator=estimator.__name__, tol=tol):
                    estimate, info = estimator(*args, tilde_J=11, seed=1, tol=tol)
                    np.testing.assert_allclose(estimate, truth, atol=1e-8)
                    self.assertTrue(info["converged"])
                    self.assertAlmostEqual(info["objective"], 0, places=8)
                    self.assertEqual(info["tol"], 1e-5)
        with self.assertRaisesRegex(ValueError, "exceeds n"):
            random_mom_estimation(np.ones((2, 2)), 1, 0, 2, tilde_J=3)
        with self.assertRaises(AssertionError):
            random_trimmed_estimation(np.ones((8, 2)), 1, 0, tilde_J=3)

    def test_invalid_sample_count_and_solver_options(self):
        data = np.ones((35, 3))
        for count in (0, -1, 1.5, True, np.bool_(True)):
            with self.subTest(tilde_J=count):
                with self.assertRaisesRegex(ValueError, "tilde_J"):
                    sample_sparse_directions(3, 1, count)
                with self.assertRaisesRegex(ValueError, "tilde_J"):
                    random_mom_estimation(data, 1, 0, 2, tilde_J=count)
                with self.assertRaisesRegex(ValueError, "tilde_J"):
                    random_trimmed_estimation(data, 1, 0, tilde_J=count)
        for estimator, args in ((random_mom_estimation, (data, 1, 0, 2)),
                                (random_trimmed_estimation, (data, 1, 0))):
            for name, invalid in (("batch_size", 0), ("batch_size", True),
                                  ("max_iter", 0), ("max_iter", True), ("tol", np.nan)):
                with self.subTest(estimator=estimator.__name__, option=name, value=invalid):
                    with self.assertRaises(ValueError):
                        estimator(*args, tilde_J=5, **{name: invalid})

    def test_iteration_limit_retains_valid_sampled_objective_and_bounds(self):
        data = np.random.default_rng(19).standard_t(4, size=(41, 4)) + [1.5, .5, 0, 0]
        for estimator, args in ((random_mom_estimation, (data, 1, 0, 4)),
                                (random_trimmed_estimation, (data, 1, 0))):
            with self.subTest(estimator=estimator.__name__):
                estimate, info = estimator(*args, tilde_J=19, seed=4, tol=1e-7, max_iter=1)
                directions = sample_sparse_directions(4, 1, 19, seed=4)
                projected = data @ directions.T
                if estimator is random_mom_estimation:
                    blocks = np.array_split(np.random.default_rng(4).permutation(len(data)), info["K"])
                    targets = np.median([projected[block].mean(axis=0) for block in blocks], axis=0)
                else:
                    targets = np.sort(projected, axis=0)[info["k"]:len(data) - info["k"]].mean(axis=0)
                optimum, _ = self.exhaustive_optimum(directions, targets, 1)
                self.assertEqual(info["iterations"], 1)
                self.assertFalse(info["converged"])
                self.assertTrue(np.isfinite(info["gap"]))
                self.assertGreater(info["gap"], info["tol"])
                self.assertLessEqual(info["lower_bound"], optimum + 1e-7)
                self.assertGreaterEqual(info["objective"], optimum - 1e-7)
                self.assertAlmostEqual(info["objective"],
                                       np.max(np.abs(targets - directions @ estimate)), places=8)


if __name__ == "__main__":
    unittest.main()
