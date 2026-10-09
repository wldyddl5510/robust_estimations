"""Check projection aggregation cuts and solutions against independent references."""

from itertools import combinations
import unittest

import gurobipy as gp
import numpy as np

from IP_algorithm import dense_estimation
from projected_algorithm import (
    aggregate_estimates,
    fixed_support_least_squares,
    fixed_support_max,
    least_squares_statistics,
    projected_estimation,
    refine_max_estimate,
    random_coordinate_projections,
)
from utils import mom_initialization


def supports(d, s):
    return [np.isin(range(d), S).astype(float)
            for k in range(s + 1) for S in combinations(range(d), k)]


def residuals(mu, projections, estimates):
    return np.sum((mu[projections] - estimates)**2, axis=1)


def random_subsets(d, r, J, rng):
    """Independent overlapping subsets, which exercise the general aggregation cuts."""
    return np.array([np.sort(rng.choice(d, size=r, replace=False)) for _ in range(J)])


def reference_max(projections, estimates, support, env):
    """Independent norm-epigraph SOCP; squaring preserves the minimizer."""
    with gp.Model(env=env) as model:
        model.Params.BarQCPConvTol = 1e-9
        model.Params.FeasibilityTol = 1e-9
        mu = model.addVars(len(support), lb=-gp.GRB.INFINITY)
        radius = model.addVar(lb=0)
        for i in np.flatnonzero(support == 0):
            model.addConstr(mu[int(i)] == 0)
        for P, estimate in zip(projections, estimates):
            residual = model.addVars(len(P), lb=-gp.GRB.INFINITY)
            for k, (i, a) in enumerate(zip(P, estimate)):
                model.addConstr(residual[k] == mu[int(i)] - float(a))
            model.addQConstr(gp.quicksum(residual[k]**2 for k in range(len(P))) <= radius**2)
        model.setObjective(radius)
        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            raise AssertionError(f"Reference SOCP status: {model.Status}")
        value = model.ObjVal**2
        candidate = np.array([mu[i].X for i in range(len(support))])
        actual = residuals(candidate, projections, estimates).max()
        if abs(actual - value) > 1e-6:
            raise AssertionError(f"Reference primal residual gap: {actual - value}")
        return value


class ProjectedAlgorithmTests(unittest.TestCase):
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
        self.d, self.s = 4, 2
        self.projections = random_subsets(self.d, 2, 5, rng)
        self.estimates = rng.normal(size=(5, 2)) * 2
        self.supports = supports(self.d, self.s)

    def test_random_partition_projections(self):
        for d, r in ((6, 1), (6, 2), (6, 3), (6, 6)):
            P = random_coordinate_projections(d, r, np.random.default_rng(0))
            self.assertEqual(P.shape, (d // r, r))
            np.testing.assert_array_equal(np.sort(P, axis=1), P)
            # Every coordinate lies in exactly one block.
            np.testing.assert_array_equal(np.sort(P.ravel()), np.arange(d))
        np.testing.assert_array_equal(random_coordinate_projections(6, 2, np.random.default_rng(1)),
                                      random_coordinate_projections(6, 2, np.random.default_rng(1)))
        with self.assertRaisesRegex(ValueError, "r must divide"):
            random_coordinate_projections(6, 4, np.random.default_rng(0))
        with self.assertRaisesRegex(ValueError, "r must be"):
            random_coordinate_projections(6, 7, np.random.default_rng(0))

    def test_partition_aggregation_keeps_block_estimates(self):
        rng = np.random.default_rng(0)
        P = random_coordinate_projections(6, 2, rng)
        estimates = rng.normal(size=(3, 2))
        flat = np.zeros(6)
        flat[P.ravel()] = estimates.ravel()
        # Disjoint blocks give one estimate per coordinate: LS keeps the s largest.
        mu, _ = aggregate_estimates(P, estimates, 6, 2, env=self.env)
        top = np.argsort(-np.abs(flat))[:2]
        np.testing.assert_allclose(mu[top], flat[top])
        self.assertEqual(np.count_nonzero(mu), 2)
        # Max minimizes the largest discarded block mass and keeps the selected estimates.
        mu, info = aggregate_estimates(P, estimates, 6, 2, objective="max", env=self.env)
        selected = mu != 0
        np.testing.assert_allclose(mu[selected], flat[selected])
        best = min(np.max(residuals(np.where(np.isin(range(6), S), flat, 0), P, estimates))
                   for S in combinations(range(6), 2))
        self.assertAlmostEqual(info["objective"], best, places=6)

    def test_weighted_statistics(self):
        P = np.array([[0, 1], [0, 2], [1, 2]])
        estimates = np.array([[2., 4.], [6., 8.], [10., 12.]])
        counts, sums, constant = least_squares_statistics(
            P, estimates, 4, weights=np.array([0., 0.25, 0.75]),
        )
        np.testing.assert_allclose(counts, [0.25, 0.75, 1., 0.])
        np.testing.assert_allclose(sums, [1.5, 7.5, 11., 0.])
        self.assertAlmostEqual(constant, 0.25 * (36 + 64) + 0.75 * (100 + 144))

    def test_least_squares_cuts_and_global_solution(self):
        counts, sums, constant = least_squares_statistics(
            self.projections, self.estimates, self.d,
        )
        values = []
        for z in self.supports:
            # Compute the reference by separately averaging observed estimates.
            reference = np.zeros(self.d)
            for i in np.flatnonzero(z):
                observed = self.estimates[self.projections == i]
                if observed.size:
                    reference[i] = observed.mean()
            value = residuals(reference, self.projections, self.estimates).sum()
            values.append(value)
            mu, lower, upper, gradient = fixed_support_least_squares(
                counts, sums, constant, z,
            )
            np.testing.assert_allclose(mu, reference)
            self.assertAlmostEqual(lower, value)
            self.assertAlmostEqual(upper, value)
            expected_gradient = -np.divide(sums**2, counts, out=np.zeros(self.d), where=counts > 0)
            np.testing.assert_allclose(gradient, expected_gradient)
            for other in self.supports:
                _, other_lower, _, _ = fixed_support_least_squares(
                    counts, sums, constant, other,
                )
                self.assertAlmostEqual(lower + gradient @ (other - z), other_lower)
        mu, info = aggregate_estimates(
            self.projections, self.estimates, self.d, self.s, tol=1e-8, rtol=0, env=self.env,
        )
        self.assertTrue(info["converged"])
        self.assertLessEqual(np.count_nonzero(mu), self.s)
        self.assertAlmostEqual(residuals(mu, self.projections, self.estimates).sum(), min(values))
        self.assertAlmostEqual(info["objective"], min(values))
        self.assertLessEqual(info["gap"], 1e-8)
        self.assertLessEqual(info["iterations"], 2)

    def test_max_cuts_and_global_solution(self):
        values = [reference_max(self.projections, self.estimates, z, self.env)
                  for z in self.supports]
        for z, optimum in zip(self.supports, values):
            mu, lower, upper, gradient = fixed_support_max(
                self.projections, self.estimates, z, tol=1e-7, env=self.env,
            )
            np.testing.assert_array_equal(mu[z == 0], 0)
            self.assertAlmostEqual(upper, residuals(mu, self.projections, self.estimates).max())
            self.assertLessEqual(lower, optimum + 1e-6)
            self.assertGreaterEqual(upper, optimum - 1e-6)
            self.assertLessEqual(upper - lower, 1e-6)
            self.assertTrue(np.all(gradient <= 1e-12))
            for other, other_value in zip(self.supports, values):
                self.assertLessEqual(lower + gradient @ (other - z), other_value + 1e-6)
        mu, info = aggregate_estimates(
            self.projections, self.estimates, self.d, self.s,
            objective="max", tol=1e-6, rtol=0, env=self.env,
        )
        self.assertTrue(info["converged"])
        self.assertLessEqual(np.count_nonzero(mu), self.s)
        self.assertAlmostEqual(info["objective"], min(values), places=5)
        self.assertAlmostEqual(info["objective"], residuals(mu, self.projections, self.estimates).max())
        self.assertLessEqual(info["gap"], 1e-6)

    def test_max_objective_uses_squared_norm(self):
        P, estimates = np.array([[0], [0]]), np.array([[1.], [7.]])
        mu, info = aggregate_estimates(P, estimates, 2, 1, objective="max", tol=1e-6, rtol=0, env=self.env)
        np.testing.assert_allclose(mu, [4, 0], atol=1e-5)
        self.assertAlmostEqual(info["objective"], 9, places=5)
        self.assertTrue(info["converged"])

    def test_max_accepts_certified_suboptimal_solution(self):
        # Barrier can stop SUBOPTIMAL on support {1} (Gurobi 13.0.3 does here);
        # its primal-dual certificate is valid regardless of the solver status.
        P = np.array([[1], [1], [1], [4], [5], [0]])
        estimates = np.array([[9.053339725439907], [11.579671610734005], [-7.642461346324156],
                              [0.9402526590547029], [0.08973864791653074], [9.567275867520655]])
        mu, info = aggregate_estimates(P, estimates, 6, 1, objective="max", tol=1e-5, rtol=0, env=self.env)
        high, low = estimates[1, 0], estimates[2, 0]
        # Coordinate 1 sits midway between its extreme observations.
        np.testing.assert_allclose(mu, [0, (high + low) / 2, 0, 0, 0, 0], atol=1e-5)
        self.assertAlmostEqual(info["objective"], ((high - low) / 2)**2, delta=1e-5)
        self.assertTrue(info["converged"])

    def test_uncovered_and_zero_estimates(self):
        P = np.array([[0, 1], [0, 2]])
        estimates = np.array([[1., 5.], [3., -4.]])
        mu, _ = aggregate_estimates(P, estimates, 4, 3, env=self.env)
        np.testing.assert_allclose(mu, [2, 5, -4, 0])
        for kind in ("least_squares", "max"):
            mu, info = aggregate_estimates(P, np.zeros_like(estimates), 4, 3, objective=kind, env=self.env)
            np.testing.assert_array_equal(mu, np.zeros(4))
            self.assertTrue(info["converged"])
            self.assertAlmostEqual(info["objective"], 0)
            self.assertAlmostEqual(info["gap"], 0)

    def test_refinement_selects_same_least_squares_tie(self):
        P = np.array([[0], [0], [1]])
        estimates = np.array([[-2.], [2.], [3.]])
        # Coordinate 0 is selected even though its optimal value is zero.
        for initial in (np.array([0., 1., 0.]), np.array([0., 5., 0.])):
            mu = refine_max_estimate(P, estimates, [1, 1, 0], initial, env=self.env)
            np.testing.assert_allclose(mu, [0, 3, 0], atol=1e-7)
            self.assertLessEqual(residuals(mu, P, estimates).max(), 4)
            self.assertLess(residuals(mu, P, estimates).sum(), residuals(initial, P, estimates).sum())

    def test_refinement_respects_active_max_constraint(self):
        P, estimates = np.zeros((3, 1), dtype=int), np.array([[0.], [0.], [4.]])
        # Unconstrained LS is 4/3; the max cap requires mu >= 1.9.
        initial = np.array([1.9, 0.])
        mu = refine_max_estimate(P, estimates, [1, 0], initial, env=self.env)
        np.testing.assert_allclose(mu, initial, atol=1e-6)
        self.assertLessEqual(residuals(mu, P, estimates).max(), 4.41 + 4.42e-9)

    def test_refinement_breaks_tie_with_tangent_ball_constraints(self):
        P = np.array([[0, 1], [0, 1], [0, 1], [2, 3]])
        estimates = np.array([[-2., 0.], [2., 0.], [2., 0.], [3., 0.]])
        for initial in (np.array([0., 0., 1., 0.]), np.array([0., 0., 5., 0.])):
            mu = refine_max_estimate(P, estimates, [1, 1, 1, 0], initial, env=self.env)
            np.testing.assert_allclose(mu, [0, 0, 3, 0], atol=1e-5)
            self.assertLessEqual(residuals(mu, P, estimates).max(), 4 + 4.01e-9)

    def test_mixed_tolerance_for_large_max_objectives(self):
        for scale in (5., 10.):
            mu, info = aggregate_estimates(
                self.projections, self.estimates * scale, self.d, self.s,
                objective="max", env=self.env,
            )
            self.assertTrue(info["converged"])
            self.assertTrue(info["refined"])
            self.assertGreater(info["objective"], 100)
            self.assertLessEqual(info["gap"], info["gap_tolerance"])
            self.assertAlmostEqual(info["gap_tolerance"], 1e-5 * info["objective"])
            self.assertAlmostEqual(info["objective"], residuals(mu, self.projections, self.estimates * scale).max())
            self.assertLessEqual(info["objective"], info["objective_before_refinement"]
                                 + 1.001 * info["refinement_max_slack"])

    def test_heavy_tailed_aggregation_refinement(self):
        # These inputs have almost tangent constraints and previously caused
        # barrier failures or stalled refinement under machine-epsilon caps.
        for seed in (1, 4, 12, 17):
            rng = np.random.default_rng(seed)
            P = random_subsets(8, 3, 10, rng)
            estimates = rng.standard_t(2.5, size=(10, 3)) * 5
            mu, info = aggregate_estimates(P, estimates, 8, 3, objective="max", env=self.env)
            self.assertTrue(info["converged"])
            self.assertTrue(info["refined"])
            self.assertLessEqual(info["objective"], info["objective_before_refinement"]
                                 + 1.001 * info["refinement_max_slack"])
            self.assertAlmostEqual(info["objective"], residuals(mu, P, estimates).max())

    def test_iteration_limit_reports_nonconvergence(self):
        P, estimates = np.array([[0], [1]]), np.array([[3.], [2.]])
        mu, info = aggregate_estimates(P, estimates, 2, 1, max_iter=1, tol=1e-8, rtol=0, env=self.env)
        self.assertFalse(info["converged"])
        self.assertEqual(info["iterations"], 1)
        self.assertGreater(info["gap"], 1e-8)
        self.assertLessEqual(np.count_nonzero(mu), 1)
        self.assertAlmostEqual(info["objective"], residuals(mu, P, estimates).sum())
        _, limited = aggregate_estimates(P, estimates, 2, 1, objective="max", max_iter=1, env=self.env)
        self.assertFalse(limited["converged"])
        self.assertFalse(limited["refined"])

    def test_dense_estimation_against_full_enumeration(self):
        data = np.random.default_rng(2).standard_t(5, size=(35, 3)) + [1, 0, -1]
        s = 1
        means, center, *_ = mom_initialization(data, s, 0, 10 / 3, delta=0.4, seed=4)
        K, d = means.shape
        estimate, info = dense_estimation(means, center, s, 0.01, env=self.env)
        with gp.Model(env=self.env) as model:
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

    def test_constant_data_and_separate_tolerances(self):
        truth = np.array([3., 0., -2., 0.])
        for kind in ("least_squares", "max"):
            estimate, info = projected_estimation(
                np.tile(truth, (35, 1)), 2, 0, 2, r=2, seed=1,
                objective=kind, tol=0.01, aggregation_tol=1e-6,
            )
            self.assertEqual(info["J"], 2)
            self.assertEqual(info["uncovered"], 0)
            np.testing.assert_allclose(estimate, truth, atol=1e-6)
            self.assertAlmostEqual(info["objective"], 0, places=6)
            self.assertEqual(info["K"], 7)
            self.assertEqual(info["aggregation"], "cutting_plane")
            self.assertEqual(info["projection_tol"], 0.01)
            self.assertEqual(info["tol"], 1e-6)
            self.assertEqual(info["aggregation_tol"], 1e-6)
            self.assertEqual(info["aggregation_rtol"], 1e-5)
            self.assertTrue(info["converged"])
            self.assertLessEqual(info["gap"], info["aggregation_tol"])
            self.assertLessEqual(info["oracle_calls"], info["inner_iterations"])
            self.assertEqual(info["oracle_calls"], 0)
        _, limited = projected_estimation(
            np.tile(truth, (35, 1)), 2, 0, 2, r=2, seed=1, max_iter=1,
        )
        self.assertFalse(limited["converged"])
        self.assertEqual(limited["iterations"], 1)
        self.assertGreater(limited["gap"], limited["aggregation_tol"])
        with self.assertRaisesRegex(ValueError, "r must be"):
            projected_estimation(np.tile(truth, (35, 1)), 2, 0, 2, r=5, seed=1)
        with self.assertRaisesRegex(ValueError, "r must divide"):
            projected_estimation(np.tile(truth, (35, 1)), 2, 0, 2, r=3, seed=1)

    def test_projection_dimension_below_sparsity(self):
        truth = np.array([3., 0., -2., 1.])
        estimate, info = projected_estimation(np.tile(truth, (35, 1)), 3, 0, 2, r=2, seed=1)
        self.assertEqual(info["uncovered"], 0)
        self.assertEqual(info["projection_tol"], 1e-5)
        np.testing.assert_allclose(estimate, truth, atol=1e-6)
        self.assertTrue(info["converged"])

    def test_seed_reproduces_estimate(self):
        data = np.random.default_rng(5).standard_t(5, size=(40, 6)) + [2, 0, 0, -2, 0, 0]
        first, _ = projected_estimation(data, 2, 0, 10 / 3, r=2, seed=7, delta=0.4)
        second, _ = projected_estimation(data, 2, 0, 10 / 3, r=2, seed=7, delta=0.4)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
