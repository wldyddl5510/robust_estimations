"""Algorithm 1 against exhaustive (S, B) and support enumeration and analytic optima."""

from itertools import combinations, product
from time import perf_counter
import unittest
from unittest import mock

import gurobipy as gp
import numpy as np

import IP_algorithm as ip
from utils import gurobi_env, mom_blocks


def segment_supremum(means, mu, s):
    """Independent Eq. (13) reference for K=3: distances to line segments."""
    residuals = np.asarray(means) - mu
    best = 0.0
    for support in combinations(range(residuals.shape[1]), min(2 * s, residuals.shape[1])):
        for blocks in combinations(range(3), 2):
            a, b = residuals[np.ix_(blocks, support)]
            edge = b - a
            weight = np.clip(-(a @ edge) / (edge @ edge), 0, 1) if edge @ edge else 0.0
            best = max(best, float(np.linalg.norm(a + weight * edge)))
    return best


def full_optimum(means, s, env):
    """Minimize over every support with all (S, B) pairs, no box bound and no cutting planes."""
    K, d = means.shape
    optimum = np.inf
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
                bounds = np.zeros(d)
                bounds[list(support)] = gp.GRB.INFINITY
                mu.LB, mu.UB = -bounds, bounds
                model.optimize()
                if model.Status != gp.GRB.OPTIMAL:
                    raise AssertionError(model.Status)
                optimum = min(optimum, model.ObjVal)
    return optimum


class OracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = gurobi_env()
        cls.env = cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.context.__exit__(None, None, None)

    def test_oracle_against_segment_distances(self):
        means = np.array([[-3, 1, 0.1, 2], [0, 2, 2.5, -0.2], [2, -3, -1, 4]])
        mu = np.array([0.2, -0.3, 0, 0.4])
        for s in (1, 2):
            exact = segment_supremum(means, mu, s)
            with self.subTest(s=s):
                lower, upper, (S, B) = ip.separation_oracle(means, mu, s, 1e-4, env=self.env)
                self.assertLessEqual(lower, exact + 1e-6)
                self.assertGreaterEqual(upper, exact - 1e-6)
                self.assertLessEqual(upper - lower, 1e-4)
                self.assertTrue(1 <= len(S) <= min(2 * s, 4))
                self.assertEqual(len(B), 2)
        exact = segment_supremum(means, mu, 1)
        cut_lower, cut_upper, _ = ip.separation_oracle(means, mu, 1, 1e-4, decision_threshold=exact / 2, env=self.env)
        self.assertGreater(cut_lower, exact / 2)
        self.assertGreaterEqual(cut_upper, exact - 1e-7)
        _, certified_upper, _ = ip.separation_oracle(means, mu, 1, 1e-4, decision_threshold=exact + 1, env=self.env)
        self.assertGreaterEqual(certified_upper, exact - 1e-7)
        self.assertLessEqual(certified_upper, exact + 1)
        # A negative median must still give a positive absolute objective.
        lower, upper, _ = ip.separation_oracle(np.array([[-4.], [-3.], [-2.], [1.], [5.]]), np.zeros(1), 1,
                                               1e-4, env=self.env)
        self.assertAlmostEqual(lower, 2.0, places=6)
        self.assertAlmostEqual(upper, 2.0, places=6)

    def test_screened_decisions_match_exact_value(self):
        rng = np.random.default_rng(0)
        for trial in range(6):
            K, d, s = (3, 4, 1) if trial < 3 else (7, 5, 2)
            means, mu = rng.normal(size=(K, d)) * 2, rng.normal(size=d) * 0.3
            exact = (segment_supremum(means, mu, s) if K == 3 else
                     ip.separation_oracle(means, mu, s, 1e-7, env=self.env)[1])
            for G in exact * np.array([0.3, 0.8, 0.97, 1.03, 1.3, 3.0]):
                with self.subTest(trial=trial, ratio=G / exact):
                    lower, upper, _ = ip.separation_oracle(means, mu, s, 1e-6, decision_threshold=G, env=self.env)
                    self.assertGreaterEqual(upper, exact - 1e-6)  # never an invalid certificate
                    self.assertLessEqual(lower, exact + 1e-6)
                    if exact > G:
                        self.assertGreater(lower, G)
                    else:
                        self.assertLessEqual(upper, G)

    def test_screening_infeasibility_is_a_certificate(self):
        # Every row norm is below the threshold, so screening leaves no row and F <= max ||r_i||.
        means, mu = np.array([[1., 0.], [0., 1.], [-1., -1.]]), np.zeros(2)
        exact = segment_supremum(means, mu, 1)
        lower, upper, _ = ip.separation_oracle(means, mu, 1, 1e-6, decision_threshold=2., env=self.env)
        self.assertEqual(lower, 0.)
        self.assertAlmostEqual(upper, np.sqrt(2))
        self.assertGreaterEqual(upper, exact - 1e-9)

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
                lower, (support, blocks), direction = ip.heuristic_separation(means, mu, s, env=self.env)
                self.assertTrue(np.isfinite(direction).all())
                self.assertLessEqual(np.linalg.norm(direction), 1 + 1e-9)
                self.assertLessEqual(np.count_nonzero(direction), min(2 * s, means.shape[1]))
                self.assertTrue(set(np.flatnonzero(direction)).issubset(support))
                self.assertEqual(len(set(blocks)), 2)
                self.assertTrue(1 <= len(support) <= min(2 * s, means.shape[1]))
                projections = (means - mu) @ direction
                self.assertAlmostEqual(lower, float(np.median(projections)), places=7)
                self.assertAlmostEqual(lower, float(np.min(projections[list(blocks)])), places=7)
                self.assertGreaterEqual(lower, -1e-9)
                self.assertLessEqual(lower, exact + 1e-6)
                self.assertGreaterEqual(bound, exact - 1e-7)
                # A heuristic warm start cannot change the optimum, including ties and zero.
                exact_lower, exact_upper, _ = ip.separation_oracle(means, mu, s, 1e-4, warm_start=direction,
                                                                   env=self.env)
                self.assertLessEqual(exact_lower, exact + 1e-6)
                self.assertGreaterEqual(exact_upper, exact - 1e-6)
                self.assertLessEqual(exact_upper - exact_lower, 1e-4)
                np.testing.assert_array_equal(means, original_means)
                np.testing.assert_array_equal(mu, original_mu)

    def test_socp_duals_against_distance_to_box(self):
        target, M = np.array([2.0, -3.0]), np.array([4.0, 5.0])
        means, pairs = np.tile(target, (3, 1)), [((0, 1), (0, 1))]

        def exact(z):
            return np.linalg.norm(np.maximum(np.abs(target) - M * z, 0))

        for support in (np.zeros(2), np.array([0.2, 0.1]), np.ones(2)):
            mu, value, gradient, _ = ip.solve_restricted_socp(means, support, M, pairs, env=self.env)
            self.assertAlmostEqual(value, exact(support), places=6)
            self.assertLessEqual(np.linalg.norm(mu - target), value + 1e-6)
            self.assertTrue(np.all(np.abs(mu) <= M * support + 1e-9))
            if np.all(support > 0) and value > 1e-6:
                expected = -M * np.maximum(np.abs(target) - M * support, 0) / exact(support)
                np.testing.assert_allclose(gradient, expected, atol=1e-5)
            for z in product((0, 0.25, 0.5, 1), repeat=2):
                z = np.array(z)
                self.assertLessEqual(value + gradient @ (z - support), exact(z) + 1e-6)

    def test_incomplete_constraint_set_still_gives_global_support_cut(self):
        means = np.array([[-3., 1., 0.1], [0., 2., 2.5], [2., -3., -1.]])
        M, source = np.full(3, 8.), np.zeros(3)
        _, lower, gradient, _ = ip.solve_restricted_socp(means, source, M, [((0, 1), (0, 1))], env=self.env)
        # The set really is incomplete where its dual cut is made.
        self.assertGreater(segment_supremum(means, np.zeros(3), 1), lower + 0.01)
        all_pairs = [(S, B) for S in combinations(range(3), 2) for B in combinations(range(3), 2)]
        for support in (np.zeros(3), *np.eye(3)):
            _, full_value, _, _ = ip.solve_restricted_socp(means, support, M, all_pairs, env=self.env)
            self.assertLessEqual(lower + gradient @ (support - source), full_value + 1e-6)

    def test_pair_direction_attains_the_fixed_pair_value(self):
        rng = np.random.default_rng(1)
        S, B = (0, 2, 3), (1, 2, 4, 6)
        for offset in (0., 0.5, 2.):
            residuals = rng.normal(size=(7, 5)) + offset
            with gp.Model(env=self.env) as model:
                v = model.addMVar(len(S), lb=-1, ub=1)
                level = model.addVar(lb=0)
                model.addConstr(v @ v <= 1)
                model.addConstr(residuals[np.ix_(B, S)] @ v >= level)
                model.setObjective(level, gp.GRB.MAXIMIZE)
                model.optimize()
                best = model.ObjVal
            with self.subTest(offset=offset, best=best):
                direction, alpha = ip._pair_direction(residuals, S, B)
                value = 0. if direction is None else float(np.min(residuals[list(B)] @ direction))
                self.assertTrue(np.all(alpha >= 0))
                self.assertAlmostEqual(alpha.sum(), 1.)
                point = residuals[np.ix_(B, S)].T @ alpha  # the min-norm point the direction normalizes
                self.assertAlmostEqual(np.linalg.norm(point), best, places=6)
                if direction is not None:
                    self.assertAlmostEqual(np.linalg.norm(direction), 1.)
                    self.assertTrue(set(np.flatnonzero(direction)) <= set(S))
                    np.testing.assert_allclose(direction[list(S)], point / np.linalg.norm(point), atol=1e-8)
                self.assertAlmostEqual(value, best, places=6)

    def test_swap_search_returns_only_new_verified_witnesses(self):
        rng = np.random.default_rng(3)
        means, s, h = rng.normal(size=(9, 6)), 2, 5
        support, M = np.array([1., 1., 0., 0., 0., 0.]), np.full(6, 10.)
        pairs = [ip.heuristic_separation(means, np.zeros(6), s, env=self.env)[1]]
        mu, L, _, weights = ip.solve_restricted_socp(means, support, M, pairs, env=self.env)
        exact = ip.separation_oracle(means, mu, s, 1e-7, env=self.env)[1]
        self.assertGreater(exact, L + 1e-3)  # the single pair leaves violations to find
        never = perf_counter() - 1
        level = L - 1e-6
        self.assertIsNone(ip.swap_separation(means, mu, s, pairs, weights, level, L, set(pairs), never))
        found = ip.swap_separation(means, mu, s, pairs, weights, level, L + 1e-3, set(pairs), perf_counter() + 5)
        self.assertIsNotNone(found)
        lower, pair, direction = found
        self.assertNotIn(pair, pairs)
        self.assertGreater(lower, L + 1e-3)
        self.assertLessEqual(lower, exact + 1e-7)
        self.assertLessEqual(np.count_nonzero(direction), 2 * s)
        self.assertLessEqual(np.linalg.norm(direction), 1 + 1e-12)
        projections = (means - mu) @ direction
        self.assertAlmostEqual(np.sort(projections)[-h], lower)
        self.assertEqual(pair[1], tuple(sorted(np.argsort(projections, kind="stable")[-h:])))
        # Nothing exceeds the exact maximum.
        self.assertIsNone(ip.swap_separation(means, mu, s, pairs, weights, level, exact + 1e-6, set(pairs),
                                             perf_counter() + 5))

    def test_swap_search_drops_a_corral_block(self):
        # d = q = 2, so there are no support swaps, and alternating the pair as is stays at (0, 1, 3).
        means, mu, s, h = np.array([[-2., 3.], [2., -4.], [-1., 3.], [0., -4.], [2., 2.]]), np.zeros(2), 1, 3
        residuals, pair, threshold = means - mu, ((0, 1), (0, 1, 3)), 1.
        direction, alpha = ip._pair_direction(residuals, *pair)
        lower, start, _ = ip._direction_witness(residuals, direction, h)
        self.assertEqual(start, pair)
        self.assertLess(lower, threshold)
        corral = {pair[1][i] for i in np.flatnonzero(alpha > 1e-9)}
        self.assertEqual(corral, {0, 1})
        found = ip.swap_separation(means, mu, s, [pair], [alpha], 0., threshold, {pair}, perf_counter() + 5)
        self.assertIsNotNone(found)
        lower, (S, B), direction = found
        self.assertTrue(corral - set(B))  # a corral block was dropped and replaced
        projections = residuals @ direction
        self.assertGreater(lower, threshold)
        self.assertAlmostEqual(np.sort(projections)[-h], lower)
        self.assertEqual(B, tuple(sorted(np.argsort(projections, kind="stable")[-h:])))
        self.assertLessEqual(lower, ip.separation_oracle(means, mu, s, 1e-8, env=self.env)[1] + 1e-7)
        # The drops are verified like the swaps: nothing above the exact maximum is returned.
        self.assertIsNone(ip.swap_separation(means, mu, s, [pair], [alpha], 0., 2.5, {pair}, perf_counter() + 5))

    def test_stalled_heuristic_cannot_certify_or_replace_upper_bound(self):
        means = np.array([[2.], [3.], [4.]])
        # At mu = 0 the first pair gives 2 while F(0) = 3; the poor heuristic always
        # returns that pair with the zero direction.
        first_pair = ((0,), (0, 1))
        for certify in (False, True):
            with self.subTest(certify=certify), \
                    mock.patch.object(ip, "heuristic_separation", return_value=(0., first_pair, np.zeros(1))), \
                    mock.patch.object(ip, "separation_oracle", wraps=ip.separation_oracle) as exact:
                solver = ip.ConstraintGeneration(means, 1, 4e-3, self.env, np.median(means, axis=0), np.zeros(1))
                _, lower, upper, _ = solver(np.zeros(1), certify=certify)
            self.assertAlmostEqual(upper, 3., places=6)
            if certify:
                # The missing pair comes from the swap search or the exact oracle, never the heuristic.
                self.assertGreaterEqual(exact.call_count + solver.stats["swap_cuts"], 1)
                self.assertLessEqual(upper - lower, 1e-3)
                self.assertGreater(len(solver.pairs), 1)
            else:
                exact.assert_not_called()
                self.assertAlmostEqual(lower, 2., places=6)
                self.assertGreater(upper - lower, 1e-3)
                self.assertEqual(solver.stats["deferred_supports"], 1)

    def test_revisit_prune_skips_certification_with_a_valid_bound(self):
        means = np.array([[2.], [3.], [4.]])
        first_pair = ((0,), (0, 1))
        with mock.patch.object(ip, "heuristic_separation", return_value=(0., first_pair, np.zeros(1))), \
                mock.patch.object(ip, "separation_oracle", wraps=ip.separation_oracle) as exact:
            solver = ip.ConstraintGeneration(means, 1, 4e-3, self.env, np.median(means, axis=0), np.zeros(1))
            solver(np.zeros(1))  # first visit: deferred, U = 3 from the deterministic bound
            solver.best_upper = 1.  # an incumbent the restricted value L = 2 cannot beat by more than tol
            _, lower, upper, _ = solver(np.zeros(1))
        exact.assert_not_called()
        self.assertEqual(solver.stats["revisit_prunes"], 1)
        self.assertAlmostEqual(lower, 2., places=6)
        self.assertGreaterEqual(upper, 3. - 1e-9)  # still a valid upper bound for F(0) = 3


class EstimatorTests(unittest.TestCase):
    data = np.random.default_rng(2).standard_t(5, size=(35, 3)) + [1, 0, 0]

    def test_against_full_enumeration(self):
        for d, s in ((3, 1), (3, 2)):
            with self.subTest(d=d, s=s):
                data = np.random.default_rng(2).standard_t(5, size=(35, d))
                data[:, :s] += 1
                original = data.copy()
                estimate, info = ip.ip_estimation(data, s, 0, delta=0.4, tol=0.01, seed=4, C=2)
                means = mom_blocks(data, s, 0, delta=0.4, seed=4, C=2)
                with gurobi_env() as env:
                    optimum = full_optimum(means, s, env)
                np.testing.assert_array_equal(data, original)
                self.assertTrue(info["converged"])
                self.assertLessEqual(np.count_nonzero(estimate), s)
                self.assertLessEqual(info["lower_bound"], optimum + 1e-6)
                self.assertGreaterEqual(info["objective"], optimum - 1e-6)
                self.assertLessEqual(info["objective"] - optimum, info["tol"] + 1e-6)
                self.assertLessEqual(info["gap"], info["tol"])
                self.assertLessEqual(info["oracle_calls"], info["inner_iterations"])
                self.assertGreater(info["heuristic_calls"], 0)
                self.assertLessEqual(info["constraints"], info["inner_iterations"] + 1)
                self.assertEqual(info["K"], len(means))
                self.assertGreater(info["runtime"], 0)

    def test_constant_data_and_iteration_limits(self):
        truth = np.array([3.0, 0.0, -2.0])
        estimate, info = ip.ip_estimation(np.tile(truth, (35, 1)), 2, 0, seed=1)
        np.testing.assert_allclose(estimate, truth, atol=1e-8)
        self.assertTrue(info["converged"])
        self.assertAlmostEqual(info["objective"], 0, places=7)
        # The initial deterministic bound is 0 on constant data, so tol falls back to rtol.
        self.assertEqual((info["C"], info["K"], info["tol"]), (3, 9, ip.RTOL))  # K = oddceil(3 * log 20)
        _, info = ip.ip_estimation(self.data, 1, 0, delta=0.4, tol=0.01, seed=4, max_iter=1)
        self.assertFalse(info["converged"])
        with mock.patch.object(ip, "MAX_INNER_ITER", 1), self.assertRaisesRegex(RuntimeError, "iteration limit"):
            ip.ip_estimation(self.data, 1, 0, delta=0.4, tol=0.01, seed=4)

    def test_default_tolerance_is_relative_to_the_initial_bound(self):
        means = mom_blocks(self.data, 1, 0, delta=.4, seed=4)
        center = np.median(means, axis=0)
        upper = ip.deterministic_upper_bound(means, center * (np.abs(center) == np.abs(center).max()), 1)
        for rtol in (ip.RTOL, 1e-2):
            with self.subTest(rtol=rtol):
                _, info = ip.ip_estimation(self.data, 1, 0, delta=.4, seed=4, rtol=rtol)
                self.assertAlmostEqual(info["tol"], rtol * upper, places=12)
                self.assertTrue(info["converged"])
                self.assertLessEqual(info["gap"], info["tol"])
        # The same data scaled by 10 gives a 10x tolerance.
        _, scaled = ip.ip_estimation(10 * self.data, 1, 0, delta=.4, seed=4)
        self.assertAlmostEqual(scaled["tol"], 10 * ip.RTOL * upper, places=10)
        self.assertEqual(ip.ip_estimation(self.data, 1, 0, delta=.4, seed=4, tol=.01)[1]["tol"], .01)
        with self.assertRaises(ValueError):
            ip.ip_estimation(self.data, 1, 0, delta=.4, seed=4, rtol=0)

    def test_block_solver_matches_estimator_and_rejects_invalid_blocks(self):
        estimate, info = ip.ip_estimation(self.data, 1, 0, delta=.4, tol=.01, seed=4)
        means = mom_blocks(self.data, 1, 0, delta=.4, seed=4)
        direct, direct_info = ip.ip_estimation_from_block_means(means, 1, tol=.01)
        np.testing.assert_allclose(estimate, direct, atol=1e-8)
        self.assertAlmostEqual(info["objective"], direct_info["objective"], places=8)
        self.assertEqual(info["K"], direct_info["K"])
        for means, s, tol in [(np.ones((4, 2)), 1, 1e-5), (np.ones((0, 2)), 1, 1e-5),
                              (np.array([[np.nan]]), 1, 1e-5), (np.ones((3, 2)), 3, 1e-5),
                              (np.ones((3, 2)), 1, 0)]:
            with self.subTest(shape=means.shape, s=s, tol=tol), self.assertRaises(ValueError):
                ip.ip_estimation_from_block_means(means, s, tol=tol)


class BlockSubsamplingTests(unittest.TestCase):
    def subsample(self, data, s, epsilon, **kwargs):
        """Run block subsampling with the solver mocked; return (info, block means it received)."""
        result = (np.zeros(data.shape[1]), {"converged": True, "runtime": 0.})
        with mock.patch.object(ip, "ip_estimation_from_block_means", return_value=result) as solve:
            _, info = ip.ip_estimation(data, s, epsilon, subsample=True, **kwargs)
        return info, solve.call_args.args[0]

    def test_selected_blocks_of_the_original_partition_reach_the_solver(self):
        data = np.arange(1500., dtype=float).reshape(300, 5)
        info, received = self.subsample(data, 2, .1, seed=3, C=2)
        self.assertEqual((info["K"], info["K_sub"]), (61, 9))
        blocks = np.array_split(np.random.default_rng(3).permutation(300), 61)
        means = np.array([data[block].mean(axis=0) for block in blocks])
        selected = info["selected_block_indices"]
        self.assertEqual(selected, sorted(set(selected)))
        np.testing.assert_array_equal(received, means[selected])

    def test_subsample_counts(self):
        # K_sub = min(K, oddceil(C * max(s*log(e*d/s), log(4/delta)))); full support gives d.
        info, _ = self.subsample(np.arange(1000., dtype=float).reshape(100, 10), 10, .01, seed=4, C=2)
        self.assertEqual((info["K"], info["K_sub"], info["selected_block_indices"]), (21, 21, list(range(21))))
        for d, expected in [(5, 9), (10, 11), (20, 15)]:
            with self.subTest(d=d):
                self.assertEqual(self.subsample(np.ones((100, d)), 2, .1, seed=0, C=2)[0]["K_sub"], expected)
        for C, K, K_sub in [(1, 11, 5), (2, 21, 11), (3, 31, 15), (4, 41, 21), (5, 51, 25)]:
            with self.subTest(C=C):
                info, _ = self.subsample(np.ones((1000, 5)), 5, .01, seed=7, C=C)
                self.assertEqual((info["K"], info["K_sub"]), (K, K_sub))

    def test_all_blocks_selected_matches_algorithm_1(self):
        data = EstimatorTests.data
        expected, full = ip.ip_estimation(data, 1, 0, delta=.4, tol=.01, seed=4)
        estimate, info = ip.ip_estimation(data, 1, 0, delta=.4, tol=.01, seed=4, subsample=True)
        self.assertEqual(info["K_sub"], info["K"])
        self.assertEqual(info["selected_block_indices"], list(range(full["K"])))
        np.testing.assert_allclose(estimate, expected, atol=1e-8)
        self.assertAlmostEqual(info["objective"], full["objective"], places=8)

    def test_reproducible_selection_leaves_global_rng_alone(self):
        data = np.random.default_rng(0).normal(size=(200, 10))
        np.random.seed(123)
        state = np.random.get_state()
        first, _ = self.subsample(data, 2, .1, seed=4)
        second, _ = self.subsample(data, 2, .1, seed=4)
        other, _ = self.subsample(data, 2, .1, seed=5)
        self.assertEqual(first["selected_block_indices"], second["selected_block_indices"])
        self.assertNotEqual(first["selected_block_indices"], other["selected_block_indices"])
        after = np.random.get_state()
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])


if __name__ == "__main__":
    unittest.main()
