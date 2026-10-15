"""SDP count certificates, including unsafe approximate-dual and fallback cases."""

import unittest
from unittest import mock

import numpy as np

import IP_algorithm as ip
try:
    import cvxpy as cp
    from sdp_separation import SDPSeparationCertifier, _dual_upper_bound
except ImportError:
    cp = None


def zero_duals(K, d):
    return [np.zeros((d + 1, d + 1)), 0., 0., 0., np.zeros(K), np.zeros(K),
            *[np.zeros((3, 3)) for _ in range(K)], np.zeros(K)]


@unittest.skipUnless(cp is not None, 'CVXPY is an optional SDP dependency')
class DualUpperBoundTests(unittest.TestCase):
    def test_zero_duals_require_stationarity_correction(self):
        # The dual constant alone is zero; all five blocks actually qualify.
        residuals = np.ones((5, 2))
        upper = _dual_upper_bound(residuals, 0.5, 2, zero_duals(5, 2))
        self.assertGreaterEqual(upper, 5.)

    def test_indefinite_dual_cannot_produce_false_certificate(self):
        residuals = np.ones((5, 1))
        duals = zero_duals(5, 1)
        duals[0][0, 0] = -100.
        upper = _dual_upper_bound(residuals, 1., 2, duals)
        self.assertGreaterEqual(upper, 5.)

    def test_arbitrary_duals_dominate_independent_feasible_witnesses(self):
        rng = np.random.default_rng(819)
        K, d, sparsity = 7, 3, 2
        residuals = rng.normal(size=(K, d))
        threshold = 0.7
        # Q=e_1 e_1^T, w=0 and p=q/(q+threshold^2), c=threshold*p
        # is a fractional feasible SDP solution, independent of the solver.
        q = residuals[:, 0] ** 2
        fractional_objective = np.sum(q / (q + threshold ** 2))
        rank_one_counts = [np.sum(residuals @ direction >= threshold)
                           for direction in np.vstack([np.eye(d), -np.eye(d)])]
        feasible_objective = max(fractional_objective, *rank_one_counts)
        for trial in range(12):
            # Include indefinite PSD multipliers and negative scalar multipliers,
            # emulating inaccurate or partially converged solver output.
            duals = zero_duals(K, d)
            matrix = rng.normal(size=(d + 1, d + 1))
            duals[0] = (matrix + matrix.T) / 2
            duals[1:4] = list(rng.normal(size=3))
            duals[4:6] = [rng.normal(size=K), rng.normal(size=K)]
            for k in range(K):
                matrix = rng.normal(size=(3, 3))
                duals[6 + k] = (matrix + matrix.T) / 2
            duals[-1] = rng.normal(size=K)
            with self.subTest(trial=trial):
                upper = _dual_upper_bound(residuals, threshold, sparsity, duals)
                self.assertGreaterEqual(upper, feasible_objective - 1e-12)

    def test_missing_or_nonfinite_dual_is_inconclusive(self):
        residuals = np.ones((3, 1))
        for bad in (None, np.nan, np.inf):
            duals = zero_duals(3, 1)
            duals[1] = bad
            with self.subTest(bad=bad):
                self.assertEqual(_dual_upper_bound(residuals, 1., 2, duals), np.inf)


@unittest.skipUnless(cp is not None and 'CLARABEL' in cp.installed_solvers(),
                     'CVXPY and Clarabel are optional SDP dependencies')
class SDPSolveTests(unittest.TestCase):
    def test_real_certificate_skips_exact_when_row_bound_is_too_large(self):
        # For three orthogonal block means, F(0)=1/sqrt(2) but the row-norm bound is 1.
        # The SDP can certify the intermediate threshold F(0)+0.2 without an exact MIP.
        from utils import gurobi_env
        means, mu = np.eye(3), np.zeros(3)
        with gurobi_env() as env, \
                mock.patch.object(ip, 'separation_oracle', side_effect=AssertionError('unexpected exact MIP')):
            solver = ip.ConstraintGeneration(means, 1, .8, env, mu, mu, sdp_time_limit=5.)
            _, lower, upper, _ = solver(mu, certify=True)
        self.assertAlmostEqual(lower, 1 / np.sqrt(2), places=6)
        self.assertAlmostEqual(upper, lower + .2)
        self.assertLess(upper, ip.deterministic_upper_bound(means, mu, 1))
        self.assertEqual(solver.stats['sdp_calls'], 1)
        self.assertEqual(solver.stats['sdp_certificates'], 1)
        self.assertEqual(solver.stats['sdp_fallbacks'], 0)
        self.assertEqual(solver.stats['oracle_calls'], 0)

    def test_nontrivial_count_certificate_and_mu_update(self):
        means = np.random.default_rng(7).normal(size=(7, 4))
        certifier = SDPSeparationCertifier(means, 1, time_limit=5., max_iter=100)
        upper = certifier.upper_bound(np.zeros(4), 2.)
        self.assertGreaterEqual(upper, 3.0898)
        self.assertLess(upper, 4.)
        # Updating mu must update the model. Every block now qualifies in e_1.
        shifted_mu = np.array([-10., 0., 0., 0.])
        shifted_upper = certifier.upper_bound(shifted_mu, 2.)
        self.assertGreaterEqual(shifted_upper, 7.)

    def test_zero_residuals_certify_positive_threshold(self):
        means = np.zeros((5, 2))
        certifier = SDPSeparationCertifier(means, 1, time_limit=5., max_iter=100)
        self.assertLess(certifier.upper_bound(np.zeros(2), 1.), 3.)

    def test_threshold_equality_is_not_certified(self):
        # Exactly h projections equal the threshold, so F equals the threshold.
        means = np.array([[1.], [1.], [1.], [-1.], [-1.]])
        certifier = SDPSeparationCertifier(means, 1, time_limit=5., max_iter=100)
        self.assertGreaterEqual(certifier.upper_bound(np.zeros(1), 1.), 3.)

    def test_nonpositive_threshold_is_inconclusive(self):
        certifier = SDPSeparationCertifier(np.zeros((3, 2)), 1, time_limit=5.)
        for threshold in (0., -1.):
            with self.subTest(threshold=threshold):
                self.assertGreaterEqual(certifier.upper_bound(np.zeros(2), threshold), 3.)

    def test_solver_failure_is_inconclusive(self):
        certifier = SDPSeparationCertifier(np.zeros((3, 2)), 1, time_limit=5.)
        with mock.patch.object(certifier.problem, 'solve', side_effect=cp.error.SolverError('test failure')):
            self.assertEqual(certifier.upper_bound(np.zeros(2), 1.), np.inf)

    def test_limited_solve_does_not_reuse_previous_duals(self):
        means = np.random.default_rng(7).normal(size=(7, 4))
        certifier = SDPSeparationCertifier(means, 1, time_limit=5., max_iter=100)
        self.assertLess(certifier.upper_bound(np.zeros(4), 2.), 4.)

        def limited_solve(**kwargs):
            certifier.problem._status = cp.USER_LIMIT
            return 0.

        with mock.patch.object(certifier.problem, 'solve', side_effect=limited_solve):
            self.assertEqual(certifier.upper_bound(np.zeros(4), 2.), np.inf)

    def test_count_certificate_is_invariant_to_common_power_of_two_scale(self):
        means = np.random.default_rng(7).normal(size=(7, 4))
        bounds = []
        for scale in (2. ** -20, 1., 2. ** 20):
            certifier = SDPSeparationCertifier(means * scale, 1, time_limit=5., max_iter=100)
            bounds.append(certifier.upper_bound(np.zeros(4), 2. * scale))
        np.testing.assert_allclose(bounds, bounds[1], atol=1e-7, rtol=0.)
        self.assertLess(max(bounds), 4.)

    def test_primal_objective_is_never_used_as_count_upper_bound(self):
        means = np.random.default_rng(7).normal(size=(7, 4))
        certifier = SDPSeparationCertifier(means, 1, time_limit=5., max_iter=100)
        certifier.upper_bound(np.zeros(4), 2.)  # populate real solver duals
        with mock.patch.object(certifier.problem, 'solve', return_value=-1e6), \
                mock.patch.object(type(certifier.problem), 'value', new_callable=mock.PropertyMock,
                                  side_effect=AssertionError('primal objective was inspected')):
            upper = certifier.upper_bound(np.zeros(4), 2.)
        self.assertGreaterEqual(upper, 3.0898)
        self.assertLess(upper, 4.)


class SDPIntegrationTests(unittest.TestCase):
    def make_solver(self):
        means, mu = np.zeros((3, 2)), np.zeros(2)
        with mock.patch.object(ip.ConstraintGeneration, 'upper_bound', return_value=10.), \
                mock.patch.object(ip.ConstraintGeneration, 'heuristic',
                                  return_value=(0., ((0,), (0, 1)), mu)):
            solver = ip.ConstraintGeneration(means, 1, 0.4, None, mu, mu, sdp_time_limit=ip.SDP_TIME_LIMIT)
        solver.upper_bound = mock.Mock(return_value=10.)
        solver.restricted = mock.Mock(return_value=(mu, 1., np.array([-2., -3.])))
        solver.heuristic = mock.Mock(return_value=(0., solver.pairs[0], mu))
        return solver

    def test_sdp_screen_is_off_by_default(self):
        means, mu = np.zeros((3, 2)), np.zeros(2)
        with mock.patch.object(ip.ConstraintGeneration, 'heuristic', return_value=(0., ((0,), (0, 1)), mu)):
            self.assertEqual(ip.ConstraintGeneration(means, 1, 0.4, None, mu, mu).sdp_time_limit, 0.)

    def test_success_skips_exact_and_returns_threshold_as_F_upper(self):
        solver = self.make_solver()
        solver.sdp_certificate = mock.Mock(return_value=True)
        solver.oracle = mock.Mock(side_effect=AssertionError('exact separation should be skipped'))
        mu, lower, upper, gradient = solver(np.zeros(2), certify=True)
        solver.sdp_certificate.assert_called_once()
        self.assertAlmostEqual(float(solver.sdp_certificate.call_args.args[1]), 1.1)
        self.assertAlmostEqual(lower, 1.)
        self.assertAlmostEqual(upper, 1.1)
        np.testing.assert_array_equal(gradient, [-2., -3.])
        self.assertEqual(len(solver.pairs), 1)
        solver.oracle.assert_not_called()

    def test_failure_uses_exact_separation(self):
        solver = self.make_solver()
        solver.sdp_certificate = mock.Mock(return_value=False)
        solver.oracle = mock.Mock(return_value=(1.05, 1.05, solver.pairs[0]))
        _, lower, upper, _ = solver(np.zeros(2), certify=True)
        solver.sdp_certificate.assert_called_once()
        solver.oracle.assert_called_once()
        self.assertAlmostEqual(lower, 1.)
        self.assertAlmostEqual(upper, 1.05)
        self.assertEqual(solver.stats['oracle_calls'], 1)
        self.assertEqual(solver.stats['sdp_calls'], 1)
        self.assertEqual(solver.stats['sdp_certificates'], 0)
        self.assertEqual(solver.stats['sdp_fallbacks'], 1)

    def test_missing_dependency_is_inconclusive(self):
        import builtins
        original_import = builtins.__import__

        def missing_sdp(name, *args, **kwargs):
            if name == 'sdp_separation':
                raise ImportError('CVXPY is unavailable')
            return original_import(name, *args, **kwargs)

        solver = self.make_solver()
        with mock.patch('builtins.__import__', side_effect=missing_sdp):
            self.assertFalse(solver.sdp_certificate(np.zeros(2), 1.1))

    def test_count_bound_requires_strict_inequality_and_finiteness(self):
        solver = self.make_solver()
        for upper, expected in ((1.9, True), (2., False), (2.1, False),
                                (np.inf, False), (np.nan, False)):
            with self.subTest(upper=upper):
                solver._sdp_certifier = mock.Mock()
                solver._sdp_certifier.upper_bound.return_value = upper
                self.assertEqual(solver.sdp_certificate(np.zeros(2), 1.1), expected)

    def test_disabled_sdp_uses_exact_without_initializing_certifier(self):
        solver = self.make_solver()
        solver.sdp_time_limit = 0.
        solver.sdp_certificate = mock.Mock(side_effect=AssertionError('SDP is disabled'))
        solver.oracle = mock.Mock(return_value=(1.05, 1.05, solver.pairs[0]))
        solver(np.zeros(2), certify=True)
        solver.sdp_certificate.assert_not_called()
        solver.oracle.assert_called_once()
        self.assertIsNone(solver._sdp_certifier)

    def test_deferred_visit_does_not_call_sdp_or_exact(self):
        solver = self.make_solver()
        solver.sdp_certificate = mock.Mock(side_effect=AssertionError('SDP on deferred visit'))
        solver.oracle = mock.Mock(side_effect=AssertionError('exact oracle on deferred visit'))
        _, lower, upper, _ = solver(np.zeros(2), certify=False)
        self.assertAlmostEqual(lower, 1.)
        self.assertAlmostEqual(upper, 10.)
        self.assertEqual(solver.stats['deferred_supports'], 1)
        solver.sdp_certificate.assert_not_called()
        solver.oracle.assert_not_called()


if __name__ == '__main__':
    unittest.main()
