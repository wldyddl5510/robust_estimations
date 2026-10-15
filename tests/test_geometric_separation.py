"""Geometric median separation: independent references and certificate edge cases.

The sampling checks below are regression checks, not proofs of certification.
Sound certification comes from the module's exact rational upper calculations.
"""

import itertools
import math
from fractions import Fraction
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import geometric_separation as geo
import IP_algorithm as ip


def exact_planar_median_reference(residuals):
    """Independent finite candidate formula for the maximum planar median.

    Between pairwise score crossings the median is one fixed row's sinusoid.
    Its maximum is at a crossing or a stationary point of that row's sinusoid.
    This reference uses floats, and tests keep a generous margin from its value.
    """
    residuals = np.asarray(residuals, dtype=float)
    candidates = [np.array([1., 0.]), np.array([0., 1.])]
    for row in residuals:
        norm = np.linalg.norm(row)
        if norm:
            candidates.append(row / norm)
    for first, second in itertools.combinations(residuals, 2):
        difference = first - second
        norm = np.linalg.norm(difference)
        if norm:
            candidates.append(np.array([-difference[1], difference[0]]) / norm)
    candidates += [-direction for direction in candidates.copy()]
    return max(float(np.median(residuals @ direction)) for direction in candidates)


class GeometricAssertions(unittest.TestCase):
    def assert_valid_cut(self, result, residuals, q, threshold):
        self.assertEqual(result.status, "CUT", result.reason)
        direction = np.asarray(result.direction)
        self.assertEqual(direction.shape, (residuals.shape[1],))
        self.assertTrue(np.isfinite(direction).all())
        self.assertLessEqual(np.count_nonzero(direction), q)
        self.assertLessEqual(np.linalg.norm(direction), 1. + 1e-12)
        actual = float(np.median(residuals @ direction))
        self.assertGreater(actual, threshold)
        self.assertLessEqual(result.lower, actual + 1e-12 * max(1., abs(actual)))


class CircleSweepTests(GeometricAssertions):
    def test_empty_circle_collection(self):
        count, _ = geo.circular_max_overlap([])
        self.assertEqual(count, 0)

    def test_wrap_around_intervals_overlap(self):
        count, theta = geo.circular_max_overlap([(0., .25), (2 * math.pi - .1, .25)])
        self.assertEqual(count, 2)
        for center in (0., 2 * math.pi - .1):
            distance = abs(math.remainder(theta - center, 2 * math.pi))
            self.assertLessEqual(distance, .25 + 1e-12)

    def test_closed_endpoints_are_counted_before_removal(self):
        # The only common point of these two arcs is theta=1.
        count, theta = geo.circular_max_overlap([(0., 1.), (2., 1.)])
        self.assertEqual(count, 2)
        self.assertAlmostEqual(theta, 1.)

    def test_full_circles_and_singleton(self):
        count, theta = geo.circular_max_overlap([(0., math.pi), (1., math.pi), (2., 0.)])
        self.assertEqual(count, 3)
        self.assertAlmostEqual(theta, 2.)

    def test_at_x_handles_offsets_and_zero_plane_radius(self):
        # At x=(.6,0), rho=.8. The first block always qualifies,
        # the second never does; the last needs .8*cos(theta) >= .5.
        residuals = np.array([[0., 0., 2., 0.],
                              [0., 0., -2., 0.],
                              [1., 0., 0., 0.]])
        count, theta = geo.circle_at_x(residuals, np.array([.6, 0.]), .5)
        self.assertEqual(count, 2)
        direction = np.array([.8 * math.cos(theta), .8 * math.sin(theta), .6, 0.])
        self.assertGreaterEqual(np.count_nonzero(residuals @ direction >= .5 - 1e-12), count)

    def test_at_x_on_ball_boundary_is_an_offset_only_decision(self):
        residuals = np.array([[100., -40., 2., 0.],
                              [-20., 100., 1., 0.],
                              [30., 20., -2., 0.]])
        count, _ = geo.circle_at_x(residuals, np.array([1., 0.]), .5)
        self.assertEqual(count, 2)

    def test_interval_interiors_find_a_strict_cut(self):
        residuals = np.array([[1., 0.], [0., 1.], [0., 0.]])
        result = geo.certify_support(residuals, .7, .8, time_limit=5.)
        self.assert_valid_cut(result, residuals, 2, .7)

    def test_threshold_tangency_does_not_become_a_strict_cut(self):
        residuals = np.array([[1., 0.], [1., 0.], [0., 0.]])
        result = geo.fast_plane_violation_search(residuals, 2, 1., time_limit=5.)
        self.assertNotEqual(result.status, "CUT")


class NodeUpperBoundTests(GeometricAssertions):
    def test_infeasible_box_is_identified(self):
        residuals = np.ones((3, 4))
        self.assertEqual(geo.node_upper_count(residuals, np.array([.8, .8]),
                                              np.array([1., 1.]), .5), -1)

    def test_boundary_box_is_not_declared_infeasible(self):
        residuals = np.array([[0., 0., 1., 0.], [0., 0., 1., 0.], [0., 0., -1., 0.]])
        upper = geo.node_upper_count(residuals, np.array([1., 0.]), np.array([1., 0.]), 1.)
        self.assertGreaterEqual(upper, 2)

    def test_node_bound_dominates_sampled_q4_directions(self):
        rng = np.random.default_rng(14023)
        theta = np.linspace(0., 2 * math.pi, 97, endpoint=False)
        for trial in range(12):
            residuals = rng.normal(size=(7, 4))
            center = rng.uniform(-.6, .6, size=2)
            half_width = rng.uniform(.02, .35, size=2)
            lo, hi = center - half_width, center + half_width
            threshold = float(rng.uniform(.25, 1.25))
            upper = geo.node_upper_count(residuals, lo, hi, threshold)
            points = np.vstack([rng.uniform(lo, hi, size=(30, 2)), center,
                                np.asarray(list(itertools.product(*zip(lo, hi))))])
            points = points[np.linalg.norm(points, axis=1) <= 1.]
            with self.subTest(trial=trial):
                self.assertGreaterEqual(upper, 0)
                self.assertLessEqual(upper, len(residuals))
                for point in points:
                    rho = math.sqrt(max(0., 1. - float(point @ point)))
                    directions = np.column_stack([rho * np.cos(theta), rho * np.sin(theta),
                                                  np.tile(point, (len(theta), 1))])
                    counts = np.sum(residuals @ directions.T >= threshold, axis=0)
                    self.assertGreaterEqual(upper, int(counts.max()))

    def test_exact_threshold_equality_is_retained(self):
        residuals = np.array([[1., 0.], [1., 0.], [0., 0.]])
        for threshold in (np.nextafter(1., 0.), 1.):
            with self.subTest(threshold=threshold):
                self.assertGreaterEqual(geo.node_upper_count(residuals, np.empty(0),
                                                              np.empty(0), threshold), 2)

    def test_adjacent_float_above_all_row_norms_prunes(self):
        residuals = np.array([[1., 0.], [1., 0.], [0., 0.]])
        threshold = np.nextafter(1., math.inf)
        self.assertEqual(geo.node_upper_count(residuals, np.empty(0), np.empty(0), threshold), 0)

    def test_half_angle_missing_point_is_counted(self):
        # At threshold one the only qualifying circle direction is (-1,0),
        # the point at infinity of the rational half-angle parameterization.
        residuals = np.array([[-1., 0.], [-1., 0.], [0., 0.]])
        self.assertEqual(geo.node_upper_count(residuals, np.empty(0), np.empty(0), 1.), 2)

    def test_adjacent_float_distinguishes_a_two_row_intersection(self):
        # max min(v_1, .75*v_2) = 3/5. Float .6 lies just below that rational;
        # its next representable successor lies above it. Both row norms exceed T.
        residuals = np.array([[1., 0.], [0., .75], [0., 0.]])
        self.assertEqual(geo.node_upper_count(residuals, np.empty(0), np.empty(0), .6), 2)
        self.assertEqual(geo.node_upper_count(residuals, np.empty(0), np.empty(0),
                                               np.nextafter(.6, math.inf)), 1)


class GlobalGeometricTests(GeometricAssertions):
    def test_q1_searches_both_signs_and_all_coordinates(self):
        residuals = np.array([[-.2, -3., 0., 0.], [.1, -2., 0., 0.], [.2, 1., 0., 0.]])
        result = geo.global_geometric_oracle(residuals, 1, 1.5, 1.7, time_limit=5.)
        self.assert_valid_cut(result, residuals, 1, 1.5)

    def test_q1_equality_is_not_certified(self):
        residuals = np.array([[1.], [1.], [0.]])
        result = geo.global_geometric_oracle(residuals, 1, .9, 1., search_cuts=False, time_limit=5.)
        self.assertEqual(result.status, "UNKNOWN")

    def test_planar_decisions_match_independent_piecewise_median_maximum(self):
        rng = np.random.default_rng(238)
        for trial in range(8):
            residuals = rng.normal(size=(7, 2))
            maximum = exact_planar_median_reference(residuals)
            with self.subTest(trial=trial, maximum=maximum):
                self.assertGreater(maximum, .01)
                cut = geo.global_geometric_oracle(residuals, 2, .7 * maximum, .8 * maximum,
                                                   time_limit=5.)
                self.assert_valid_cut(cut, residuals, 2, .7 * maximum)
                cert = geo.global_geometric_oracle(residuals, 2, maximum + .05, maximum + .1,
                                                    time_limit=5.)
                self.assertEqual(cert.status, "CERTIFIED", cert.reason)

    def test_global_search_does_not_stop_after_certifying_first_support(self):
        # Any cut needs coordinates 2 AND 3: the first five size-two supports
        # have F<=.8, while the last support has F=sqrt(2)*.8.
        residuals = np.tile(np.array([0., 0., .8, .8]), (3, 1))
        result = geo.global_geometric_oracle(residuals, 2, 1., 1.05, time_limit=5.)
        self.assert_valid_cut(result, residuals, 2, 1.)
        self.assertEqual(tuple(np.flatnonzero(result.direction)), (2, 3))

    def test_zero_residuals_are_certified(self):
        residuals = np.zeros((5, 6))
        result = geo.global_geometric_oracle(residuals, 4, .1, .2, time_limit=5.)
        self.assertEqual(result.status, "CERTIFIED", result.reason)

    def test_q4_antipodal_blocks_certify_a_positive_threshold(self):
        # Paired projections and a zero block have median exactly zero.
        residuals = np.vstack([np.eye(4), -np.eye(4), np.zeros((1, 4))])
        result = geo.global_geometric_oracle(residuals, 4, .2, .3,
                                              time_limit=5., max_nodes=10000)
        self.assertEqual(result.status, "CERTIFIED", result.reason)

    def test_finite_budget_cannot_certify_unresolved_support(self):
        residuals = np.tile(np.ones(4), (3, 1))
        result = geo.global_geometric_oracle(residuals, 4, .1, .2,
                                              max_nodes=0, time_limit=5., search_cuts=False)
        self.assertEqual(result.status, "UNKNOWN")

    def test_two_threshold_overlap_can_validly_cut_or_certify(self):
        residuals = np.tile(np.array([1., 0.]), (3, 1))
        cut = geo.fast_plane_violation_search(residuals, 2, .9, time_limit=5.)
        self.assert_valid_cut(cut, residuals, 2, .9)
        cert = geo.global_geometric_oracle(residuals, 2, .9, 1.1,
                                            time_limit=5., search_cuts=False)
        self.assertEqual(cert.status, "CERTIFIED", cert.reason)

    def test_certifier_is_invariant_under_large_power_of_two_scaling(self):
        residuals = np.array([[1., 0.], [0., 1.], [0., 0.]])
        for scale in (2. ** -100, 1., 2. ** 100):
            with self.subTest(scale=scale):
                result = geo.global_geometric_oracle(residuals * scale, 2, .8 * scale, .9 * scale,
                                                       time_limit=5., search_cuts=False)
                self.assertEqual(result.status, "CERTIFIED", result.reason)


class GeometricIntegrationTests(unittest.TestCase):
    def make_solver(self, *, enabled=True, data=None):
        data = np.zeros((3, 3)) if data is None else np.asarray(data, dtype=float)
        mu = np.zeros(data.shape[1])
        initial_pair = ((0,), (0, 1))
        with mock.patch.object(ip.ConstraintGeneration, "upper_bound", return_value=10.), \
                mock.patch.object(ip.ConstraintGeneration, "heuristic",
                                  return_value=(0., initial_pair, mu)):
            solver = ip.ConstraintGeneration(data, 1, .4, None, mu, mu, sdp_time_limit=0.,
                                               geometric_time_limit=.25 if enabled else 0.)
        solver.upper_bound = mock.Mock(return_value=10.)
        solver.restricted = mock.Mock(return_value=(mu, 1., np.array([-2., -3., -4.])))
        solver.heuristic = mock.Mock(return_value=(0., initial_pair, mu))
        solver.oracle = mock.Mock(return_value=(1.05, 1.05, initial_pair))
        return solver

    def result(self, status, direction=None):
        return SimpleNamespace(status=status, direction=direction, stats={"nodes": 7, "supports": 2})

    def test_certificate_skips_exact_and_returns_threshold_not_count(self):
        solver = self.make_solver()
        solver.geometric = mock.Mock(return_value=self.result("CERTIFIED"))
        solver.oracle.side_effect = AssertionError("geometric certificate must skip exact MIP")
        _, lower, upper, gradient = solver(np.ones(3), certify=True)
        self.assertAlmostEqual(lower, 1.)
        self.assertAlmostEqual(upper, 1.1)
        np.testing.assert_array_equal(gradient, [-2., -3., -4.])
        self.assertAlmostEqual(solver.heuristic.call_args.args[1], 1.05)
        self.assertAlmostEqual(solver.geometric.call_args.args[1], 1.05)
        self.assertAlmostEqual(solver.geometric.call_args.args[2], 1.1)
        self.assertEqual(solver.stats["geometric_certificates"], 1)
        self.assertEqual(solver.stats["geometric_nodes"], 7)
        self.assertEqual(solver.stats["geometric_supports"], 2)
        solver.oracle.assert_not_called()

    def test_unknown_falls_back_to_exact(self):
        solver = self.make_solver()
        solver.geometric = mock.Mock(return_value=self.result("UNKNOWN"))
        _, _, upper, _ = solver(np.ones(3), certify=True)
        solver.oracle.assert_called_once()
        self.assertAlmostEqual(upper, 1.05)
        self.assertEqual(solver.stats["geometric_fallbacks"], 1)
        self.assertEqual(solver.stats["geometric_certificates"], 0)

    def test_disabled_option_preserves_existing_threshold_and_path(self):
        solver = self.make_solver(enabled=False)
        solver.geometric = mock.Mock(side_effect=AssertionError("geometry is disabled"))
        solver(np.ones(3), certify=True)
        solver.geometric.assert_not_called()
        solver.oracle.assert_called_once()
        self.assertAlmostEqual(solver.heuristic.call_args.args[1], 1.1)

    def test_deferred_visit_does_not_run_geometry(self):
        solver = self.make_solver()
        solver.geometric = mock.Mock(side_effect=AssertionError("deferred visit"))
        solver(np.ones(3), certify=False)
        solver.geometric.assert_not_called()
        solver.oracle.assert_not_called()
        self.assertEqual(solver.stats["deferred_supports"], 1)

    def test_valid_cut_adds_its_pair_and_continues_restricted_solve(self):
        solver = self.make_solver(data=[[0., 2., 0.], [0., 2., 0.], [0., 0., 0.]])
        solver.geometric = mock.Mock(side_effect=[self.result("CUT", np.array([0., 1., 0.])),
                                                  self.result("CERTIFIED")])
        solver.restricted.side_effect = [(np.zeros(3), 1., np.zeros(3)),
                                         (np.zeros(3), 2., np.zeros(3))]
        _, lower, upper, _ = solver(np.ones(3), certify=True)
        self.assertEqual(solver.pairs[-1], ((1,), (0, 1)))
        self.assertEqual(solver.stats["geometric_cuts"], 1)
        self.assertEqual(solver.restricted.call_count, 2)
        self.assertEqual(lower, 2.)
        self.assertAlmostEqual(upper, 2.1)
        solver.oracle.assert_not_called()

    def test_malformed_or_infeasible_cut_is_rechecked_before_adding(self):
        for direction in (np.zeros(3), np.ones(2), np.array([np.nan, 0., 0.]), np.ones(3)):
            with self.subTest(direction=direction):
                solver = self.make_solver(data=np.full((3, 3), 2.))
                solver.geometric = mock.Mock(return_value=self.result("CUT", direction))
                solver(np.ones(3), certify=True)
                solver.oracle.assert_called_once()
                self.assertEqual(solver.stats["geometric_cuts"], 0)
                self.assertEqual(len(solver.pairs), 1)

    def test_oversized_direction_is_normalized_before_witness_check(self):
        solver = self.make_solver(data=np.tile([0., .1, 0.], (3, 1)))
        solver.geometric = mock.Mock(return_value=self.result("CUT", np.array([0., 100., 0.])))
        solver(np.ones(3), certify=True)
        solver.oracle.assert_called_once()
        self.assertEqual(solver.stats["geometric_cuts"], 0)

    def test_duplicate_cut_falls_back_without_adding_again(self):
        solver = self.make_solver(data=[[2., 0., 0.], [2., 0., 0.], [0., 0., 0.]])
        solver.geometric = mock.Mock(return_value=self.result("CUT", np.array([1., 0., 0.])))
        solver(np.ones(3), certify=True)
        solver.oracle.assert_called_once()
        self.assertEqual(len(solver.pairs), 1)
        self.assertEqual(solver.stats["geometric_fallbacks"], 1)

    def test_trimmed_algorithm_keeps_median_geometry_disabled(self):
        from trimmed_IP_algorithm import TrimmedConstraintGeneration
        data, mu = np.zeros((5, 3)), np.zeros(3)
        with mock.patch.object(TrimmedConstraintGeneration, "upper_bound", return_value=10.), \
                mock.patch.object(TrimmedConstraintGeneration, "heuristic",
                                  return_value=(0., ((0,), (0, 1, 2, 3)), mu)):
            solver = TrimmedConstraintGeneration(data, 1, 1, .4, None, mu, mu)
        self.assertEqual(solver.geometric_time_limit, 0.)
        self.assertEqual(solver.sdp_time_limit, 0.)

    def test_residual_roundoff_tightens_thresholds_and_lowers_cut_bound(self):
        # All rounded residual coordinates equal -2**53; their exact errors
        # are 1/2, 1/4, 1/8. A two-sparse direction needs error E=3/4,
        # not the sum of all three coordinate errors.
        data = np.tile([.5, .25, .125], (3, 1))
        solver = self.make_solver(data=data)
        mu = np.full(3, 2. ** 53)
        original_cut, original_cert = 2. ** 53, 2. ** 53 + 16.
        original_lower = 2. ** 53 + 8.
        warm_start = np.array([1., 0., 0.])
        kernel_result = geo.GeometricResult("CUT", warm_start.copy(), original_lower)
        with mock.patch.object(geo, "geometric_separation", return_value=kernel_result) as kernel:
            result = solver.geometric(mu, original_cut, original_cert, warm_start)
        args, kwargs = kernel.call_args
        np.testing.assert_array_equal(args[0], np.full((3, 3), -(2. ** 53)))
        self.assertEqual(args[1], 2)
        error = Fraction(3, 4)
        self.assertGreaterEqual(Fraction(args[2]), Fraction(original_cut) + error)
        self.assertLessEqual(Fraction(args[3]), Fraction(original_cert) - error)
        self.assertEqual(args[2], original_cut + 2.)
        self.assertEqual(args[3], original_cert - 2.)
        self.assertEqual(result.stats["residual_roundoff_bound"], .75)
        self.assertLessEqual(Fraction(result.lower), Fraction(original_lower) - error)
        self.assertEqual(result.lower, original_lower - 2.)
        self.assertGreaterEqual(kwargs["time_limit"], 0.)
        self.assertLessEqual(kwargs["time_limit"], solver.geometric_time_limit)
        np.testing.assert_array_equal(kwargs["warm_start"], warm_start)

    def test_positive_threshold_gap_exhausted_by_roundoff_is_unknown(self):
        solver = self.make_solver(data=np.tile([.5, .25, .125], (3, 1)))
        mu = np.full(3, 2. ** 53)
        # The original gap is positive. Tightening each side by E=3/4
        # reverses its ordering, so the kernel must not receive these thresholds.
        with mock.patch.object(geo, "geometric_separation",
                               side_effect=AssertionError("no valid adjusted gap")) as kernel:
            result = solver.geometric(mu, 1., 2., None)
        self.assertEqual(result.status, "UNKNOWN")
        self.assertEqual(result.reason, "residual_roundoff_exhausted_threshold_gap")
        kernel.assert_not_called()


if __name__ == "__main__":
    unittest.main()
