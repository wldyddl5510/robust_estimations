"""Check sampled-block semantics, original-partition reuse, and solver integration."""
import contextlib
import io
import unittest
from unittest import mock

import numpy as np

import experiments
import random_block_selection as sampled
from IP_algorithm import ip_estimation, ip_estimation_from_block_means
from utils import mom_initialization


class RandomBlockSelectionTests(unittest.TestCase):
    def test_original_partition_and_independent_draws_reach_solver_unchanged(self):
        data = np.arange(1500., dtype=float).reshape(300, 5)
        original = data.copy()
        result = (np.zeros(5), {"converged": True, "runtime": 0.})
        with mock.patch.object(sampled, "ip_estimation_from_block_means", return_value=result) as solve:
            _, info = sampled.random_block_selection(data, 2, .1, 6, seed=3)
        self.assertEqual(info["K"], 61)
        self.assertEqual(info["K_sub"], 13)
        blocks = np.array_split(np.random.default_rng(3).permutation(300), 61)
        means = np.array([data[b].mean(axis=0) for b in blocks])
        selected = np.array(info["selected_block_indices"])
        np.testing.assert_array_equal(solve.call_args.args[0], means[selected])
        self.assertEqual(solve.call_args.args[1], 2)
        np.testing.assert_array_equal(data, original)
        self.assertEqual(solve.call_args.kwargs["tol"], 1e-5)
        self.assertFalse(info["sampling_with_replacement"])
        self.assertEqual(len(set(selected)), info["K_sub"])
        self.assertGreaterEqual(info["runtime"], 0.)

    def test_full_support_uses_dense_complexity_for_both_block_counts(self):
        data = np.arange(1000., dtype=float).reshape(100, 10)
        with mock.patch.object(sampled, "ip_estimation_from_block_means",
                               return_value=(np.zeros(10), {})) as solve:
            _, info = sampled.random_block_selection(data, 10, .01, 6, seed=4)
        self.assertEqual(info["K"], 21)
        self.assertEqual(info["K_sub_requested"], 29)
        self.assertEqual(info["K_sub"], 21)
        self.assertEqual(info["selected_block_indices"], list(range(21)))
        groups = np.array_split(np.random.default_rng(4).permutation(100), 21)
        expected = np.array([data[group].mean(axis=0) for group in groups])
        np.testing.assert_array_equal(solve.call_args.args[0], expected)

    def test_formula_for_benchmark_dimensions(self):
        for d, expected in [(5, 13), (10, 17), (20, 19)]:
            with self.subTest(d=d), mock.patch.object(
                sampled, "ip_estimation_from_block_means", return_value=(np.zeros(d), {})
            ):
                _, info = sampled.random_block_selection(np.ones((100, d)), 2, .1, 6, seed=0)
                self.assertEqual(info["K"], 21)
                self.assertEqual(info["K_sub"], expected)

    def test_multiplier_three_extends_draws_without_changing_original_blocks(self):
        for d, expected in [(5, 19), (10, 23), (20, 27)]:
            with self.subTest(d=d), mock.patch.object(
                sampled, "ip_estimation_from_block_means", side_effect=lambda *a, **kw: (np.zeros(d), {})
            ) as solve:
                data = np.arange(100*d, dtype=float).reshape(100, d)
                _, old = sampled.random_block_selection(data, 2, .1, 6, seed=7, replace=True)
                _, new = sampled.random_block_selection(data, 2, .1, 6, seed=7, subsample_multiplier=3, replace=True)
                self.assertEqual(new["K"], old["K"])
                self.assertEqual(new["K_sub"], expected)
                self.assertEqual(new["subsample_multiplier"], 3.)
                self.assertEqual(new["selected_block_indices"][:old["K_sub"]], old["selected_block_indices"])
                np.testing.assert_array_equal(solve.call_args_list[1].args[0][:old["K_sub"]], solve.call_args_list[0].args[0])

    def test_capped_subset_uses_all_original_blocks_and_matches_full_algorithm(self):
        data = np.random.default_rng(2).standard_t(5, size=(35, 3)) + [1, 0, 0]
        expected, full = ip_estimation(data, 1, 0, 10/3, delta=.4, tol=.01, seed=4)
        estimate, info = sampled.random_block_selection(data, 1, 0, 10/3, delta=.4, tol=.01, seed=4)
        self.assertGreater(info["K_sub_requested"], info["K"])
        self.assertEqual(info["K_sub"], info["K"])
        self.assertEqual(info["K_sub"] % 2, 1)
        self.assertEqual(info["selected_block_indices"], list(range(full["K"])))
        np.testing.assert_allclose(estimate, expected, atol=1e-8)
        self.assertAlmostEqual(info["objective"], full["objective"], places=8)

    def test_larger_without_replacement_subset_is_nested_and_capped(self):
        data = np.arange(1000., dtype=float).reshape(100, 10)
        with mock.patch.object(sampled, "ip_estimation_from_block_means", side_effect=lambda *a, **kw: (np.zeros(10), {})):
            _, small = sampled.random_block_selection(data, 2, .1, 6, seed=4)
            _, large = sampled.random_block_selection(data, 2, .1, 6, seed=4, subsample_multiplier=3)
        self.assertEqual(small["K_sub"], 17)
        self.assertEqual(large["K_sub_requested"], 23)
        self.assertEqual(large["K_sub"], 21)
        self.assertEqual(large["unique_selected_blocks"], 21)
        self.assertTrue(set(small["selected_block_indices"]).issubset(large["selected_block_indices"]))

    def test_invalid_multiplier_fails_before_block_construction(self):
        with mock.patch.object(sampled, "mom_initialization") as initialize:
            for value in [0, -1, float("nan"), float("inf"), True, "3"]:
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "subsample_multiplier"):
                    sampled.random_block_selection(np.ones((100, 5)), 2, .1, 6, subsample_multiplier=value)
            initialize.assert_not_called()

    def test_reproducibility_selection_seed_and_global_rng_isolation(self):
        data = np.random.default_rng(0).normal(size=(200, 10))
        np.random.seed(123)
        state = np.random.get_state()
        with mock.patch.object(sampled, "ip_estimation_from_block_means", side_effect=lambda *a, **kw: (np.zeros(10), {})) as solve:
            _, a = sampled.random_block_selection(data, 2, .1, 6, seed=4)
            _, b = sampled.random_block_selection(data, 2, .1, 6, seed=4)
            _, c = sampled.random_block_selection(data, 2, .1, 6, seed=4, block_selection_seed=99)
        self.assertEqual(a["selected_block_indices"], b["selected_block_indices"])
        self.assertNotEqual(a["selected_block_indices"], c["selected_block_indices"])
        blocks, *_ = mom_initialization(data, 2, .1, 6, seed=4)
        np.testing.assert_array_equal(solve.call_args_list[2].args[0], blocks[c["selected_block_indices"]])
        after = np.random.get_state()
        self.assertEqual(state[0], after[0]); np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])

    def test_replacement_preserves_duplicates_even_when_k_sub_exceeds_k(self):
        data = np.arange(35., dtype=float)[:, None]
        estimate, info = sampled.random_block_selection(data, 1, 0, 6, seed=4, tol=None, replace=True)
        self.assertEqual(info["K"], 7)
        self.assertEqual(info["K_sub"], 11)
        self.assertLess(info["unique_selected_blocks"], info["K_sub"])
        blocks, *_ = mom_initialization(data, 1, 0, 6, seed=4)
        expected = np.median(blocks[info["selected_block_indices"]], axis=0)
        np.testing.assert_allclose(estimate, expected, atol=1e-7)
        self.assertTrue(info["converged"])
        self.assertEqual(info["tol"], 1e-5)
        self.assertLessEqual(info["gap"], 1e-5)

    def test_subset_center_and_bounds_ignore_unselected_blocks(self):
        # Selecting only positive rows must not initialize at the full collection's zero median.
        data = np.arange(100., dtype=float)[:, None]
        with mock.patch.object(sampled, "mom_initialization", return_value=(
            np.array([[-100.], [0.], [7.]]), np.array([0.]), np.ones(1), np.array([0.]), 1e-5
        )), mock.patch.object(sampled.np.random, "default_rng") as rng:
            rng.return_value.integers.return_value = np.full(11, 2)
            estimate, info = sampled.random_block_selection(data, 1, 0, 6, replace=True)
        np.testing.assert_allclose(estimate, [7.], atol=1e-7)
        self.assertAlmostEqual(info["objective"], 0, places=7)

    def test_direct_block_solver_matches_algorithm1(self):
        data = np.random.default_rng(2).standard_t(5, size=(35, 3)) + [1, 0, 0]
        estimate, info = ip_estimation(data, 1, 0, 10/3, delta=.4, tol=.01, seed=4)
        means, *_ = mom_initialization(data, 1, 0, 10/3, delta=.4, tol=.01, seed=4)
        direct, direct_info = ip_estimation_from_block_means(means, 1, tol=.01)
        np.testing.assert_allclose(estimate, direct, atol=1e-8)
        self.assertAlmostEqual(info["objective"], direct_info["objective"], places=8)
        self.assertEqual(info["K"], direct_info["K"])

    def test_rejects_invalid_block_solver_inputs(self):
        for means, s, tol in [(np.ones((4, 2)), 1, 1e-5), (np.ones((0, 2)), 1, 1e-5),
                              (np.array([[np.nan]]), 1, 1e-5), (np.ones((3, 2)), 3, 1e-5),
                              (np.ones((3, 2)), 1, 0)]:
            with self.subTest(shape=means.shape, s=s, tol=tol), self.assertRaises(ValueError):
                ip_estimation_from_block_means(means, s, tol=tol)

    def test_experiment_and_cli_forward_selection_seed(self):
        with mock.patch.object(experiments, "random_block_selection", return_value=(np.zeros(10), {"converged":True,"runtime":.1})) as solve:
            metrics = experiments.run_experiment(100, .1, 3, 2, d=10, methods=["random_block_selection"], block_selection_seed=27, subsample_multiplier=2)
        self.assertEqual(set(metrics), {"random_block_selection"})
        self.assertEqual(solve.call_args.kwargs["block_selection_seed"], 27)
        self.assertEqual(solve.call_args.kwargs["subsample_multiplier"], 2)
        self.assertFalse(solve.call_args.kwargs["replace"])
        self.assertEqual(solve.call_args.args[1:], (2, .1, 6., .05))
        args=["experiments.py", "--n", "100", "--d", "10", "--s", "2", "--nu", "3", "--epsilon", ".1", "--methods", "random_block_selection", "--block-selection-seed", "27", "--subsample-multiplier", "2", "--block-selection-with-replacement"]
        with mock.patch("sys.argv", args), mock.patch.object(experiments, "run_experiment", return_value={}) as run, contextlib.redirect_stdout(io.StringIO()):
            experiments.main()
        self.assertEqual(run.call_args.kwargs["methods"], ["random_block_selection"])
        self.assertEqual(run.call_args.kwargs["block_selection_seed"], 27)
        self.assertEqual(run.call_args.kwargs["subsample_multiplier"], 2)
        self.assertTrue(run.call_args.kwargs["block_selection_with_replacement"])


if __name__ == "__main__":
    unittest.main()
