import contextlib
import io
import unittest
from unittest import mock

import numpy as np

import experiments


class RandomExperimentTests(unittest.TestCase):
    def test_random_methods_forward_distinct_signatures(self):
        result = (np.zeros(3), {"converged": True, "runtime": 0.1})
        with mock.patch.object(experiments, "random_mom_estimation", return_value=result) as mom, \
                mock.patch.object(experiments, "random_trimmed_estimation", return_value=result) as trimmed:
            metrics = experiments.run_experiment(
                50, 0.02, 5, 1, d=3, seed=12, C=3, tol=0.002,
                methods=["random_mom", "random_trimmed"], tilde_J=23, direction_seed=7,
            )
        self.assertEqual(set(metrics), {"random_mom", "random_trimmed"})
        np.testing.assert_array_equal(mom.call_args.args[0], trimmed.call_args.args[0])
        self.assertEqual(mom.call_args.args[1:], (1, 0.02, 2 * 5 / 3, 0.05))
        self.assertEqual(trimmed.call_args.args[1:], (1, 0.02, 0.05))
        self.assertEqual(mom.call_args.kwargs, {
            "tol": 0.002, "seed": 12, "C": 3, "tilde_J": 23, "direction_seed": 7,
        })
        self.assertEqual(trimmed.call_args.kwargs, {
            "tol": 0.002, "seed": 12, "tilde_J": 23, "direction_seed": 7,
        })

    def test_random_methods_reject_invalid_count_before_data_generation(self):
        with mock.patch.object(experiments, "sparse_t_dist_data_generation") as generate:
            for method in ("random_mom", "random_trimmed"):
                for count in (None, 0, -2, 2.5, True, np.bool_(True)):
                    with self.subTest(method=method, count=count):
                        with self.assertRaisesRegex(ValueError, "positive integer tilde_J"):
                            experiments.run_experiment(
                                50, 0.02, 5, 1, d=3, methods=[method], tilde_J=count,
                            )
            generate.assert_not_called()

    def test_default_methods_are_unchanged(self):
        result = (np.zeros(3), {"converged": True, "runtime": 0.1})
        names = ("brute_force_estimation", "ip_estimation", "coordinate_mom_estimation",
                 "geometric_mom_estimation", "sample_mean_ht_estimation")
        with contextlib.ExitStack() as stack:
            for name in names:
                stack.enter_context(mock.patch.object(experiments, name, return_value=result))
            random_mom = stack.enter_context(mock.patch.object(experiments, "random_mom_estimation"))
            random_trimmed = stack.enter_context(mock.patch.object(experiments, "random_trimmed_estimation"))
            metrics = experiments.run_experiment(50, 0.02, 5, 1, d=3)
        self.assertEqual(set(metrics), {
            "brute_force", "algorithm_1", "coordinate_mom", "geometric_mom", "sample_mean",
        })
        random_mom.assert_not_called()
        random_trimmed.assert_not_called()

    def test_sample_mean_ht_uses_magnitude_and_stable_ties(self):
        data = np.array([[6., -10., 6., 0.], [4., -8., 4., 2.]])
        estimate, info = experiments.sample_mean_ht_estimation(data, 2)
        np.testing.assert_array_equal(estimate, [5., -9., 0., 0.])
        self.assertTrue(info["converged"])
        self.assertGreaterEqual(info["runtime"], 0)

    def test_sample_mean_metrics_use_thresholded_estimate(self):
        data = np.array([[1., 5., 0.], [3., 7., 0.]])
        truth = np.array([2., 0., 0.])
        with mock.patch.object(experiments, "sparse_t_dist_data_generation", return_value=(data, truth)):
            metrics = experiments.run_experiment(2, 0, 5, 1, d=3, methods=["sample_mean"])
        self.assertAlmostEqual(metrics["sample_mean"]["error"], np.sqrt(40))
        self.assertEqual(metrics["sample_mean"]["support_recovery"], 0.)

    def test_cli_parses_random_options(self):
        argv = ["experiments.py", "--n", "50", "--epsilon", "0.02", "--nu", "5",
                "--s", "1", "--d", "3", "--methods", "random_mom", "random_trimmed",
                "--tilde-J", "23", "--direction-seed", "7"]
        with mock.patch("sys.argv", argv), \
                mock.patch.object(experiments, "run_experiment", return_value={}) as run, \
                contextlib.redirect_stdout(io.StringIO()):
            experiments.main()
        self.assertEqual(run.call_args.kwargs["methods"], ["random_mom", "random_trimmed"])
        self.assertEqual(run.call_args.kwargs["tilde_J"], 23)
        self.assertEqual(run.call_args.kwargs["direction_seed"], 7)


if __name__ == "__main__":
    unittest.main()
