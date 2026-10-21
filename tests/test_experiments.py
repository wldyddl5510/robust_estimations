"""The experiment method registry."""

import unittest
from unittest import mock

import numpy as np

import experiments
from baselines import geometric_mom_estimation, sample_mean_ht_estimation


class RegistryTests(unittest.TestCase):
    data = np.random.default_rng(4).standard_t(5, size=(60, 4)) + [2, 0, 0, 0]

    def test_runs_estimators_with_the_shared_setting(self):
        estimate, _ = experiments.run_method("geometric_mom", self.data, s=1, epsilon=.05, seed=3, C=3)
        np.testing.assert_array_equal(estimate, geometric_mom_estimation(self.data, 1, .05, C=3, seed=3)[0])
        estimate, _ = experiments.run_method("sample_mean", self.data, s=2, epsilon=.05)
        np.testing.assert_array_equal(estimate, sample_mean_ht_estimation(self.data, 2)[0])

    def test_forwards_method_options(self):
        result = (np.zeros(4), {})
        with mock.patch.object(experiments, "haar_estimation", return_value=result) as haar, \
                mock.patch.object(experiments, "ip_estimation", return_value=result) as ip:
            experiments.run_method("haar", self.data, s=1, epsilon=.05, seed=3, r=2, J=5)
            experiments.run_method("block_subsampling", self.data, s=1, epsilon=.05, tol=1e-4)
        # Without tol each method keeps its own default.
        self.assertEqual(haar.call_args.kwargs, {"delta": .05, "C": 3, "seed": 3, "r": 2, "J": 5})
        self.assertTrue(ip.call_args.kwargs["subsample"])
        self.assertEqual(ip.call_args.kwargs["tol"], 1e-4)
        with self.assertRaises(TypeError):
            experiments.run_method("sample_mean", self.data, s=1, epsilon=0, r=2)
        with self.assertRaisesRegex(ValueError, "unknown method"):
            experiments.run_method("median", self.data, s=1, epsilon=0)

    def test_metrics(self):
        metrics = experiments.metrics(np.array([2., 0., 1e-9, 4.]), np.array([3., 1., 0., 4.]))
        self.assertAlmostEqual(metrics["error"], np.sqrt(2))
        self.assertAlmostEqual(metrics["support_recovery"], 2 / 3)
        self.assertAlmostEqual(metrics["f1"], 2 * 2 / (2 + 3))


if __name__ == "__main__":
    unittest.main()
