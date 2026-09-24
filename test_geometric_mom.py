import unittest

import numpy as np

from experiments import geometric_mom_estimation


class GeometricMomTests(unittest.TestCase):
    def test_geometric_median_then_hard_threshold(self):
        data = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        estimate, info = geometric_mom_estimation(
            data, 1, 0, 6, 0.5, seed=0, C=3,
        )
        np.testing.assert_allclose(estimate, [0, 1 / np.sqrt(3)], atol=1e-7)
        self.assertEqual(info['K'], 3)


if __name__ == '__main__':
    unittest.main()
