"""Independent numerical checks for the new dense estimators/SDP primitives."""
import unittest
import numpy as np
from scipy.linalg import expm
from positive_sdp import factorized_gibbs, covering_sdp, decision_psdp
from depersin_lecue import GeneralCoveringSDP, block_means, depersin_lecue_estimation
from cherapanamjeri_flammarion_bartlett import cfb_estimation, MeanTestSDP, estimate_distance


class DenseChecks(unittest.TestCase):
    def test_factorized_exponential_matches_full_matrix(self):
        rng=np.random.default_rng(17)
        for k,d in [(3,1),(7,4)]:
            z=rng.normal(size=(k,d));x=rng.uniform(.01,.1,k)
            rho,c,scale=.6,.9*k,1.3
            m,y,g,_=factorized_gibbs(z,x,rho,c,scale)
            full=np.zeros((k+d,k+d))
            full[:d,:d]=scale*rho*z.T@(x[:,None]*z)
            full[d:,d:]=np.diag(scale*c*x)
            density=expm(full);density/=np.trace(density)
            np.testing.assert_allclose(m,density[:d,:d],atol=1e-10)
            np.testing.assert_allclose(y,np.diag(density)[d:],atol=1e-10)

    def test_bounds_against_independent_sdp(self):
        z=np.random.default_rng(19).normal(size=(6,3))
        reference=GeneralCoveringSDP(z,1e-5).solve(.3)
        got=covering_sdp(z,.3,max_iter=20)
        self.assertLessEqual(got['lower'],reference['upper']+1e-7)
        self.assertGreaterEqual(got['upper'],reference['lower']-1e-7)
        vals=.3*np.einsum('ij,jk,ik->i',z,got['matrix'],z)+5.4*got['slack']
        self.assertGreaterEqual(vals.min(),1-1e-8)
        self.assertFalse(got['converged'])
        self.assertEqual(got['termination'],'iteration_limit')

    def test_zero_residual_covering_has_known_solution(self):
        z=np.zeros((4,2))
        got=covering_sdp(z,1,max_iter=10)
        self.assertAlmostEqual(got['upper'],10/9)
        self.assertAlmostEqual(got['lower'],10/9,places=9)
        self.assertTrue(got['converged'])

    def test_cfb_radius_scales_and_uses_point9K(self):
        # All buckets at +a in one dimension: SDP value=min(K,K*a^2/r^2),
        # hence radius threshold a/sqrt(.9), independently of K.
        z=np.full((5,1),2.)
        model=MeanTestSDP(z)
        low,high,_=estimate_distance(model,z,np.zeros(1),1e-5)
        exact=2/np.sqrt(.9)
        self.assertLessEqual(low,exact+2e-5)
        self.assertGreaterEqual(high,exact-2e-5)
        self.assertLess(high-low,1.0001e-5)

    def test_cfb_finite_iterations_on_identical_nonzero_blocks(self):
        # The one-dimensional SDP gives d_t=(a-x_t)/sqrt(.9), g_t=+1.
        # With five evaluated iterates, the best returned point is x_4.
        data=np.full((9,1),2.)
        estimate,info=cfb_estimation(data,K=3,tol=1e-4,max_iter=5)
        expected=2*(1-(1-.05/np.sqrt(.9))**4)
        np.testing.assert_allclose(estimate,[expected],atol=2e-5)
        self.assertEqual(info['iterations'],5)
        self.assertEqual(info['termination'],'requested_iterations_completed')
        self.assertFalse(info['outer_accuracy_certified'])

    def test_degenerate_inputs_and_balanced_blocks(self):
        data=np.full((11,3),2.5)
        for backend in ('ptz','clarabel'):
            estimate,info=depersin_lecue_estimation(data,K=5,backend=backend)
            np.testing.assert_allclose(estimate,2.5)
            self.assertTrue(info['converged'])
        estimate,info=cfb_estimation(np.zeros((9,3)),K=3)
        np.testing.assert_array_equal(estimate,np.zeros(3))
        self.assertTrue(info['converged'])
        with self.assertRaises(ValueError):
            block_means(data,12,0)
        with self.assertRaises(ValueError):
            depersin_lecue_estimation(data,K=5,sdp_eta=.01)


if __name__=='__main__':
    unittest.main()
