"""Depersin--Lecue covSDPofMeans (arXiv:1906.03058), Algorithm 4.

Backend 'ptz' uses the specialized positive SDP implementation; 'clarabel'
is an explicitly named general-solver control with the same outer routine.
K is supplied by the experiment: using small practical K does not confer
the paper's theorem. Balanced blocks retain all observations, unlike the
equal-block/discard convention in the paper. No hard thresholding.
"""
from time import perf_counter
import math
import numpy as np
import cvxpy as cp
from positive_sdp import covering_sdp


def block_means(data, K, seed):
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or min(data.shape) < 1 or not np.isfinite(data).all():
        raise ValueError('data must be a finite nonempty (n,d) array')
    if isinstance(K, (bool,np.bool_)) or not isinstance(K, (int,np.integer)) or not 1 <= K <= len(data):
        raise ValueError('K must be an integer in [1,n]')
    groups = np.array_split(np.random.default_rng(seed).permutation(len(data)), K)
    return np.array([data[g].mean(axis=0) for g in groups])


class GeneralCoveringSDP:
    """Numerical reference; primal and dual residuals checked explicitly."""
    def __init__(self, z, tol):
        self.z = z
        self.k, self.d = z.shape
        self.c = .9*self.k
        self.rho = cp.Parameter(nonneg=True)
        self.m = cp.Variable((self.d,self.d), PSD=True)
        self.y = cp.Variable(self.k, nonneg=True)
        self.values = cp.hstack([cp.sum(cp.multiply(np.outer(row,row), self.m)) for row in z])
        self.constraints = [self.rho*self.values+self.c*self.y >= 1]
        self.problem = cp.Problem(cp.Minimize(cp.trace(self.m)+cp.sum(self.y)), self.constraints)
        self.tol = min(tol/100,1e-8)

    def solve(self, rho, eta=1e-4):
        start = perf_counter()
        self.rho.value = rho
        self.problem.solve(solver='CLARABEL', tol_gap_abs=self.tol, tol_gap_rel=self.tol,
                           tol_feas=self.tol, max_iter=500, max_threads=1)
        if self.problem.status not in ('optimal','optimal_inaccurate'):
            raise RuntimeError('Covering SDP: '+self.problem.status)
        w, v = np.linalg.eigh((self.m.value+self.m.value.T)/2)
        m = (v*np.maximum(w,0)) @ v.T
        y = np.maximum(self.y.value,0)
        value = rho*np.einsum('ij,jk,ik->i',self.z,m,self.z)+self.c*y
        correction = max(1.,1/max(float(value.min()),1e-300))*(1+1e-12)
        m *= correction
        y *= correction
        packing = np.maximum(self.constraints[0].dual_value,0)
        norm = max(float(np.linalg.eigvalsh(rho*(self.z.T@(packing[:,None]*self.z)))[-1]),
                   float(np.max(self.c*packing)), 1.)
        packing /= norm*(1+1e-12)
        lower, upper = float(packing.sum()),float(np.trace(m)+y.sum())
        return dict(matrix=m, slack=y, lower=lower, upper=upper,
                    relative_gap=upper/max(lower,1e-300)-1,
                    converged=upper <= (1+eta)*lower, iterations=self.problem.solver_stats.num_iters,
                    solver_status=self.problem.status, runtime=perf_counter()-start)


def depersin_lecue_estimation(data, *, K, seed=None, tol=1e-5,
                             backend='ptz', sdp_eta=1e-4, sdp_max_iter=100_000,
                             progress=None):
    if backend not in ('ptz','clarabel'):
        raise ValueError('unknown backend')
    if tol <= 0 or not np.isfinite(tol):
        raise ValueError('tol must be positive')
    # The 0.9981 threshold in SolveSDP was derived with eta=0.0001.
    if not 0 < sdp_eta <= 1e-4:
        raise ValueError('paper SolveSDP requires sdp_eta <= 1e-4')
    start = perf_counter()
    blocks = block_means(data,K,seed)
    d = blocks.shape[1]
    initial = np.median(blocks,axis=0)
    x = initial.copy()
    spread = float(np.median(np.linalg.norm(blocks-initial,axis=1)))
    # Algorithm 1 and the squared-distance contraction proof give this
    # conservative bound; Algorithm 4 omits the square root in its denominator.
    max_outer = math.ceil(math.log(8*math.sqrt(d))/math.log(1/math.sqrt(.81)))
    max_bisect = math.ceil(math.log2(1e9*d**4))
    trace = []
    converged = True
    termination = 'outer_iteration_bound'
    if spread == 0:
        return initial, dict(converged=True, termination='zero_median_radius',
                             runtime=perf_counter()-start,K=K,backend=backend,iterations=0,
                             theorem_conditions_verified=False,history=[])
    for outer in range(max_outer):
        z = blocks-x
        generic = GeneralCoveringSDP(z,tol) if backend == 'clarabel' else None
        def solve(rho):
            result = (generic.solve(rho, sdp_eta) if generic is not None else
                      covering_sdp(z,rho,eta=sdp_eta,max_iter=sdp_max_iter,
                                   callback=progress))
            item = {key: val for key,val in result.items() if key not in ('matrix','slack')}
            item.update(outer=outer,rho=float(rho))
            trace.append(item)
            return result
        high = d/spread**2
        low = 0.
        result = solve(high)
        if not result['converged']:
            converged,termination=False,'sdp_iteration_limit_or_gap'
            break
        accepted = .9981 <= result['upper'] <= 1.
        for _ in range(max_bisect):
            if accepted:
                break
            middle = (low+high)/2
            result = solve(middle)
            if not result['converged']:
                break
            value = result['upper']
            if .9981 <= value <= 1.:
                accepted = True
                break
            if value < .9981:
                high = middle
            else:
                low = middle
        if not result['converged']:
            converged,termination=False,'sdp_iteration_limit_or_gap'
            break
        if not accepted:
            result = solve(1/(d*spread)**2)
            if not result['converged']:
                converged,termination=False,'sdp_iteration_limit_or_gap'
            else:
                if result['upper'] < 1:
                    x = initial.copy()
                    termination='initial_center_certificate'
                else:
                    termination='current_center_certificate'
            break
        m = result['matrix']
        if np.trace(m) <= 0:
            converged,termination=False,'zero_sdp_matrix'
            break
        _,v=np.linalg.eigh(m)
        direction=v[:,-1]
        step=float(np.median(z@direction))
        x=x+step*direction
        if progress:
            progress(dict(outer=outer+1,step=step,elapsed=perf_counter()-start))
    return x, dict(converged=converged,termination=termination,K=K,backend=backend,
                   runtime=perf_counter()-start,iterations=outer+1,history=trace,
                   sdp_calls=len(trace),sdp_iterations=sum(t['iterations'] for t in trace),
                   tol=tol,sdp_eta=sdp_eta,sdp_max_iter=sdp_max_iter,
                   max_outer=max_outer,max_bisect=max_bisect,
                   theorem_conditions_verified=False,
                   nearly_linear_runtime_claim=False)
