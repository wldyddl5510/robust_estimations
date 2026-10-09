"""CFB COLT 2019 Algorithm 1 with SDP Algorithms 4/5 (not MTE).

Practical K is explicit. T is an input, as in Algorithm 1. All iterates use
one fixed block partition and x0=0. Return the iterate with smallest
estimated distance, not the final iterate. No access to the true mean.
The outer loop has no optimization-gap certificate; completion of T
iterations is reported as such, not as proof of statistical accuracy.
"""
from time import perf_counter
import numpy as np
import cvxpy as cp
from depersin_lecue import block_means


class MeanTestSDP:
    def __init__(self,z,tol=1e-5):
        k,d=z.shape
        self.k,self.d=k,d
        self.centered=cp.Parameter((k,d))
        self.radius=cp.Parameter(nonneg=True)
        self.matrix=cp.Variable((1+k+d,1+k+d),PSD=True)
        a=self.matrix
        self.constraints=[a[0,0]==1,cp.diag(a)[1:1+k]==a[0,1:1+k],
                          cp.trace(a[1+k:,1+k:])==1,
                          cp.sum(cp.multiply(self.centered,a[1:1+k,1+k:]),axis=1)
                          >= self.radius*a[0,1:1+k]]
        self.problem=cp.Problem(cp.Maximize(cp.sum(a[0,1:1+k])),self.constraints)
        self.tol=min(tol/100,1e-8)
        self.calls=0
        self.max_residual=0.

    def solve(self,centered,radius):
        self.centered.value=centered
        self.radius.value=radius
        self.problem.solve(solver='CLARABEL',tol_gap_abs=self.tol,tol_gap_rel=self.tol,
                           tol_feas=self.tol,max_iter=500,max_threads=1)
        self.calls+=1
        if self.problem.status not in ('optimal','optimal_inaccurate'):
            raise RuntimeError('MT SDP: '+self.problem.status)
        a=(self.matrix.value+self.matrix.value.T)/2
        k=self.k
        b=a[0,1:k+1]
        residual=max(abs(a[0,0]-1),float(np.max(np.abs(np.diag(a)[1:k+1]-b))),
                     abs(np.trace(a[k+1:,k+1:])-1),
                     max(0.,-float(np.linalg.eigvalsh(a)[0])),
                     max(0.,float(np.max(radius*b-np.sum(centered*a[1:k+1,k+1:],axis=1)))))
        self.max_residual=max(self.max_residual,residual)
        if residual > max(1e-5,100*self.tol):
            raise RuntimeError(f'MT SDP residual too large: {residual}')
        return float(b.sum()),a


def estimate_distance(model,z,x,tol):
    centered=z-x
    # From PSD Cauchy--Schwarz: sum b_i <= sum ||z_i-x||^2/r^2.
    # This gives a data-dependent upper bracket valid for the SDP relaxation.
    high=float(np.sqrt(np.sum(centered**2)/(.9*len(z))))*(1+1e-6)
    if high <= tol:
        return 0.,0.,None
    low=0.
    best=None
    while high-low > tol:
        middle=(low+high)/2
        value,a=model.solve(centered,middle)
        if value >= .9*len(z):
            low,best=middle,a
        else:
            high=middle
    if best is None:
        return 0.,high,None
    return low,high,best


def cfb_estimation(data, *, K, seed=None, tol=1e-5, max_iter=100,
                   step_size=.05, progress=None):
    if tol <= 0 or not np.isfinite(tol) or not 0 < step_size < 1:
        raise ValueError('invalid tolerance or step size')
    if not isinstance(max_iter,(int,np.integer)) or max_iter < 1:
        raise ValueError('max_iter must be positive')
    start=perf_counter()
    z=block_means(data,K,seed)
    # Exact span reduction from Section 4.2; no random projection.
    _,sv,vt=np.linalg.svd(z,full_matrices=False)
    rank=int(np.sum(sv > (max(z.shape)*np.finfo(float).eps*(sv[0] if len(sv) else 0))))
    if rank == 0:
        return np.zeros(z.shape[1]),dict(converged=True,termination='all_zero_blocks',
            runtime=perf_counter()-start,K=K,sdp_calls=0,iterations=0,history=[])
    basis=vt[:rank].T
    reduced=z@basis
    model=MeanTestSDP(reduced,tol)
    x=np.zeros(rank)
    best=x.copy()
    best_distance=float('inf')
    history=[]
    termination='requested_iterations_completed'
    for iteration in range(max_iter):
        distance,upper,a=estimate_distance(model,reduced,x,tol)
        if distance < best_distance:
            best_distance=distance
            best=x.copy()
        item=dict(iteration=iteration,distance=distance,distance_upper=upper,
                  best_distance=best_distance,sdp_calls=model.calls,elapsed=perf_counter()-start)
        history.append(item)
        if progress:
            progress(item)
        if a is None or upper <= tol:
            termination='distance_within_tolerance'
            break
        _,v=np.linalg.eigh(a[K+1:,K+1:])
        g=v[:,-1]
        if np.count_nonzero((reduced-x)@g >= 0) < .9*K:
            g=-g
        x=x+step_size*distance*g
    return basis@best,dict(converged=True,termination=termination,runtime=perf_counter()-start,
        K=K,iterations=iteration+1,requested_iterations=max_iter,step_size=step_size,tol=tol,
        sdp_calls=model.calls,sdp_max_residual=model.max_residual,history=history,
        reduced_dimension=rank,best_distance=best_distance,
        distance_within_numerical_tolerance=termination=='distance_within_tolerance',
        outer_accuracy_certified=False,
        theorem_conditions_verified=False)
