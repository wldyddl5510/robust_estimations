"""PTZ (arXiv:1201.5135v3), Algorithm 3.1, for DL's covering SDP.

A_i = diag(rho*z_i*z_i.T, (0.9*K)*e_i*e_i.T).  The packing
problem is max sum(x) with sum(x_i*A_i) <= I.  The covering problem
is min trace(M)+sum(y) with rho*z_i.T*M*z_i + 0.9*K*y_i >= 1.

This is a reference implementation of the PTZ multiplicative updates, not
a call to a generic SDP optimizer. Exponentials use the factorized Taylor
primitive of PTZ Section 4. At small dimension an identity sketch avoids
JL sampling error. Dense d-by-d accumulation/certification is intentional:
this version benchmarks the actual updates at small d and DOES NOT claim
the paper's full high-dimensional nearly-linear implementation.

Every reported solution has independently computed packing/covering bounds.
Iteration-limit results are failures, never silently replaced by another solver.
"""

from time import perf_counter
import math
import numpy as np


def _validate(z, rho):
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or min(z.shape) < 1 or not np.isfinite(z).all():
        raise ValueError('z must be a nonempty finite (K,d) matrix')
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError('rho must be positive')
    return z


def factorized_gibbs(z, x, rho, c, scale=1., *, exp_tol=1e-10):
    """Normalized exp(sum x_i*scale*A_i), without a (K+d)^2 matrix.

The rank-one block is applied as Z.T @ (x[:,None]*(Z @ V)). The
diagonal block is evaluated analytically. Taylor degree follows PTZ Lemma
4.2. An identity sketch is exact before Taylor truncation; for the present
small-d reference implementation this is faster and reproducible.
"""
    d = z.shape[1]
    weights = scale*rho*x
    diag = scale*c*x
    # PSD trace is a safe operator-norm bound, with no eigenvalue oracle.
    trace_b = float(weights @ np.einsum('ij,ij->i', z, z))
    kappa = max(trace_b/2, 0.)
    degree = max(1, math.ceil(max(math.e**2*kappa, math.log(2/exp_tol))))
    if degree > 100_000 or max(trace_b, float(diag.max())) > 1200:
        raise ArithmeticError('Taylor primitive exceeds representable reference range')
    shift = max(trace_b, float(diag.max()))
    term = np.eye(d)*np.exp(-shift/2)
    factor = term.copy()
    for j in range(1, degree):
        term = (z.T @ (weights[:, None]*(z @ term))) / (2*j)
        factor += term
    m = factor @ factor.T
    y = np.exp(diag-shift)
    total = np.trace(m)+y.sum()
    if not np.isfinite(total) or total <= 0:
        raise ArithmeticError('Nonfinite Gibbs density')
    m /= total
    y /= total
    gradients = scale*(rho*np.einsum('ij,jk,ik->i', z, m, z)+c*y)
    return m, y, gradients, degree


def certificate(z, rho, c, x, m, y):
    """Feasible bounds for the original C_rho, also when PTZ is unfinished."""
    gram = rho*(z.T @ (x[:, None]*z))
    lam = max(float(np.linalg.eigvalsh(gram)[-1]), float(np.max(c*x)))
    packing = x/max(lam, np.finfo(float).tiny)
    # Inflate denominators slightly to make numerical feasibility conservative.
    packing /= 1+1e-12
    values = rho*np.einsum('ij,jk,ik->i', z, m, z)+c*y
    cover_scale = max(float(np.min(values)), np.finfo(float).tiny)/(1+1e-12)
    cover_m, cover_y = m/cover_scale, y/cover_scale
    lower, upper = float(packing.sum()), float(np.trace(cover_m)+cover_y.sum())
    return dict(lower=lower, upper=upper, packing=packing,
                matrix=cover_m, slack=cover_y,
                relative_gap=max(0., upper/lower-1) if lower > 0 else float('inf'))


def decision_psdp(z, rho, target, *, eta=1e-4, max_iter=100_000,
                  check_every=100, callback=None):
    """PTZ decision problem for target*A_i; exact paper update size.

Internal epsilon=eta/40 leaves room for the 10*epsilon decision loss.
Early exit is allowed only by explicit primal/dual certificates.
The finite cap is a diagnostic resource budget, not a paper stopping rule.
"""
    z = _validate(z, rho)
    if not 0 < eta < 1 or not np.isfinite(target) or target <= 0:
        raise ValueError('require 0 < eta < 1 and positive target')
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError('max_iter must be positive')
    start = perf_counter()
    k, d = z.shape
    c = .9*k
    eps = eta/40
    # PTZ's n is the number of packing coordinates/constraints, here K.
    log_size = math.log(max(k, 2))
    h = (1+log_size)/eps
    alpha = (eps/h)/(1+10*eps)
    theoretical_cap = math.ceil(32*log_size/(eps*alpha))
    traces = target*(rho*np.einsum('ij,ij->i', z, z)+c)
    x = 1/(k*traces)
    initial_sum = float(x.sum())
    minimum_growth_iterations = math.ceil(max(0., math.log(h/initial_sum))/math.log1p(alpha))
    sum_m, sum_y = np.zeros((d,d)), np.zeros(k)
    best = None
    history = []
    reason = 'iteration_limit'
    max_degree = 0
    for iteration in range(1, min(max_iter, theoretical_cap)+1):
        m, y, grad, degree = factorized_gibbs(z, x, rho, c, target,
                                             exp_tol=min(1e-10, eps/100))
        max_degree = max(max_degree, degree)
        sum_m += m
        sum_y += y
        if iteration == 1 or iteration % check_every == 0 or iteration == max_iter:
            cand = certificate(z, rho, c, x, sum_m/iteration, sum_y/iteration)
            if best is None:
                best = cand
            else:
                if cand['upper'] < best['upper']:
                    for key in ('upper', 'matrix', 'slack'):
                        best[key] = cand[key]
                if cand['lower'] > best['lower']:
                    best['lower'] = cand['lower']
                    best['packing'] = cand['packing']
            best['relative_gap'] = max(0., best['upper']/best['lower']-1)
            entry = dict(iteration=iteration, lower=best['lower'], upper=best['upper'],
                         relative_gap=best['relative_gap'], elapsed=perf_counter()-start)
            if len(history) < 2 or iteration % max(check_every, max_iter//10) == 0:
                history.append(entry)
            if callback:
                callback(entry)
            if best['upper'] <= target:
                reason = 'covering_certificate'
                break
            if best['lower'] >= target/(1+eta/2):
                reason = 'packing_certificate'
                break
        update = grad <= 1+eps
        # This is exactly PTZ Algorithm 3.1, line 6; no larger heuristic step.
        x[update] *= 1+alpha
        if float(x.sum()) > h:
            # Next certificate uses spectral normalization, stronger than the
            # paper's conservative (1+10*eps)*h normalization.
            cand = certificate(z, rho, c, x, sum_m/iteration, sum_y/iteration)
            if cand['lower'] >= target/(1+eta/2):
                best = cand
                reason = 'packing_certificate'
                break
    return dict(**best, converged=reason != 'iteration_limit', termination=reason,
                iterations=iteration, runtime=perf_counter()-start, eta=eta,
                internal_epsilon=eps, step_alpha=alpha, growth_threshold=h,
                theoretical_iteration_bound=theoretical_cap,
                minimum_iterations_to_growth_exit=minimum_growth_iterations,
                initial_weight_sum=initial_sum, final_weight_sum=float(x.sum()),
                max_taylor_degree=max_degree, history=history,
                kernel='factorized_taylor_identity_sketch',
                dense_certificate_dimension=d, full_nearly_linear_kernel=False)


def covering_sdp(z, rho, *, eta=1e-4, max_iter=100_000, callback=None):
    """Relative approximation of C_rho using certified PTZ decision calls."""
    z = _validate(z, rho)
    start = perf_counter()
    k, d = z.shape
    c = .9*k
    # Always feasible covering solution M=0, y_i=1/c.
    upper, lower = k/c, float(np.max(1/np.maximum(rho*np.sum(z*z, axis=1), c)))
    matrix, slack = np.zeros((d,d)), np.full(k, 1/c)
    calls = []
    if upper/lower <= 1+eta:
        return dict(matrix=matrix, slack=slack, lower=lower, upper=upper,
                    relative_gap=upper/lower-1, converged=True, iterations=0,
                    calls=[], runtime=perf_counter()-start, termination='analytic_certificate')
    for _ in range(100):
        target = (lower+upper)/2
        result = decision_psdp(z, rho, target, eta=eta, max_iter=max_iter, callback=callback)
        calls.append({key: val for key,val in result.items()
                      if key not in ('matrix','slack','packing')})
        lower = max(lower, result['lower'])
        if result['upper'] < upper:
            upper, matrix, slack = result['upper'], result['matrix'], result['slack']
        if upper/lower <= 1+eta or not result['converged']:
            break
    converged = upper/lower <= 1+eta
    return dict(matrix=matrix, slack=slack, lower=lower, upper=upper,
                relative_gap=max(0., upper/lower-1), converged=converged,
                iterations=sum(c['iterations'] for c in calls), calls=calls,
                runtime=perf_counter()-start,
                termination='relative_gap' if converged else 'iteration_limit')
