"""Heavy-tailed sample generation and adversarial contamination (global NumPy RNG)."""

import numpy as np
from scipy.special import gammaln
from scipy.stats import multivariate_t


def _sparse_mean(d, s, loc):
    """Vector with the scalar loc on s uniformly chosen coordinates."""
    if not 0 <= s <= d:
        raise ValueError("s must satisfy 0 <= s <= d")
    if np.ndim(loc) != 0 or not np.isfinite(loc) or (s > 0 and loc == 0):
        raise ValueError("loc must be a finite scalar, nonzero when s > 0")
    mu = np.zeros(d)
    mu[np.random.choice(d, size=s, replace=False)] = loc
    return mu


def _shape_factor(scale, d):
    """Cholesky factor of the t shape matrix given as a scalar (times I), diagonal, or matrix."""
    shape = np.asarray(scale, dtype=float)
    if shape.ndim == 0:
        shape = shape * np.eye(d)
    elif shape.shape == (d,):
        shape = np.diag(shape)
    if shape.shape != (d, d) or not np.isfinite(shape).all() or not np.allclose(shape, shape.T):
        raise ValueError("scale must be a finite, symmetric d-by-d shape matrix")
    try:
        return shape, np.linalg.cholesky(shape)
    except np.linalg.LinAlgError as error:
        raise ValueError("scale must be positive definite") from error


def _skew_mean_factor(nu):
    """E[sqrt(nu / chi2_nu)] * sqrt(2/pi), the mean shift per unit skew."""
    if not np.isfinite(nu) or nu <= 1:
        raise ValueError("nu must be finite and greater than 1 for a finite mean")
    return np.sqrt(nu / np.pi) * np.exp(gammaln((nu - 1) / 2) - gammaln(nu / 2))


def t_dist_data_generation(n, loc, nu, d, scale=1):
    """(n, d) multivariate t samples; covariance is nu/(nu-2) times the shape matrix."""
    loc = np.asarray(loc, dtype=float)
    if loc.ndim != 0 and loc.shape != (d,):
        raise ValueError("loc must be a scalar or a length-d vector")
    return multivariate_t.rvs(loc=np.broadcast_to(loc, (d,)), shape=scale, df=nu, size=n).reshape(n, d)


def sparse_t_dist_data_generation(n, loc, nu, d, s, scale=1):
    """Return (data, mu): t samples whose mean mu has loc on s random coordinates."""
    if not nu > 1:
        raise ValueError("nu must be greater than 1 for a finite mean")
    mu = _sparse_mean(d, s, loc)
    return t_dist_data_generation(n, mu, nu, d, scale=scale), mu


def skew_t_dist_data_generation(n, loc, nu, d, scale=1, *, skew):
    """Skew-t samples whose actual mean is loc; skew=0 gives the symmetric t law."""
    mean_factor = _skew_mean_factor(nu)
    loc, skew = np.asarray(loc, dtype=float), np.asarray(skew, dtype=float)
    if (loc.ndim != 0 and loc.shape != (d,)) or not np.isfinite(loc).all():
        raise ValueError("loc must be a finite scalar or length-d vector")
    if skew.shape != (d,) or not np.isfinite(skew).all():
        raise ValueError("skew must be a finite length-d vector")
    _, factor = _shape_factor(scale, d)
    normal = np.random.standard_normal((n, d)) @ factor.T
    half_normal = np.abs(np.random.standard_normal((n, 1)))
    mixing = np.sqrt(nu / np.random.chisquare(nu, size=(n, 1)))
    return loc + mixing * (normal + half_normal * skew) - mean_factor * skew


def sparse_skew_t_dist_data_generation(n, loc, nu, d, s, scale=1, *, skew):
    """Return (data, mu): skew-t samples whose mean mu has loc on s random coordinates."""
    mu = _sparse_mean(d, s, loc)
    return skew_t_dist_data_generation(n, mu, nu, d, scale, skew=skew), mu


def skew_t_covariance(nu, skew, scale=1):
    """Covariance of skew_t_dist_data_generation (nu > 2)."""
    if not np.isfinite(nu) or nu <= 2:
        raise ValueError("nu must be finite and greater than 2 for a finite covariance")
    skew = np.asarray(skew, dtype=float)
    shape, _ = _shape_factor(scale, skew.size)
    variance = nu / (nu - 2)
    return variance * shape + (variance - _skew_mean_factor(nu)**2) * np.outer(skew, skew)


def adversarial_sparse_contamination(data, epsilon, s, strength=30):
    """Copy of data with floor(epsilon * n) rows moved far along one sparse direction.

    Coordinate medians estimate the support. The direction weakens their top-s coordinates and
    reinforces the next s; the lowest projections move to median + strength * MAD (std, then 1,
    if the MAD is zero), keeping orthogonal components. A heuristic, not a worst-case attack.
    """
    if not 0 <= epsilon < 0.5:
        raise ValueError("epsilon must satisfy 0 <= epsilon < 0.5")
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError("strength must be positive and finite")
    contaminated = np.array(data, dtype=float, copy=True)
    n, d = contaminated.shape
    if not 1 <= s <= d:
        raise ValueError("s must satisfy 1 <= s <= d")
    k = int(epsilon * n)
    if k == 0:
        return contaminated

    center = np.median(contaminated, axis=0)
    order = np.argsort(-np.abs(center))
    signs = np.where(center >= 0, 1, -1)
    direction = np.zeros(d)
    direction[order[:s]] = -signs[order[:s]]
    direction[order[s:2 * s]] = signs[order[s:2 * s]]
    direction /= np.linalg.norm(direction)

    projections = (contaminated - center) @ direction
    midpoint = np.median(projections)
    spread = np.median(np.abs(projections - midpoint))
    if spread == 0:
        spread = np.std(projections) or 1.0
    target = midpoint + strength * spread
    indices = np.argpartition(projections, k - 1)[:k]
    contaminated[indices] += (target - projections[indices])[:, None] * direction
    return contaminated
