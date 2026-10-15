"""Sparse-plane witnesses and a verified geometric median certifier.

The certificate applies to the *represented binary floating point* residual
matrix.  A caller forming residuals by rounded subtraction must account for
that subtraction error separately.  Numerical trigonometry is used only for
finding witnesses.  Pruning uses exact rational arithmetic and rational
outward bounds for square roots, without a trigonometric endpoint calculation.

For u=((1-t*t)/(1+t*t), 2*t/(1+t*t)), a circular cap P @ u >= c
is a quadratic inequality in t.  Its two roots are enclosed by rational
intervals; expanding the feasible intervals and sweeping their exact endpoints
gives a conservative count.  The missing circle point u=(-1, 0) is checked
separately.  These conservative counts can prove absence, never presence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations
from math import isqrt, isfinite, pi
from time import perf_counter
from typing import Iterable

import numpy as np


_ZERO = Fraction(0)
_ONE = Fraction(1)
_SQRT_BITS = 64
_TWO_PI = 2.0 * pi


@dataclass
class GeometricResult:
    status: str
    direction: np.ndarray | None = None
    lower: float = 0.0
    stats: dict = field(default_factory=dict)
    reason: str = ""


def _new_stats() -> dict:
    return {
        "planes": 0,
        "nodes": 0,
        "pruned": 0,
        "supports": 0,
        "certified_supports": 0,
        "unknown_supports": 0,
        "screened_supports": 0,
        "infeasible_nodes": 0,
        "bound_seconds": 0.0,
        "witness_seconds": 0.0,
        "preparation_seconds": 0.0,
        "elapsed": 0.0,
        "support_records": [],
        "certificate_arithmetic": "exact_rational_with_outward_sqrt_bounds",
    }


def _fraction(value: float | Fraction) -> Fraction:
    return value if isinstance(value, Fraction) else Fraction(float(value))


def _sqrt_bounds(value: Fraction, bits: int = _SQRT_BITS) -> tuple[Fraction, Fraction]:
    """Rational outward square-root bounds with scale-adaptive precision."""
    if value < 0:
        raise ValueError("cannot take a square root of a negative rational")
    if value == 0:
        return _ZERO, _ZERO
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    power = bits - exponent // 2
    if power >= 0:
        scale = 1 << power
        root = isqrt((value.numerator << (2 * power)) // value.denominator)
        lower, step = Fraction(root, scale), Fraction(1, scale)
    else:
        scale = 1 << (-power)
        root = isqrt(value.numerator // (value.denominator << (-2 * power)))
        lower, step = Fraction(root * scale), Fraction(scale)
    if lower * lower == value:
        return lower, lower
    return lower, lower + step


def _float_lower(value: Fraction) -> float:
    result = float(value)
    if Fraction(result) > value:
        result = np.nextafter(result, -np.inf)
    return float(result)


def _validate_data(A, q) -> tuple[np.ndarray, int, int]:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2 or not A.shape[0] or not A.shape[1]:
        raise ValueError("A must be a nonempty two-dimensional residual matrix")
    if not np.all(np.isfinite(A)):
        raise ValueError("A must contain only finite values")
    if A.shape[0] % 2 != 1:
        raise ValueError("the median counting oracle requires an odd number of blocks")
    if isinstance(q, (bool, np.bool_)) or int(q) != q or not 1 <= q <= A.shape[1]:
        raise ValueError("q must be an integer between 1 and the number of coordinates")
    return A, int(q), (A.shape[0] + 1) // 2


def _validate_thresholds(T_cut, T_cert=None):
    if not isfinite(float(T_cut)) or not float(T_cut) > 0:
        raise ValueError("T_cut must be finite and positive")
    if T_cert is not None and (
        not isfinite(float(T_cert)) or not float(T_cert) > float(T_cut)
    ):
        raise ValueError("T_cert must be finite and strictly greater than T_cut")


def _validate_budget(time_limit, max_nodes, max_nodes_per_support):
    if not isfinite(float(time_limit)) or float(time_limit) < 0:
        raise ValueError("time_limit must be finite and nonnegative")
    for name, value in (
        ("max_nodes", max_nodes),
        ("max_nodes_per_support", max_nodes_per_support),
    ):
        if isinstance(value, (bool, np.bool_)) or int(value) != value or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")


def _circular_events(arcs: Iterable[tuple[float, float]]):
    full = 0
    events: dict[float, list[int]] = {}

    def interval(left, right):
        events.setdefault(float(left), [0, 0])[0] += 1
        events.setdefault(float(right), [0, 0])[1] += 1

    for center, half_width in arcs:
        center, half_width = float(center), float(half_width)
        if not isfinite(center) or not isfinite(half_width):
            raise ValueError("arc center and half-width must be finite")
        if half_width < 0:
            continue
        if half_width >= pi:
            full += 1
            continue
        center %= _TWO_PI
        left, right = center - half_width, center + half_width
        if left < 0:
            interval(0.0, right)
            interval(left + _TWO_PI, _TWO_PI)
        elif right >= _TWO_PI:
            interval(left, _TWO_PI)
            interval(0.0, right - _TWO_PI)
        else:
            interval(left, right)
    events.setdefault(0.0, [0, 0])
    events.setdefault(_TWO_PI, [0, 0])
    return full, events


def _circle_sweep_details(arcs, h=1):
    """Floating point heuristic sweep, including all relevant open cells."""
    full, events = _circular_events(arcs)
    positions = sorted(events)
    active = full
    best_count, best_theta = full, 0.0
    interiors, endpoints = [], []
    for index, point in enumerate(positions):
        starts, ends = events[point]
        at_point = active + starts
        if point < _TWO_PI:
            if at_point > best_count:
                best_count, best_theta = at_point, point
            if at_point >= h:
                endpoints.append(point)
        active = at_point - ends
        if index + 1 < len(positions) and positions[index + 1] > point:
            theta = point + (positions[index + 1] - point) / 2.0
            if active >= best_count:
                best_count, best_theta = active, theta
            if active >= h:
                interiors.append(theta)
    return best_count, float(best_theta % _TWO_PI), interiors + endpoints


def circular_max_overlap(arcs) -> tuple[int, float]:
    """Maximum overlap of closed arcs supplied as (center, half_width).

    Angles are radians.  A half-width >= pi denotes a full circle; a negative
    half-width denotes an empty arc.  This floating point routine is used for
    witness search only; certificate sweeps use exact rational endpoints.
    """
    count, theta, _ = _circle_sweep_details(arcs)
    return count, theta


def _offset_circle_arcs(P, offset, T):
    radius = np.hypot(P[:, 0], P[:, 1])
    phase = np.arctan2(P[:, 1], P[:, 0])
    arcs = []
    for R, phi, beta in zip(radius, phase, offset):
        delta = float(T) - float(beta)
        if R == 0:
            if delta <= 0:
                arcs.append((0.0, pi))
        elif delta <= -R:
            arcs.append((0.0, pi))
        elif delta <= R:
            arcs.append((float(phi), float(np.arccos(np.clip(delta / R, -1.0, 1.0)))))
    return arcs


def circle_at_x(A_S, x, T) -> tuple[int, float]:
    """Heuristically sweep all angles at a fixed feasible x (including offsets)."""
    A_S = np.asarray(A_S, dtype=float)
    x = np.asarray(x, dtype=float)
    if A_S.ndim != 2 or A_S.shape[1] < 2 or x.shape != (A_S.shape[1] - 2,):
        raise ValueError("x must have length q-2 and A_S must have q >= 2 columns")
    if not np.all(np.isfinite(A_S)) or not np.all(np.isfinite(x)) or not isfinite(float(T)):
        raise ValueError("circle inputs must be finite")
    norm_sq = sum((_fraction(z) ** 2 for z in x), _ZERO)
    if norm_sq > 1:
        raise ValueError("x must lie in the unit ball")
    rho = float(np.sqrt(max(0.0, float(_ONE - norm_sq))))
    return circular_max_overlap(_offset_circle_arcs(A_S[:, :2] * rho, A_S[:, 2:] @ x, T))


def _verified_witness(A, direction, T):
    """Verify sparsity externally; verify ball membership and median exactly here."""
    direction = np.asarray(direction, dtype=float).copy()
    if not np.all(np.isfinite(direction)):
        return None
    norm = float(np.linalg.norm(direction))
    if not norm > 0 or not isfinite(norm):
        return None
    direction /= norm
    for _ in range(8):
        nonzero = np.flatnonzero(direction)
        values = [_fraction(direction[j]) for j in nonzero]
        norm_sq = sum((z * z for z in values), _ZERO)
        if norm_sq <= 1:
            break
        direction = np.nextafter(direction, np.zeros_like(direction))
    else:
        return None
    approximate = float(np.median(A @ direction))
    if approximate < 0:
        direction = -direction
        values = [-z for z in values]
        approximate = -approximate
    # The exact check is inexpensive only for promising directions.
    if approximate <= float(T) - 1e-12 * max(1.0, abs(float(T))):
        return None
    exact_scores = [
        sum((_fraction(row[j]) * value for j, value in zip(nonzero, values)), _ZERO)
        for row in A
    ]
    median = sorted(exact_scores)[len(exact_scores) // 2]
    if median > _fraction(T):
        return direction, _float_lower(median)
    return None


def _circle_witness(A_S, x, T, deadline):
    norm_sq = sum((_fraction(z) ** 2 for z in x), _ZERO)
    if norm_sq > 1:
        return None
    rho = float(np.sqrt(max(0.0, float(_ONE - norm_sq))))
    arcs = _offset_circle_arcs(A_S[:, :2] * rho, A_S[:, 2:] @ x, T)
    _, _, candidates = _circle_sweep_details(arcs, (len(A_S) + 1) // 2)
    for theta in candidates:
        if perf_counter() >= deadline:
            return None
        v = np.concatenate(([rho * np.cos(theta), rho * np.sin(theta)], x))
        witness = _verified_witness(A_S, v, T)
        if witness is not None:
            return witness
    return None


def _quadratic_outer_intervals(a, b, c):
    """Closed intervals conservatively containing {t : a*t**2+b*t+c >= 0}.

    None represents an infinite endpoint.  Intervals belonging to one block
    never overlap, so a sweep cannot count the same block twice.
    """
    if a == 0:
        if b == 0:
            return [(None, None)] if c >= 0 else []
        root = -c / b
        return [(root, None)] if b > 0 else [(None, root)]
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return [(None, None)] if a > 0 else []
    if discriminant == 0:
        if a > 0:
            return [(None, None)]
        root = -b / (2 * a)
        return [(root, root)]
    sqrt_lo, sqrt_hi = _sqrt_bounds(discriminant)
    center = -b / (2 * a)
    half_lo, half_hi = sqrt_lo / (2 * abs(a)), sqrt_hi / (2 * abs(a))
    if a < 0:
        return [(center - half_hi, center + half_hi)]
    left_end, right_start = center - half_lo, center + half_lo
    if left_end >= right_start:
        return [(None, None)]
    return [(None, left_end), (right_start, None)]


def _rational_interval_max_overlap(intervals):
    active = 0
    events: dict[Fraction, list[int]] = {}
    for left, right in intervals:
        if left is None:
            active += 1
        else:
            events.setdefault(left, [0, 0])[0] += 1
        if right is not None:
            events.setdefault(right, [0, 0])[1] += 1
    maximum = active
    for point in sorted(events):
        starts, ends = events[point]
        active += starts  # Closed endpoints: starts precede ends.
        maximum = max(maximum, active)
        active -= ends
    return maximum


@dataclass
class _SupportData:
    array: np.ndarray
    rows: list
    g_norm_upper: list
    row_norm_sq: list

    @classmethod
    def make(cls, A_S):
        A_S = np.asarray(A_S, dtype=float)
        rows = [[_fraction(z) for z in row] for row in A_S]
        row_norm_sq = [sum((z * z for z in row), _ZERO) for row in rows]
        g_norm_upper = [
            _sqrt_bounds(sum((z * z for z in row[2:]), _ZERO))[1] for row in rows
        ]
        return cls(A_S, rows, g_norm_upper, row_norm_sq)


def _node_upper_count(data, lo, hi, T):
    lo, hi = [_fraction(z) for z in lo], [_fraction(z) for z in hi]
    min_sq = sum((0 if l <= 0 <= u else min(l * l, u * u) for l, u in zip(lo, hi)), _ZERO)
    if min_sq > 1:
        return -1
    max_sq = sum((max(l * l, u * u) for l, u in zip(lo, hi)), _ZERO)
    rho_minus = _sqrt_bounds(max(_ZERO, _ONE - max_sq))[0]
    rho_plus = _sqrt_bounds(_ONE - min_sq)[1]
    threshold = _fraction(T)
    intervals = []
    at_infinity = 0
    for row, norm_upper in zip(data.rows, data.g_norm_upper):
        beta_box = sum((max(g * l, g * u) for g, l, u in zip(row[2:], lo, hi)), _ZERO)
        beta = min(beta_box, norm_upper)
        delta = threshold - beta
        if delta > 0:
            if rho_plus == 0:
                continue
            cap_threshold = delta / rho_plus
        else:
            if rho_minus == 0:
                intervals.append((None, None))
                at_infinity += 1
                continue
            cap_threshold = delta / rho_minus
        px, py = row[:2]
        # P dot u >= cap_threshold under the rational half-angle map.
        a, b, c = -px - cap_threshold, 2 * py, px - cap_threshold
        intervals.extend(_quadratic_outer_intervals(a, b, c))
        at_infinity += int(-px >= cap_threshold)
    return max(at_infinity, _rational_interval_max_overlap(intervals))


def node_upper_count(A_S, lo, hi, T, *, verified=True) -> int:
    """Verified upper count for every x in the box intersected with the ball.

    Returns -1 for an empty box/ball intersection.  All arithmetic affecting
    the certificate is rational; ``verified=False`` is rejected deliberately.
    """
    if not verified:
        raise ValueError("this implementation provides only verified node bounds")
    A_S = np.asarray(A_S, dtype=float)
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    if A_S.ndim != 2 or A_S.shape[1] < 2:
        raise ValueError("node bounds require a two-dimensional A_S with q >= 2")
    if lo.shape != (A_S.shape[1] - 2,) or hi.shape != lo.shape or np.any(lo > hi):
        raise ValueError("lo and hi must define a box of dimension q-2")
    if not all(np.all(np.isfinite(z)) for z in (A_S, lo, hi)) or not isfinite(float(T)):
        raise ValueError("node inputs must be finite")
    return _node_upper_count(_SupportData.make(A_S), lo, hi, T)


def _feasible_center(lo, hi):
    center = lo + (hi - lo) / 2.0
    if sum((_fraction(z) ** 2 for z in center), _ZERO) <= 1:
        return center
    center = np.minimum(np.maximum(np.zeros_like(lo), lo), hi)
    if sum((_fraction(z) ** 2 for z in center), _ZERO) <= 1:
        return center
    return None


def _support_search(data, T_cut, T_cert, h, context, search_cuts):
    """A support search sharing the global deadline and node counter."""
    stats, deadline, global_limit, support_limit = context
    start_nodes = stats["nodes"]
    if perf_counter() >= deadline:
        return "UNKNOWN", None, 0.0, "wall_time_limit"
    T_sq = _fraction(T_cert) ** 2
    if sum(value >= T_sq for value in data.row_norm_sq) < h:
        stats["screened_supports"] += 1
        return "CERTIFIED", None, 0.0, "support_row_norm_bound"
    q = data.array.shape[1]
    if q == 1:
        if stats["nodes"] >= global_limit or support_limit == 0:
            return "UNKNOWN", None, 0.0, "node_limit"
        stats["nodes"] += 1
        if search_cuts:
            for sign in (1.0, -1.0):
                witness = _verified_witness(data.array, np.array([sign]), T_cut)
                if witness is not None:
                    return "CUT", witness[0], witness[1], "signed_coordinate_witness"
        upper_count = max(
            sum(sign * row[0] >= _fraction(T_cert) for row in data.rows)
            for sign in (_ONE, -_ONE)
        )
        if upper_count < h:
            stats["pruned"] += 1
            return "CERTIFIED", None, 0.0, "signed_coordinate_certificate"
        return "UNKNOWN", None, 0.0, "unresolved_coordinate_threshold"
    queue = [(np.full(q - 2, -1.0), np.full(q - 2, 1.0))]
    while queue:
        if perf_counter() >= deadline:
            return "UNKNOWN", None, 0.0, "wall_time_limit"
        if stats["nodes"] >= global_limit:
            return "UNKNOWN", None, 0.0, "global_node_limit"
        if stats["nodes"] - start_nodes >= support_limit:
            return "UNKNOWN", None, 0.0, "support_node_limit"
        lo, hi = queue.pop()
        stats["nodes"] += 1
        x0 = _feasible_center(lo, hi)
        if x0 is None:
            stats["pruned"] += 1
            stats["infeasible_nodes"] += 1
            continue
        if search_cuts:
            witness_started = perf_counter()
            witness = _circle_witness(data.array, x0, T_cut, deadline)
            stats["witness_seconds"] += perf_counter() - witness_started
            if witness is not None:
                return "CUT", witness[0], witness[1], "fixed_x_circle_witness"
        if perf_counter() >= deadline:
            return "UNKNOWN", None, 0.0, "wall_time_limit"
        bound_started = perf_counter()
        upper = _node_upper_count(data, lo, hi, T_cert)
        stats["bound_seconds"] += perf_counter() - bound_started
        if upper < h:
            stats["pruned"] += 1
            continue
        if q == 2:
            return "UNKNOWN", None, 0.0, "circle_threshold_not_certified"
        axis = int(np.argmax(hi - lo))
        middle = lo[axis] + (hi[axis] - lo[axis]) / 2.0
        if middle <= lo[axis] or middle >= hi[axis]:
            return "UNKNOWN", None, 0.0, "floating_box_resolution_limit"
        left_hi, right_lo = hi.copy(), lo.copy()
        left_hi[axis] = middle
        right_lo[axis] = middle
        queue.append((right_lo, hi.copy()))
        queue.append((lo.copy(), left_hi))
    return "CERTIFIED", None, 0.0, "all_boxes_pruned"


def certify_support(A_S, T_cut, T_cert, h=None, max_nodes=5000, *, time_limit=0.25, search_cuts=True):
    """Search/certify one fixed support; direction is in support coordinates."""
    A_S = np.asarray(A_S, dtype=float)
    A_S, q, expected_h = _validate_data(A_S, A_S.shape[1] if A_S.ndim == 2 else 0)
    if h is not None and h != expected_h:
        raise ValueError("h must equal (K+1)//2")
    _validate_thresholds(T_cut, T_cert)
    _validate_budget(time_limit, max_nodes, max_nodes)
    started = perf_counter()
    stats = _new_stats()
    stats["supports"] = 1
    preparation_started = perf_counter()
    data = _SupportData.make(A_S)
    stats["preparation_seconds"] = perf_counter() - preparation_started
    outcome = _support_search(data, T_cut, T_cert, expected_h, (stats, started + time_limit, max_nodes, max_nodes), search_cuts)
    status, direction, lower, reason = outcome
    stats["certified_supports"] = int(status == "CERTIFIED")
    stats["unknown_supports"] = int(status == "UNKNOWN")
    stats["elapsed"] = perf_counter() - started
    return GeometricResult(status, direction, lower, stats, reason)


def _support_candidates(A, q, warm_start):
    d = A.shape[1]
    energy_order = np.argsort(-np.sum(A * A, axis=0), kind="stable")
    candidates = []
    seen = set()

    def add(vector):
        vector = np.asarray(vector, dtype=float)
        support = tuple(sorted(np.argsort(-np.abs(vector), kind="stable")[:q].tolist()))
        if support not in seen:
            candidates.append(support)
            seen.add(support)

    if warm_start is not None:
        warm = np.asarray(warm_start, dtype=float)
        if warm.shape == (d,) and np.all(np.isfinite(warm)):
            add(warm)
    add(np.mean(A, axis=0))
    add(np.median(A, axis=0))
    energy = np.zeros(d)
    energy[energy_order[:q]] = np.arange(q, 0, -1)
    add(energy)
    row_order = np.argsort(-np.linalg.norm(A, axis=1), kind="stable")
    for row in row_order[:8]:
        add(A[row])
    return candidates


def _orthonormal_plane(first, second):
    first, second = np.array(first, dtype=float), np.array(second, dtype=float)
    norm = np.linalg.norm(first)
    if not norm > 1e-14:
        return None
    first /= norm
    second -= float(first @ second) * first
    norm = np.linalg.norm(second)
    if not norm > 1e-12:
        return None
    second /= norm
    return first, second


def fast_plane_violation_search(A, q, T_cut, *, warm_start=None, time_limit=0.25, max_planes=128, planes_per_support=8):
    """Search a deterministic collection of sparse planes for verified cuts."""
    A, q, h = _validate_data(A, q)
    _validate_thresholds(T_cut)
    _validate_budget(time_limit, max_planes, planes_per_support)
    started = perf_counter()
    deadline = started + float(time_limit)
    stats = _new_stats()

    def finish(status, witness=None, reason="no_plane_witness"):
        stats["elapsed"] = perf_counter() - started
        if witness is None:
            return GeometricResult(status, stats=stats, reason=reason)
        return GeometricResult(status, witness[0], witness[1], stats, reason)

    if q == 1:
        for j in range(A.shape[1]):
            if perf_counter() >= deadline:
                return finish("UNKNOWN", reason="wall_time_limit")
            direction = np.zeros(A.shape[1])
            direction[j] = 1.0
            witness = _verified_witness(A, direction, T_cut)
            if witness is not None:
                return finish("CUT", witness, "signed_coordinate_witness")
        return finish("UNKNOWN")
    for support in _support_candidates(A, q, warm_start):
        if perf_counter() >= deadline:
            return finish("UNKNOWN", reason="wall_time_limit")
        stats["supports"] += 1
        indices = list(support)
        A_S = A[:, indices]
        seeds = [np.mean(A_S, axis=0), np.median(A_S, axis=0)]
        if warm_start is not None and np.asarray(warm_start).shape == (A.shape[1],):
            seeds.insert(0, np.asarray(warm_start)[indices])
        row_order = np.argsort(-np.linalg.norm(A_S, axis=1), kind="stable")
        seeds.extend(A_S[j].copy() for j in row_order[:6])
        seeds.extend(np.eye(q))
        pairs = list(combinations(range(min(len(seeds), 10)), 2))
        # A deterministic coordinate plane first, then data-informed planes.
        planes = [(np.eye(q)[0], np.eye(q)[1])]
        planes.extend((seeds[i], seeds[j]) for i, j in pairs)
        support_planes = 0
        for first, second in planes:
            if perf_counter() >= deadline:
                return finish("UNKNOWN", reason="wall_time_limit")
            if stats["planes"] >= max_planes:
                return finish("UNKNOWN", reason="plane_limit")
            if support_planes >= planes_per_support:
                break
            plane = _orthonormal_plane(first, second)
            if plane is None:
                continue
            u1, u2 = plane
            support_planes += 1
            stats["planes"] += 1
            P = np.column_stack((A_S @ u1, A_S @ u2))
            arcs = _offset_circle_arcs(P, np.zeros(len(A)), T_cut)
            _, _, angles = _circle_sweep_details(arcs, h)
            for theta in angles:
                if perf_counter() >= deadline:
                    return finish("UNKNOWN", reason="wall_time_limit")
                direction = np.zeros(A.shape[1])
                direction[indices] = u1 * np.cos(theta) + u2 * np.sin(theta)
                witness = _verified_witness(A, direction, T_cut)
                if witness is not None:
                    return finish("CUT", witness, "sparse_plane_witness")
    return finish("UNKNOWN")


def global_geometric_oracle(A, q, T_cut, T_cert, *, time_limit=0.25, max_nodes=5000, max_nodes_per_support=5000, search_cuts=True, warm_start=None):
    """Cover all q-supports with a shared wall-clock and node budget.

    Smaller supports are included because zeros are allowed inside a q-support.
    UNKNOWN is returned whenever even one required support remains unresolved.
    ``warm_start`` is accepted for API consistency; support order is lexicographic.
    """
    A, q, h = _validate_data(A, q)
    _validate_thresholds(T_cut, T_cert)
    _validate_budget(time_limit, max_nodes, max_nodes_per_support)
    started = perf_counter()
    deadline = started + float(time_limit)
    stats = _new_stats()
    context = stats, deadline, int(max_nodes), int(max_nodes_per_support)
    any_unknown = False
    reason = "all_supports_certified"
    for support in combinations(range(A.shape[1]), q):
        if perf_counter() >= deadline:
            any_unknown, reason = True, "wall_time_limit"
            break
        if stats["nodes"] >= max_nodes:
            any_unknown, reason = True, "global_node_limit"
            break
        support_started = perf_counter()
        before_nodes, before_pruned = stats["nodes"], stats["pruned"]
        before_bound, before_witness = stats["bound_seconds"], stats["witness_seconds"]
        stats["supports"] += 1
        preparation_started = perf_counter()
        data = _SupportData.make(A[:, list(support)])
        stats["preparation_seconds"] += perf_counter() - preparation_started
        status, direction, lower, support_reason = _support_search(data, T_cut, T_cert, h, context, search_cuts)
        stats["support_records"].append({
            "support": list(support), "status": status,
            "nodes": stats["nodes"] - before_nodes,
            "pruned": stats["pruned"] - before_pruned,
            "bound_seconds": stats["bound_seconds"] - before_bound,
            "witness_seconds": stats["witness_seconds"] - before_witness,
            "elapsed": perf_counter() - support_started,
            "reason": support_reason,
        })
        if status == "CUT":
            full_direction = np.zeros(A.shape[1])
            full_direction[list(support)] = direction
            stats["elapsed"] = perf_counter() - started
            return GeometricResult("CUT", full_direction, lower, stats, support_reason)
        if status == "CERTIFIED":
            stats["certified_supports"] += 1
        else:
            any_unknown = True
            stats["unknown_supports"] += 1
            reason = support_reason
    stats["elapsed"] = perf_counter() - started
    return GeometricResult("UNKNOWN" if any_unknown else "CERTIFIED", stats=stats, reason=reason)


def geometric_separation(A, q, T_cut, T_cert, *, time_limit=0.25, max_nodes=5000, max_nodes_per_support=5000, warm_start=None, search_cuts=True):
    """Plane search followed by global BnB, under one overall time budget.

    Plane search is capped at a quarter of the budget so it cannot starve the
    global search.  Disabling cut search bypasses planes and all witness work.
    """
    A, q, _ = _validate_data(A, q)
    _validate_thresholds(T_cut, T_cert)
    _validate_budget(time_limit, max_nodes, max_nodes_per_support)
    started = perf_counter()
    plane_result = None
    if search_cuts and time_limit > 0:
        plane_result = fast_plane_violation_search(
            A, q, T_cut, warm_start=warm_start, time_limit=min(0.1, time_limit / 4.0)
        )
        if plane_result.status == "CUT":
            plane_result.stats["phase"] = "planes"
            plane_result.stats["elapsed"] = perf_counter() - started
            return plane_result
    remaining = max(0.0, time_limit - (perf_counter() - started))
    result = global_geometric_oracle(
        A, q, T_cut, T_cert, time_limit=remaining, max_nodes=max_nodes,
        max_nodes_per_support=max_nodes_per_support, search_cuts=search_cuts,
        warm_start=warm_start,
    )
    result.stats["phase"] = "global"
    if plane_result is not None:
        result.stats["planes"] = plane_result.stats["planes"]
        result.stats["plane_supports"] = plane_result.stats["supports"]
        result.stats["plane_elapsed"] = plane_result.stats["elapsed"]
    result.stats["elapsed"] = perf_counter() - started
    return result
