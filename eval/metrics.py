"""Shared scoring primitives: ECE, calibration bins, Wilson CI, exact McNemar, Holm, phi.

One implementation for every scorer (evaluate.py, eval_analysis.py,
tumor_eval_analysis.py, paired_system_comparison.py, ensemble_posthoc.py, the
experiments/ sweep analysis). Light imports only (numpy + stdlib), so it can be
imported from anywhere without pulling in torch or sklearn.

ECE binning
-----------
Equal-width bins. Bin 0 is the closed interval [0, 1/B]; every other bin is
(lo, hi]. An earlier version used (lo, hi] for *every* bin, so a row with
confidence exactly 0.0 fell in no bin: it still counted in accuracy and mean
confidence but contributed nothing to ECE. That produced impossible published
values (acc .944, mean conf .756, ECE .175 < |.756-.944|). With every row binned,
ECE >= |mean_conf - accuracy| holds by the triangle inequality, and
`check_ece_bound` asserts it.
"""

from __future__ import annotations

import math

import numpy as np

N_BINS = 10


def _bin_masks(confs: np.ndarray, n_bins: int):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if i == 0:
            m = (confs >= lo) & (confs <= hi)
        else:
            m = (confs > lo) & (confs <= hi)
        yield lo, hi, m


def _prep_arrays(confidences, correct) -> tuple[np.ndarray, np.ndarray]:
    c = np.clip(np.asarray(confidences, dtype=float), 0.0, 1.0)
    k = np.asarray(correct, dtype=float)
    if c.shape != k.shape:
        raise ValueError(f"confidences {c.shape} and correct {k.shape} differ in shape")
    if np.isnan(c).any():
        raise ValueError("compute_ece: NaN confidence; substitute a value explicitly first")
    return c, k


def compute_ece(confidences, correct, n_bins: int = N_BINS) -> float:
    """Expected Calibration Error, equal-width bins, first bin [0, 1/B].

    Confidences are clipped to [0, 1]; every row lands in exactly one bin.
    Returns nan for an empty input.
    """
    c, k = _prep_arrays(confidences, correct)
    n = len(c)
    if n == 0:
        return float("nan")
    ece = 0.0
    total = 0
    for _, _, m in _bin_masks(c, n_bins):
        cnt = int(m.sum())
        if cnt:
            total += cnt
            ece += (cnt / n) * abs(float(c[m].mean()) - float(k[m].mean()))
    assert total == n, f"ECE binning lost rows ({total} of {n})"
    return float(ece)


def calibration_bins(confidences, correct, n_bins: int = N_BINS) -> list[dict]:
    """Per-bin rows: bin_lo, bin_hi, n, mean_conf, mean_acc (same binning as compute_ece)."""
    c, k = _prep_arrays(confidences, correct)
    rows = []
    for lo, hi, m in _bin_masks(c, n_bins):
        cnt = int(m.sum())
        rows.append({
            "bin_lo": round(float(lo), 2),
            "bin_hi": round(float(hi), 2),
            "n": cnt,
            "mean_conf": round(float(c[m].mean()), 4) if cnt else float("nan"),
            "mean_acc": round(float(k[m].mean()), 4) if cnt else float("nan"),
        })
    return rows


def check_ece_bound(confidences, correct, ece: float | None = None,
                    n_bins: int = N_BINS, tol: float = 1e-9) -> float:
    """Assert ECE >= |mean_conf - accuracy| (must hold when every row is binned)."""
    c, k = _prep_arrays(confidences, correct)
    if not len(c):
        return float("nan")
    e = compute_ece(c, k, n_bins) if ece is None else ece
    gap = abs(float(c.mean()) - float(k.mean()))
    assert e + tol >= gap, f"ECE {e:.6f} < |mean_conf - acc| = {gap:.6f}"
    return e


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score 95% interval (matches statsmodels proportion_confint(method='wilson'))."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(a_correct, b_correct) -> dict:
    """Exact (binomial) McNemar test on paired 0/1 correctness vectors.

    Same as statsmodels `mcnemar(table, exact=True)`: statistic = min(b, c),
    p = min(1, 2 * BinomCDF(min(b, c); b + c, 0.5)).
    """
    a = np.asarray(a_correct).astype(int)
    b = np.asarray(b_correct).astype(int)
    a_only = int(((a == 1) & (b == 0)).sum())
    b_only = int(((a == 0) & (b == 1)).sum())
    both = int(((a == 1) & (b == 1)).sum())
    neither = int(((a == 0) & (b == 0)).sum())
    n_disc = a_only + b_only
    lo = min(a_only, b_only)
    if n_disc == 0:
        p = 1.0
    else:
        cdf = sum(math.comb(n_disc, i) for i in range(lo + 1)) / (2 ** n_disc)
        p = min(1.0, 2 * cdf)
    return {"n": len(a), "both_correct": both, "a_only_correct": a_only,
            "b_only_correct": b_only, "neither_correct": neither,
            "statistic": float(lo), "p_raw": float(p)}


def holm_bonferroni(pvals) -> list[float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    pvals = list(pvals)
    m = len(pvals)
    order = np.argsort(pvals, kind="stable")
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min((m - rank) * pvals[idx], 1.0))
        adj[idx] = running
    return adj


def phi_coefficient(x, y) -> float:
    """Matthews / phi correlation between two 0/1 vectors (nan if either is constant)."""
    x = np.asarray(x).astype(int)
    y = np.asarray(y).astype(int)
    n11 = int(((x == 1) & (y == 1)).sum())
    n10 = int(((x == 1) & (y == 0)).sum())
    n01 = int(((x == 0) & (y == 1)).sum())
    n00 = int(((x == 0) & (y == 0)).sum())
    den = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return float("nan") if den == 0 else (n11 * n00 - n10 * n01) / den


def parse_bool(value) -> bool | None:
    """Parse a JSON/CSV boolean. "false"/"0"/"no" -> False (unlike bool("false"))."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, float):
        return None if math.isnan(value) else bool(value)
    s = str(value).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    return None


def _self_test() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 300))
        c = rng.choice([0.0, 0.1, 0.3, 0.5, 0.95, 1.0], size=n)
        c = np.where(rng.random(n) < 0.5, c, rng.random(n))
        k = (rng.random(n) < 0.7).astype(float)
        check_ece_bound(c, k)
    # The published-impossible case: 28 rows of confidence 0.0 that are correct.
    c = np.r_[np.zeros(28), np.full(72, 0.9)]
    k = np.ones(100)
    e = check_ece_bound(c, k)
    assert abs(e - (0.28 * 1.0 + 0.72 * 0.1)) < 1e-9, e
    assert parse_bool("false") is False and parse_bool("True") is True
    assert parse_bool(float("nan")) is None


if __name__ == "__main__":
    _self_test()
    print("eval.metrics self-test passed")
