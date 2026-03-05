from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .cost import cost


@dataclass(frozen=True)
class OracleParams:
    alpha: float = 1e6   # hard penalty if k > W
    gamma: float = 50.0  # demand shortfall penalty weight
    beta: float = 1.0    # linear operating cost
    lam: float = 10.0    # switching penalty weight


def _round_to_grid(x: float, grid: float) -> float:
    return float(x) if grid <= 0 else round(x / grid) * grid


def oracle_k(
    W: float,
    D: float,
    k_prev: float,
    params: OracleParams,
    *,
    k_grid_mw: float = 0.1,
    k_min: float = 0.0,
) -> float:
    alpha, gamma, beta, lam = params.alpha, params.gamma, params.beta, params.lam
    candidates: list[float] = []

    # Region A: k >= D  (demand penalty inactive)
    if lam > 0:
        k_A = k_prev - beta / (2.0 * lam)
        if k_A >= D and k_A <= W:
            candidates.append(k_A)

    # Region B: k < D  (quadratic demand penalty active)
    denom = gamma + lam
    if denom > 0:
        k_B = (gamma * D + lam * k_prev) / denom - beta / (2.0 * denom)
        if k_B < D and k_B <= W:
            candidates.append(k_B)

    # boundaries (always feasible)
    candidates.extend([k_min, min(D, W), W])

    best_k = None
    best_c = None
    for k in candidates:
        k_clipped = max(k_min, min(W, k))
        k_rounded = _round_to_grid(k_clipped, k_grid_mw)
        k_rounded = max(k_min, min(W, k_rounded))
        c = cost(
            k_rounded,
            W=W,
            D=D,
            k_prev=k_prev,
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            lam=lam,
        )
        if best_c is None or c < best_c:
            best_c = c
            best_k = k_rounded

    return float(best_k)


def run_oracle_series(
    W_series: Iterable[float],
    D_series: Iterable[float],
    *,
    k0: float,
    params: OracleParams,
    k_grid_mw: float = 0.1,
) -> list[float]:
    ks: list[float] = []
    k_prev = float(k0)
    for W, D in zip(W_series, D_series):
        k = oracle_k(float(W), float(D), k_prev, params, k_grid_mw=k_grid_mw)
        ks.append(k)
        k_prev = k
    return ks
