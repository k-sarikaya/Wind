from __future__ import annotations

def cost(
    k: float,
    W: float,
    D: float,
    k_prev: float,
    *,
    alpha: float,
    gamma: float,
    beta: float,
    lam: float,
) -> float:
    cap_pen = alpha if (W < k) else 0.0
    demand_pen = gamma * max(0.0, D - k) ** 2
    op_cost = beta * k
    stab_pen = lam * (k - k_prev) ** 2
    return cap_pen + demand_pen + op_cost + stab_pen
