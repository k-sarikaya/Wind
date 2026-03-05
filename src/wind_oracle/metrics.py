from __future__ import annotations
import numpy as np

def effective_service_rate(W: np.ndarray, D: np.ndarray, k: np.ndarray) -> float:
    return float(np.mean((W >= k) & (k >= D)))

def switching_cost_l1(k: np.ndarray) -> float:
    return 0.0 if len(k) <= 1 else float(np.sum(np.abs(np.diff(k))))
