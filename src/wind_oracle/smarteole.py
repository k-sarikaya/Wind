from __future__ import annotations
import numpy as np
import pandas as pd

def load_windcube_1min(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "time" not in df.columns:
        raise ValueError("Expected 'time' column in WindCube CSV")
    df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
    return df

def load_power_curve(csv_path: str) -> pd.DataFrame:
    pc = pd.read_csv(csv_path)
    if not {"V", "P"}.issubset(pc.columns):
        raise ValueError("Expected columns V,P in power curve CSV")
    return pc.sort_values("V").reset_index(drop=True)

def windspeed_to_power_mw(ws: np.ndarray, pc: pd.DataFrame) -> np.ndarray:
    v = pc["V"].to_numpy(float)
    p_kw = pc["P"].to_numpy(float)
    p_mw = p_kw / 1000.0
    ws = np.asarray(ws, float)
    ws = np.clip(ws, v.min(), v.max())
    return np.interp(ws, v, p_mw)

def make_demand_proxy(W_mw: np.ndarray, q: float = 0.8) -> np.ndarray:
    return q * np.asarray(W_mw, float)
