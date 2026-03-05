from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from wind_oracle.oracle import OracleParams, run_oracle_series
from wind_oracle.metrics import effective_service_rate, switching_cost_l1
from wind_oracle.smarteole import (
    load_windcube_1min,
    load_power_curve,
    windspeed_to_power_mw,
    make_demand_proxy,
)

LAMBDAS = [0, 0.1, 0.5, 1, 5, 10, 20, 50, 100]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windcube", required=True)
    ap.add_argument("--power-curve", required=True)
    ap.add_argument("--ws-col", default="ws_avg_80")
    ap.add_argument("--demand-q", type=float, default=0.8)
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_windcube_1min(args.windcube)
    pc = load_power_curve(args.power_curve)

    if args.ws_col not in df.columns:
        raise ValueError(f"Missing wind speed column: {args.ws_col}")

    ws = df[args.ws_col].to_numpy(float)
    W = windspeed_to_power_mw(ws, pc)  # MW
    D = make_demand_proxy(W, q=args.demand_q)

    rows = []
    for lam in LAMBDAS:
        params = OracleParams(lam=float(lam))
        k = np.array(
            run_oracle_series(
                W, D, k0=float(D[0]), params=params, k_grid_mw=0.1
            ),
            dtype=float,
        )
        rows.append(
            {
                "lam": lam,
                "eff_service": effective_service_rate(W, D, k),
                "switch_l1": switching_cost_l1(k),
                "W_mean": float(np.mean(W)),
                "D_mean": float(np.mean(D)),
                "k_mean": float(np.mean(k)),
            }
        )

    res = pd.DataFrame(rows).sort_values("lam")
    res.to_csv(outdir / "pareto.csv", index=False)

    plt.figure(figsize=(6, 4))
    plt.plot(res["switch_l1"], res["eff_service"], marker="o")
    for _, r in res.iterrows():
        plt.annotate(str(r["lam"]), (r["switch_l1"], r["eff_service"]), fontsize=8)
    plt.xlabel("Switching cost  Σ|Δk| (MW)")
    plt.ylabel("Effective Service Rate  P(W≥k≥D)")
    plt.title("SMARTEOLE λ-sweep Pareto")
    plt.tight_layout()
    plt.savefig(outdir / "pareto.png", dpi=200)

    print(res.to_string(index=False))
    print(f"Wrote: {outdir/'pareto.csv'} and {outdir/'pareto.png'}")


if __name__ == "__main__":
    main()
