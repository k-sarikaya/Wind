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

DEFAULT_LAMBDAS = [0, 0.1, 0.5, 1, 5, 10, 20, 50, 100]


def _parse_csv_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def run_lambda_sweep(W: np.ndarray, D: np.ndarray, lambdas: list[float]) -> pd.DataFrame:
    rows = []
    for lam in lambdas:
        params = OracleParams(lam=float(lam))
        k = np.array(
            run_oracle_series(W, D, k0=float(D[0]), params=params, k_grid_mw=0.1),
            dtype=float,
        )
        rows.append(
            {
                "lam": float(lam),
                "eff_service": effective_service_rate(W, D, k),
                "switch_l1": switching_cost_l1(k),
                "W_mean": float(np.mean(W)),
                "D_mean": float(np.mean(D)),
                "k_mean": float(np.mean(k)),
            }
        )
    return pd.DataFrame(rows).sort_values("lam")


def plot_pareto(res: pd.DataFrame, outpath: Path, title: str) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(res["switch_l1"], res["eff_service"], marker="o")
    for _, r in res.iterrows():
        plt.annotate(str(r["lam"]), (r["switch_l1"], r["eff_service"]), fontsize=8)
    plt.xlabel("Switching cost  Σ|Δk| (MW)")
    plt.ylabel("Effective Service Rate  P(W≥k≥D)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windcube", required=True)
    ap.add_argument("--power-curve", required=True)
    ap.add_argument("--ws-col", default="ws_avg_80")

    ap.add_argument("--demand-q", type=float, default=0.8, help="Single demand multiplier q (D=qW)")
    ap.add_argument(
        "--demand-q-sweep",
        type=str,
        default="",
        help="Comma-separated q values, e.g. 0.6,0.7,0.8,0.9 (overrides --demand-q)",
    )

    ap.add_argument("--outdir", default="outputs")
    ap.add_argument(
        "--lambdas",
        type=str,
        default="",
        help="Comma-separated lambdas. If empty uses default list.",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    lambdas = DEFAULT_LAMBDAS if not args.lambdas.strip() else _parse_csv_floats(args.lambdas)

    # Load data
    df = load_windcube_1min(args.windcube)
    pc = load_power_curve(args.power_curve)
    if args.ws_col not in df.columns:
        raise ValueError(f"Missing wind speed column: {args.ws_col}")

    ws = df[args.ws_col].to_numpy(float)
    W = windspeed_to_power_mw(ws, pc)  # MW

    q_list = [args.demand_q] if not args.demand_q_sweep.strip() else _parse_csv_floats(args.demand_q_sweep)

    all_rows = []
    for q in q_list:
        D = make_demand_proxy(W, q=float(q))
        res = run_lambda_sweep(W, D, lambdas)

        qdir = outdir / f"q_{q:g}"
        qdir.mkdir(parents=True, exist_ok=True)

        res.to_csv(qdir / "pareto.csv", index=False)
        plot_pareto(res, qdir / "pareto.png", title=f"SMARTEOLE λ-sweep Pareto (q={q:g})")

        res2 = res.copy()
        res2.insert(0, "q", float(q))
        all_rows.append(res2)

        print(f"\n=== q={q:g} ===")
        print(res.to_string(index=False))

    all_df = pd.concat(all_rows, ignore_index=True)
    all_df.to_csv(outdir / "pareto_all.csv", index=False)
    print(f"\nWrote: {outdir/'pareto_all.csv'} plus per-q folders under {outdir}")


if __name__ == "__main__":
    main()
