#!/usr/bin/env python3
"""
Per-scenario analysis of rubric scores across decoy counts.

Analysis 1: Linear regression of score on decoy count (slope, R-squared)
Analysis 2: Cohen's d for each decoy level vs decoy=0
Analysis 3: Cross-scenario beta variability

Both overall (all scenarios pooled) and per-scenario breakdowns are produced
for Manual and Auto object detection methods.
"""

import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

_OUTPUT_ROOT = Path(__file__).resolve().parent.parent / "output"
_DEFAULT_MODEL = "gpt"


def _find_latest_batch(model: str = _DEFAULT_MODEL) -> str:
    """Return the ``test/`` path of the most recent batch for *model*.

    Batches are named ``batch_YYYYMMDD_HHMMSS`` so lexicographic sort
    gives chronological order.
    """
    model_dir = _OUTPUT_ROOT / model
    if not model_dir.is_dir():
        return str(model_dir / "<no_batch_found>" / "test")
    batches = sorted(
        [d for d in model_dir.iterdir() if d.is_dir() and d.name.startswith("batch_")],
        key=lambda d: d.name,
    )
    if not batches:
        return str(model_dir / "<no_batch_found>" / "test")
    return str(batches[-1] / "test")


METHOD_MAP = {
    "manual_object_detection": "Manual",
    "auto_object_detection": "Auto",
}

SCENARIO_RENAME = {
    f"scenario{i:02d}": f"T{i}" for i in range(1, 11)
}


def extract_decoy_number(dir_name):
    for part in dir_name.split('_'):
        if part.startswith('decoy'):
            return int(part.replace('decoy', ''))
    return None


def collect_data(base_path):
    """Collect rubric scores for all scenarios, methods, and iterations."""
    records = []
    for scenario_dir in sorted(base_path.iterdir(), key=lambda x: x.name):
        if not scenario_dir.is_dir() or "decoy" not in scenario_dir.name:
            continue
        decoy_count = extract_decoy_number(scenario_dir.name)
        if decoy_count is None:
            continue
        scenario_id = scenario_dir.name.split('_')[0]

        iter_dirs = sorted(
            d for d in scenario_dir.iterdir()
            if d.is_dir() and d.name.startswith("iter_")
        )
        for iter_dir in iter_dirs:
            eval_file = iter_dir / "evaluation" / "rubric_evaluation_results.json"
            if not eval_file.exists():
                continue
            try:
                with open(eval_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                continue

            evaluations = data.get("evaluations", {})
            for method_key, method_short in METHOD_MAP.items():
                if method_key not in evaluations:
                    continue
                score = evaluations[method_key].get("final_score")
                if score is None:
                    continue
                iter_num = int(iter_dir.name.replace("iter_", ""))
                records.append({
                    "case": scenario_id,
                    "decoy": decoy_count,
                    "method": method_short,
                    "iteration": iter_num,
                    "score": float(score),
                })

    result = pd.DataFrame(records)
    if result.empty:
        return result
    result["case"] = result["case"].map(SCENARIO_RENAME).fillna(result["case"])
    return result


# =========================================================================
# Linear regression (slope, R-squared)
# =========================================================================

def perform_regression(X, y):
    """Run linear regression via scipy and return a dict of statistics."""
    n = len(X)
    if n < 3:
        return None
    res = stats.linregress(X, y)
    slope, intercept, r_value = res.slope, res.intercept, res.rvalue
    r_squared = r_value ** 2

    y_pred = slope * X + intercept
    residuals = y - y_pred
    mse = np.sum(residuals ** 2) / (n - 2)
    residual_se = np.sqrt(mse)

    ss_x = np.sum((X - np.mean(X)) ** 2)
    se_slope = residual_se / np.sqrt(ss_x)
    t_slope = slope / se_slope
    p_slope = stats.t.cdf(t_slope, n - 2)  # one-sided: H0: beta>=0, H1: beta<0

    ss_reg = np.sum((y_pred - np.mean(y)) ** 2)
    ss_res = np.sum(residuals ** 2)
    f_stat = (ss_reg / 1) / (ss_res / (n - 2)) if ss_res > 0 else np.nan
    f_p = 1 - stats.f.cdf(f_stat, 1, n - 2) if not np.isnan(f_stat) else np.nan

    return {
        "n": n,
        "beta": slope,
        "SE": se_slope,
        "t": t_slope,
        "p": p_slope,
        "R2": r_squared,
        "F": f_stat,
        "F_p": f_p,
    }


def build_regression_table(df):
    """Build regression table for overall and per-scenario breakdowns."""
    rows = []
    methods = ["Manual", "Auto"]
    scopes = ["Overall"] + sorted(df["case"].unique())

    for scope in scopes:
        for method in methods:
            subset = (df[df["method"] == method] if scope == "Overall"
                      else df[(df["case"] == scope) & (df["method"] == method)])
            if len(subset) < 3:
                continue
            res = perform_regression(subset["decoy"].values.astype(float),
                                     subset["score"].values.astype(float))
            if res:
                rows.append({"scope": scope, "method": method, **res})

    return pd.DataFrame(rows)


# =========================================================================
# Cohen's d (each decoy level vs decoy=0)
# =========================================================================

def cohens_d(group0, groupK):
    """Cohen's d = (groupK - group0) / pooled_std.  Negative means score drops."""
    n0, nK = len(group0), len(groupK)
    if n0 < 2 or nK < 2:
        return np.nan
    mean0, meanK = np.mean(group0), np.mean(groupK)
    std0, stdK = np.std(group0, ddof=1), np.std(groupK, ddof=1)
    pooled = np.sqrt(((n0 - 1) * std0**2 + (nK - 1) * stdK**2) / (n0 + nK - 2))
    if pooled == 0:
        return 0.0
    return (meanK - mean0) / pooled


def build_cohens_d_table(df):
    """Build Cohen's d table for overall and per-scenario breakdowns."""
    rows = []
    methods = ["Manual", "Auto"]
    decoy_levels = [1, 2, 3, 4]
    scopes = ["Overall"] + sorted(df["case"].unique())

    for scope in scopes:
        for method in methods:
            subset = (df[df["method"] == method] if scope == "Overall"
                      else df[(df["case"] == scope) & (df["method"] == method)])
            baseline = subset[subset["decoy"] == 0]["score"].values
            row = {"scope": scope, "method": method,
                   "n_decoy0": len(baseline),
                   "mean_decoy0": np.mean(baseline) if len(baseline) > 0 else np.nan}
            for k in decoy_levels:
                gk = subset[subset["decoy"] == k]["score"].values
                d = cohens_d(baseline, gk)
                row[f"d_decoy{k}_vs_0"] = d
                row[f"n_decoy{k}"] = len(gk)
                row[f"mean_decoy{k}"] = np.mean(gk) if len(gk) > 0 else np.nan
            rows.append(row)

    return pd.DataFrame(rows)


# =========================================================================
# Cross-scenario beta variability
# =========================================================================

def compute_beta_variability(reg_df):
    """Compute SD and summary statistics of per-scenario betas."""
    methods = ["Manual", "Auto"]
    all_scenarios = sorted(
        reg_df[reg_df["scope"] != "Overall"]["scope"].unique()
    )
    results = []
    for method in methods:
        case_rows = reg_df[
            (reg_df["scope"] != "Overall") & (reg_df["method"] == method)
        ]
        case_betas = case_rows.set_index("scope")["beta"]
        if len(case_betas) > 1:
            row = {
                "method": method,
                "n_scenarios": len(case_betas),
                "mean_beta": case_betas.mean(),
                "sd_beta": case_betas.std(ddof=1),
                "min_beta": case_betas.min(),
                "max_beta": case_betas.max(),
                "range_beta": case_betas.max() - case_betas.min(),
                "cv_beta": case_betas.std(ddof=1) / abs(case_betas.mean())
                           if case_betas.mean() != 0 else np.nan,
            }
            for s in all_scenarios:
                row[f"beta_{s}"] = case_betas.get(s, np.nan)
            results.append(row)
    return pd.DataFrame(results)


# =========================================================================
# Merged summary: regression + Cohen's d + per-decoy means, one CSV per method
# =========================================================================

def build_method_summary(reg_df, d_df, method):
    """Merge regression and Cohen's d tables into one DataFrame for *method*.

    Each row = one scope (Overall or a scenario).
    Columns: per-decoy mean/n, Cohen's d, and regression statistics.
    """
    d_method = d_df[d_df["method"] == method].drop(columns=["method"]).set_index("scope")
    r_method = reg_df[reg_df["method"] == method].drop(columns=["method"]).set_index("scope")

    merged = d_method.join(r_method, how="outer", lsuffix="_d")
    if "n_d" in merged.columns:
        merged = merged.drop(columns=["n_d"])

    scopes = ["Overall"] + [s for s in merged.index if s != "Overall"]
    merged = merged.reindex(scopes).reset_index().rename(columns={"index": "scope"})

    return merged


# =========================================================================
# Main
# =========================================================================

def main():
    default_path = _find_latest_batch()
    parser = argparse.ArgumentParser(
        description="Per-scenario analysis: R-squared and effect sizes")
    parser.add_argument(
        "base_path", nargs="?", default=default_path,
        help=(
            "Path to test/ directory containing scenario directories. "
            f"Default: latest batch under output/{_DEFAULT_MODEL}/ "
            f"(currently {default_path})"
        ),
    )
    args = parser.parse_args()

    base_path = Path(args.base_path)
    output_dir = base_path / "cross_analysis" / "per_scenario_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {base_path.resolve()}")
    print(f"Output: {output_dir.resolve()}")
    print("Collecting data ...")
    df = collect_data(base_path)
    if df.empty:
        print("  [ERROR] No data found. Please check the path.")
        return
    print(f"  Total records: {len(df)}, scenarios: {df['case'].nunique()}, "
          f"decoy range: {df['decoy'].min()}-{df['decoy'].max()}")

    # --- Mean scores per decoy level ---
    print("\nMean scores per decoy level:")
    decoy_levels = sorted(df["decoy"].unique())
    for method in ["Manual", "Auto"]:
        mdf = df[df["method"] == method]
        print(f"\n  {method} (Overall):")
        parts = [f"d={k}: {mdf[mdf['decoy']==k]['score'].mean():.1f}" for k in decoy_levels]
        print(f"    {', '.join(parts)}")
        for case in sorted(mdf["case"].unique()):
            cdf = mdf[mdf["case"] == case]
            parts = [f"d={k}: {cdf[cdf['decoy']==k]['score'].mean():.1f}"
                     for k in decoy_levels if len(cdf[cdf["decoy"] == k]) > 0]
            print(f"    {case:6s}  {', '.join(parts)}")

    # --- Linear regression ---
    print("\nLinear regression (slope, R-squared) ...")
    reg_df = build_regression_table(df)

    overall_reg = reg_df[reg_df["scope"] == "Overall"]
    print("\n  Overall (all scenarios pooled):")
    for _, row in overall_reg.iterrows():
        p_str = ("p < .001" if row["p"] < 0.001
                 else ("p < .01" if row["p"] < 0.01
                       else ("p < .05" if row["p"] < 0.05
                             else f"p = {row['p']:.3f}")))
        print(f"    {row['method']:10s}  beta={row['beta']:+.3f}  SE={row['SE']:.3f}  "
              f"t={row['t']:.3f}  {p_str}  R2={row['R2']:.4f}")

    print("\n  Per scenario:")
    case_reg = reg_df[reg_df["scope"] != "Overall"]
    for case in sorted(case_reg["scope"].unique()):
        for _, row in case_reg[case_reg["scope"] == case].iterrows():
            print(f"    {case:6s} {row['method']:10s}  beta={row['beta']:+.3f}  "
                  f"R2={row['R2']:.4f}  p={row['p']:.4f}")

    # --- Cohen's d ---
    print("\nCohen's d (each decoy level vs decoy=0) ...")
    d_df = build_cohens_d_table(df)

    overall_d = d_df[d_df["scope"] == "Overall"]
    print("\n  Overall:")
    for _, row in overall_d.iterrows():
        print(f"    {row['method']:10s}  "
              f"d1={row['d_decoy1_vs_0']:+.3f}  "
              f"d2={row['d_decoy2_vs_0']:+.3f}  "
              f"d3={row['d_decoy3_vs_0']:+.3f}  "
              f"d4={row['d_decoy4_vs_0']:+.3f}")

    # --- Beta variability ---
    print("\nCross-scenario beta variability ...")
    beta_var_df = compute_beta_variability(reg_df)
    beta_cols = [c for c in beta_var_df.columns if c.startswith("beta_")]
    for _, row in beta_var_df.iterrows():
        print(f"  {row['method']:10s}  "
              f"n={row['n_scenarios']:.0f}  "
              f"mean(beta)={row['mean_beta']:+.4f}  "
              f"SD(beta)={row['sd_beta']:.4f}  "
              f"min={row['min_beta']:+.4f}  max={row['max_beta']:+.4f}  "
              f"range={row['range_beta']:.4f}  "
              f"CV={row['cv_beta']:.3f}")
        scenario_strs = [f"{c.replace('beta_', '')}={row[c]:+.3f}"
                         for c in beta_cols if not np.isnan(row[c])]
        print(f"             beta per scenario: {', '.join(scenario_strs)}")
    beta_var_df.to_csv(output_dir / "beta_variability.csv", index=False, encoding="utf-8")
    print(f"  -> {output_dir / 'beta_variability.csv'}")

    # --- Merged summary CSV per method ---
    print("\nBuilding per-method summary CSVs ...")
    for method in ["Manual", "Auto"]:
        summary_df = build_method_summary(reg_df, d_df, method)
        csv_path = output_dir / f"summary_{method.lower()}.csv"
        summary_df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"  -> {csv_path}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Done. Output directory: {output_dir.resolve()}")
    print(f"{'='*60}")
    for f in sorted(output_dir.glob("*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
