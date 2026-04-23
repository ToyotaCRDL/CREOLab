#!/usr/bin/env python3
"""
NLP Metrics (BLEU / METEOR / CIDEr / BERTScore) Decoy Progression Analysis

Uses pycocoevalcap (MS COCO official evaluation tools) to compute
BLEU-4, METEOR (v1.5), and CIDEr-D, plus bert-score for BERTScore (F1),
between generated procedures and the gold-reference procedure.
Outputs a consolidated CSV of score statistics and linear regression
across decoy numbers (0-4) for Auto and Manual object detection methods.

Requirements:
    pip install pycocoevalcap bert-score
    Java 1.8+ (required by METEOR)
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import pandas as pd
from scipy import stats

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from bert_score import score as bert_score_fn

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

METHODS = ["auto_object_detection", "manual_object_detection"]
METHOD_LABELS = {
    "auto_object_detection": "Auto Object Detection",
    "manual_object_detection": "Manual Object Detection",
}

SCENARIO_RENAME = {
    "scenario01": "T1", "scenario02": "T2", "scenario03": "T3", "scenario04": "T4",
    "scenario05": "T5", "scenario06": "T6", "scenario07": "T7", "scenario08": "T8",
    "scenario09": "T9", "scenario10": "T10",
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def extract_gold_reference(eval_json_path: Path) -> str | None:
    """Extract the GOLD REFERENCE PROCEDURE from rubric evaluation JSON."""
    try:
        with open(eval_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for method_key in data["evaluations"]:
            eval_prompt = data["evaluations"][method_key]["evaluation_prompt"]
            match = re.search(
                r"GOLD REFERENCE PROCEDURE:\n(.*?)\n\nCANDIDATE PROCEDURE:",
                eval_prompt,
                re.DOTALL,
            )
            if match:
                gold = match.group(1).strip().replace("\\n", "\n")
                return gold
        return None
    except Exception as e:
        print(f"  [WARN] Could not extract gold reference from {eval_json_path}: {e}")
        return None


def load_generated_procedure(txt_path: Path) -> str | None:
    """Read a generated procedure txt file, stripping the header line."""
    try:
        text = txt_path.read_text(encoding="utf-8").strip()
        lines = text.splitlines()
        body_lines = [l for l in lines if not l.startswith("#")]
        return "\n".join(body_lines).strip()
    except Exception as e:
        print(f"  [WARN] Could not read {txt_path}: {e}")
        return None


# ---------------------------------------------------------------------------
# Data collection + pycocoevalcap scoring
# ---------------------------------------------------------------------------

def collect_all_data(base_path: Path):
    """
    Walk the experiment directory, collect text pairs, and score them
    using pycocoevalcap (BLEU-4, METEOR 1.5, CIDEr-D) and BERTScore.

    Returns
    -------
    tuple of (aggregated, per_item_records)
        aggregated : dict  –  {method: {decoy_number: {"bleu": [...], ...}}}
        per_item_records : list[dict]  – each dict has keys:
            method, case, decoy, bleu, meteor, cider, bertscore
    """
    scenario_dirs = sorted(
        [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("scenario")]
    )

    raw_pairs: list[dict] = []

    for scenario_dir in scenario_dirs:
        parts = scenario_dir.name.split("_")
        scenario_id = parts[0]
        decoy_number = int(parts[1].replace("decoy", ""))

        iter_dirs = sorted(
            [d for d in scenario_dir.iterdir() if d.is_dir() and d.name.startswith("iter_")]
        )

        for iter_dir in iter_dirs:
            eval_file = iter_dir / "evaluation" / "rubric_evaluation_results.json"
            gold = extract_gold_reference(eval_file) if eval_file.exists() else None
            if gold is None:
                continue

            for method in METHODS:
                proc_file = (
                    iter_dir / "integrations" / f"{method}_integrated_procedure.txt"
                )
                candidate = load_generated_procedure(proc_file) if proc_file.exists() else None
                if candidate is None:
                    continue

                raw_pairs.append({
                    "method": method,
                    "case": SCENARIO_RENAME.get(scenario_id, scenario_id),
                    "decoy": decoy_number,
                    "gold": gold,
                    "candidate": candidate,
                })

    if not raw_pairs:
        print("  [ERROR] No data pairs found.")
        return {m: {} for m in METHODS}

    print(f"  Found {len(raw_pairs)} text pairs across {len(scenario_dirs)} scenario directories.")

    # Build gts / res in standard pycocoevalcap format: {id: [{"caption": text}]}
    gts_raw: dict[int, list[dict]] = {}
    res_raw: dict[int, list[dict]] = {}
    for idx, pair in enumerate(raw_pairs):
        gts_raw[idx] = [{"caption": pair["gold"]}]
        res_raw[idx] = [{"caption": pair["candidate"]}]

    # Tokenize with standard PTBTokenizer
    print("  Tokenizing with PTBTokenizer ...")
    tokenizer = PTBTokenizer()
    gts = tokenizer.tokenize(gts_raw)
    res = tokenizer.tokenize(res_raw)

    # Run scorers on tokenized data
    print("  Computing BLEU-4 (pycocoevalcap) ...")
    bleu_scorer = Bleu(4)
    bleu_corpus, bleu_per_item = bleu_scorer.compute_score(gts, res)
    bleu4_scores = bleu_per_item[3]
    print(f"    Corpus BLEU-4: {bleu_corpus[3]:.4f}")

    print("  Computing METEOR 1.5 (pycocoevalcap, Java) ...")
    meteor_scorer = Meteor()
    meteor_corpus, meteor_per_item = meteor_scorer.compute_score(gts, res)
    print(f"    Corpus METEOR: {meteor_corpus:.4f}")

    print("  Computing CIDEr-D (pycocoevalcap) ...")
    cider_scorer = Cider()
    cider_corpus, cider_per_item = cider_scorer.compute_score(gts, res)
    print(f"    Corpus CIDEr-D: {cider_corpus:.4f}")

    print("  Computing BERTScore (roberta-large) ...")
    _, _, bertscore_f1_tensor = bert_score_fn(
        [p["candidate"] for p in raw_pairs],
        [p["gold"] for p in raw_pairs],
        lang="en",
        rescale_with_baseline=True,
        verbose=True,
    )
    bertscore_f1 = bertscore_f1_tensor.tolist()
    print(f"    Corpus BERTScore F1: {mean(bertscore_f1):.4f}")

    # Aggregate into {method: {decoy: {metric: [scores]}}}
    result: dict[str, dict[int, dict[str, list[float]]]] = {}
    for method in METHODS:
        result[method] = defaultdict(lambda: defaultdict(list))

    per_item_records: list[dict] = []

    for idx, pair in enumerate(raw_pairs):
        bucket = result[pair["method"]][pair["decoy"]]
        b4 = float(bleu4_scores[idx])
        met = float(meteor_per_item[idx])
        cid = float(cider_per_item[idx])
        bsc = float(bertscore_f1[idx])
        bucket["bleu"].append(b4)
        bucket["meteor"].append(met)
        bucket["cider"].append(cid)
        bucket["bertscore"].append(bsc)

        per_item_records.append({
            "method": pair["method"],
            "case": pair["case"],
            "decoy": pair["decoy"],
            "bleu": b4,
            "meteor": met,
            "cider": cid,
            "bertscore": bsc,
        })

    return result, per_item_records


METRIC_LABELS = {
    "bleu": "BLEU-4",
    "meteor": "METEOR",
    "cider": "CIDEr",
    "bertscore": "BERTScore",
}

METRIC_SPECS = [
    ("Checklist (OUR)",  "rubric",     1.0),
    ("BLEU-4 × 100",    "bleu",     100.0),
    ("METEOR × 100",    "meteor",   100.0),
    ("CIDEr",            "cider",     1.0),
    ("BERTScore × 100",  "bertscore", 100.0),
]

METRIC_FMT = {
    "rubric": ".1f",
    "bleu":   ".1f",
    "meteor": ".1f",
    "cider":  ".2f",
    "bertscore": ".1f",
}


# ---------------------------------------------------------------------------
# Rubric (checklist) score collection & consolidated table
# ---------------------------------------------------------------------------

def _flatten_by_scenario(
    by_case: dict[str, dict[int, list[float]]],
) -> dict[int, list[float]]:
    """Collapse per-scenario rubric data into {decoy_number: [scores]}."""
    flat: dict[int, list[float]] = defaultdict(list)
    for scenario_scores in by_case.values():
        for dn, vals in scenario_scores.items():
            flat[dn].extend(vals)
    return dict(flat)


def _cohens_d(group: list[float], baseline: list[float]) -> float | None:
    """Compute Cohen's d (group vs baseline)."""
    n1, n2 = len(baseline), len(group)
    if n1 < 2 or n2 < 2:
        return None
    m1, m2 = mean(baseline), mean(group)
    s1, s2 = stdev(baseline), stdev(group)
    pooled = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled == 0:
        return None
    return (m2 - m1) / pooled


def collect_rubric_scores_by_case(base_path: Path):
    """Collect rubric scores grouped by (method, scenario, decoy).

    Returns
    -------
    dict  –  {method_key: {scenario_label: {decoy_number: [scores]}}}
    """
    result: dict[str, dict[str, dict[int, list[float]]]] = {
        m: defaultdict(lambda: defaultdict(list)) for m in METHODS
    }
    scenario_dirs = sorted(
        [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("scenario")]
    )
    for scenario_dir in scenario_dirs:
        parts = scenario_dir.name.split("_")
        scenario_id = parts[0]
        scenario_label = SCENARIO_RENAME.get(scenario_id, scenario_id)
        decoy_number = int(parts[1].replace("decoy", ""))

        iter_dirs = sorted(
            [d for d in scenario_dir.iterdir() if d.is_dir() and d.name.startswith("iter_")]
        )
        for iter_dir in iter_dirs:
            eval_file = iter_dir / "evaluation" / "rubric_evaluation_results.json"
            if not eval_file.exists():
                continue
            try:
                with open(eval_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for method_key in METHODS:
                    if method_key in data.get("evaluations", {}):
                        score = data["evaluations"][method_key].get("final_score")
                        if score is not None:
                            result[method_key][scenario_label][decoy_number].append(float(score))
            except Exception:
                continue
    return result


def compute_beta_variability_per_metric(
    per_item_records: list[dict],
    rubric_data_by_case: dict[str, dict[int, list[float]]],
    method_key: str,
) -> pd.DataFrame:
    """Compute per-scenario regression slope (beta) for each metric and report SD.

    Parameters
    ----------
    per_item_records : list[dict]
        Per-item records returned by collect_all_data.
    rubric_data_by_case : dict
        Checklist scores in {scenario: {decoy: [scores]}} format.
    method_key : str
        "auto_object_detection" or "manual_object_detection"

    Returns
    -------
    pd.DataFrame  — columns: metric, n_scenarios, mean_beta, sd_beta, min_beta, max_beta, range_beta, cv_beta
                    + per-scenario β columns (T1, T2, ...)
    """
    items_df = pd.DataFrame([r for r in per_item_records if r["method"] == method_key])
    scenarios = sorted(items_df["case"].unique()) if not items_df.empty else []

    rows = []
    for label, key, scale in METRIC_SPECS:
        betas: dict[str, float] = {}

        for scenario in scenarios:
            if key == "rubric":
                case_scores = rubric_data_by_case.get(scenario, {})
                xs, ys = [], []
                for dn, vals in case_scores.items():
                    for v in vals:
                        xs.append(float(dn))
                        ys.append(float(v))
            else:
                sdf = items_df[items_df["case"] == scenario]
                xs = sdf["decoy"].values.astype(float).tolist()
                ys = (sdf[key].values.astype(float) * scale).tolist()

            if len(xs) < 3:
                continue
            slope = stats.linregress(xs, ys).slope
            betas[scenario] = slope

        if len(betas) < 2:
            continue

        vals = list(betas.values())
        mean_b = np.mean(vals)
        sd_b = np.std(vals, ddof=1)
        row = {
            "metric": label,
            "n_scenarios": len(betas),
            "mean_beta": mean_b,
            "sd_beta": sd_b,
            "min_beta": np.min(vals),
            "max_beta": np.max(vals),
            "range_beta": np.max(vals) - np.min(vals),
            "cv_beta": sd_b / abs(mean_b) if mean_b != 0 else np.nan,
        }
        for s in scenarios:
            row[s] = betas.get(s, np.nan)
        rows.append(row)

    return pd.DataFrame(rows)


def build_consolidated_df(
    nlp_data: dict[int, dict[str, list[float]]],
    rubric_data: dict[int, list[float]],
) -> pd.DataFrame:
    """Build a DataFrame: one row per metric, columns for per-decoy means, β, R², p."""
    decoy_numbers = sorted(set(nlp_data.keys()) | set(rubric_data.keys()))

    rows = []
    for label, key, scale in METRIC_SPECS:
        fmt = METRIC_FMT[key]
        row: dict[str, str] = {"Metric": label}

        all_scores_for_regression: list[tuple[int, float]] = []

        if key == "rubric":
            score_by_decoy = rubric_data
        else:
            score_by_decoy = {
                dn: [v * scale for v in nlp_data[dn].get(key, [])]
                for dn in decoy_numbers
                if dn in nlp_data
            }

        baseline = score_by_decoy.get(0, [])

        for dn in decoy_numbers:
            vals = score_by_decoy.get(dn, [])
            if not vals:
                row[f"d={dn}"] = "-"
                continue

            m = mean(vals)
            for v in vals:
                all_scores_for_regression.append((dn, v))

            if dn == 0:
                row[f"d={dn}"] = f"{m:{fmt}}"
            else:
                d_val = _cohens_d(vals, baseline) if baseline else None
                if d_val is not None:
                    row[f"d={dn}"] = f"{m:{fmt}}\n({d_val:+.2f})"
                else:
                    row[f"d={dn}"] = f"{m:{fmt}}"

        if len(all_scores_for_regression) >= 3:
            xs = np.array([t[0] for t in all_scores_for_regression])
            ys = np.array([t[1] for t in all_scores_for_regression])
            res = stats.linregress(xs, ys)
            slope, intercept, r_value = res.slope, res.intercept, res.rvalue
            n_reg = len(xs)
            t_val = slope / (np.std(ys - (slope * xs + intercept), ddof=1)
                             / np.sqrt(np.sum((xs - xs.mean()) ** 2)))
            p_value = stats.t.cdf(t_val, n_reg - 2)  # one-sided: H0: β≥0, H1: β<0
            row["beta"] = f"{slope:+.2f}"
            row["R2"] = f"{r_value**2:.3f}"
            row["p_raw"] = p_value
            if p_value < 0.001:
                row["p"] = "<.001"
            elif p_value < 0.01:
                row["p"] = "<.01"
            elif p_value < 0.05:
                row["p"] = "<.05"
            else:
                row["p"] = f"{p_value:.3f}"
        else:
            row["beta"] = "-"
            row["R2"] = "-"
            row["p_raw"] = None
            row["p"] = "-"

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    default_path = _find_latest_batch()
    parser = argparse.ArgumentParser(
        description="NLP Metrics Decoy Progression Analysis")
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
    output_dir = base_path / "cross_analysis" / "by_method_decoy_progression_nlp_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("NLP Metrics Decoy Progression Analysis")
    print("  (pycocoevalcap: BLEU-4, METEOR 1.5, CIDEr-D + BERTScore)")
    print("=" * 60)
    print(f"Input:  {base_path.resolve()}")
    print(f"Output: {output_dir.resolve()}")

    print("\n[1/3] Collecting NLP metric data ...")
    all_data, per_item_records = collect_all_data(base_path)

    print("\n[2/3] Collecting rubric (checklist) scores ...")
    rubric_by_case = collect_rubric_scores_by_case(base_path)
    rubric_data = {m: _flatten_by_scenario(rubric_by_case[m]) for m in METHODS}

    print("\n[3/3] Building consolidated CSV per method ...")
    for method_key in METHODS:
        label = METHOD_LABELS[method_key]

        con_df = build_consolidated_df(all_data[method_key], rubric_data[method_key])
        for col in con_df.columns:
            con_df[col] = con_df[col].astype(str).str.replace("\n", " ", regex=False)

        beta_df = compute_beta_variability_per_metric(
            per_item_records, rubric_by_case[method_key], method_key,
        )

        if not beta_df.empty:
            beta_df = beta_df.rename(columns={"metric": "Metric"})
            merged = con_df.merge(beta_df, on="Metric", how="left")
        else:
            merged = con_df

        csv_path = output_dir / f"metrics_summary_{method_key}.csv"
        merged.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"  Saved: {csv_path}")

        print(f"\n  === {label}: Cross-scenario beta variability ===")
        if beta_df.empty:
            print("    (insufficient data)")
        else:
            for _, row in beta_df.iterrows():
                print(f"    {row['Metric']:20s}  "
                      f"n={row['n_scenarios']:.0f}  "
                      f"mean(beta)={row['mean_beta']:+.4f}  "
                      f"SD(beta)={row['sd_beta']:.4f}  "
                      f"min={row['min_beta']:+.4f}  max={row['max_beta']:+.4f}  "
                      f"range={row['range_beta']:.4f}  "
                      f"CV={row['cv_beta']:.3f}")

    print(f"\nOutput directory: {output_dir.resolve()}")
    print("Done.")


if __name__ == "__main__":
    main()
