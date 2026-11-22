"""Does classifier guidance raise agreement with the guiding classifier, and at what cost?

For every (target family, guidance strength) cell this compares a guided sample set with the
unguided (lambda = 0) set made with the same generator checkpoint, temperature, length settings,
sample count, seed and device. It reuses the existing evaluator: validity, uniqueness and nearest
training identity are read from the `eval.csv` that `eval/eval.py` wrote for each sample CSV.
Only the family assignment is new, and it is made by the same frozen classifier that guides the
sampler, so it measures AGREEMENT WITH THE GUIDING CLASSIFIER, not independent family membership.

Inputs are a manifest of the runs (eval/guidance_runs.csv - sample CSVs carry no metadata, so
the settings are declared there) and the per-run eval.csv files. Outputs are one results CSV and
one figure. Runs that are absent, or whose settings differ from the baseline, are flagged in the
`status` column and never pooled. Nothing here reads data/test.csv.

    uv run python eval/guidance_agreement.py
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from protgen.data import FAMILIES, total_tokens
from protgen.metrics import read_sequences
from protgen.models import load_family_classifier

ROOT = Path(__file__).resolve().parent.parent
TOKEN_ID = {token: index for index, token in enumerate(total_tokens)}
PLANNED_LAMBDAS = (0.5, 1.0, 2.0, 4.0)

# Settings that must agree between a guided run and its baseline for the difference to mean
# anything. `num_samples` is compared on the declared value and on the rows actually present.
MATCH_FIELDS = ("generator_checkpoint", "temperature", "length_mode", "max_length", "seed",
                "num_samples", "device")
Z = 1.959964


def wilson(k, n):
    """95% Wilson interval for a binomial proportion, as (low, high) fractions."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    return centre - half, centre + half


def newcombe(k1, n1, k0, n0):
    """95% Newcombe (hybrid score) interval for p1 - p0 between two independent samples."""
    if n1 == 0 or n0 == 0:
        return float("nan"), float("nan")
    p1, p0 = k1 / n1, k0 / n0
    l1, u1 = wilson(k1, n1)
    l0, u0 = wilson(k0, n0)
    return (p1 - p0 - math.sqrt((p1 - l1) ** 2 + (u0 - p0) ** 2),
            p1 - p0 + math.sqrt((u1 - p1) ** 2 + (p0 - l0) ** 2))


def bootstrap_median_diff(guided, base, draws=2000, seed=0):
    """95% percentile-bootstrap interval for median(guided) - median(base)."""
    guided, base = np.asarray(guided, float), np.asarray(base, float)
    guided, base = guided[~np.isnan(guided)], base[~np.isnan(base)]
    if not len(guided) or not len(base):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    diffs = [np.median(rng.choice(guided, len(guided))) - np.median(rng.choice(base, len(base)))
             for _ in range(draws)]
    return tuple(np.percentile(diffs, [2.5, 97.5]))


@torch.inference_mode()
def classify(sequences, classifier_path):
    """Family probabilities for complete sequences, one row per sequence.

    Each sequence is scored as [BOS, residues...] at its last residue, the same position as the
    "clean (full sequence)" row of results/classifier/prefix_accuracy.csv and the position the
    sampler's guidance uses. The GRU is causal, so right padding cannot change that state.
    Returns (probabilities as an (n, 3) array in FAMILIES order, the checkpoint's fam_to_idx).
    """
    backbone, head, fam_to_idx = load_family_classifier(classifier_path, "cpu")
    out = np.zeros((len(sequences), len(FAMILIES)))
    for start in range(0, len(sequences), 128):
        chunk = sequences[start:start + 128]
        width = max(len(sequence) for sequence in chunk) + 1
        tokens = torch.zeros(len(chunk), width, dtype=torch.long)
        for row, sequence in enumerate(chunk):
            tokens[row, :len(sequence) + 1] = torch.tensor(
                [TOKEN_ID["<BOS>"]] + [TOKEN_ID[residue] for residue in sequence])
        states = backbone.hidden_states(tokens)
        last = states[torch.arange(len(chunk)), torch.tensor([len(s) for s in chunk])]
        probs = head(last).softmax(-1)
        for family in FAMILIES:
            out[start:start + len(chunk), FAMILIES.index(family)] = probs[:, fam_to_idx[family]]
    return out, fam_to_idx


def load_run(row, classifier_path, root):
    """One manifest row -> (per-sample table with the evaluator's columns + family assignment)."""
    samples = read_sequences(root / row["csv"])
    eval_path = root / "results/eval" / Path(row["csv"]).stem / "eval.csv"
    if not eval_path.is_file():
        raise SystemExit(f"{eval_path} is missing; score the samples first:\n"
                         f"  uv run python eval/eval.py --samples {row['csv']}")
    table = pd.read_csv(eval_path, skip_blank_lines=False)
    table["sequence"] = table["sequence"].fillna("").astype(str)
    if table["sequence"].tolist() != samples["sequence"].tolist():
        raise SystemExit(f"{eval_path} does not describe {row['csv']} (sequences differ); "
                         f"rerun eval/eval.py on it")
    table["valid"] = table["valid"].astype(bool)
    table["is_unique"] = table["is_unique"].astype(bool)

    table["pred_family"] = "unscored"
    valid = table["valid"].to_numpy()
    probs = np.full((len(table), len(FAMILIES)), np.nan)
    if valid.any():
        probs[valid], _ = classify(table.loc[valid, "sequence"].tolist(), classifier_path)
        table.loc[valid, "pred_family"] = [FAMILIES[i] for i in probs[valid].argmax(1)]
    for index, family in enumerate(FAMILIES):
        table[f"p_{family}"] = probs[:, index]
    return table


def shorten(path, root):
    try:
        return Path(path).resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path)


def csv_name(family, lam, seed=42):
    suffix = "" if seed == 42 else f"_seed{seed}"
    return f"results/sequences/gru_{family}_lambda{lam:g}{suffix}.csv"


def generate_command(family, lam, base, classifier):
    return (f"uv run python generate.py --checkpoint {base['generator_checkpoint']} "
            f"--classifier-checkpoint {classifier} --target-family {family} "
            f"--guidance-strength {lam:g} --num-samples {int(base['num_samples'])} "
            f"--temperature {base['temperature']:g} --seed {int(base['seed'])} "
            f"--device {base['device']} --out {csv_name(family, lam, int(base['seed']))}")


def mismatches(run, base, n_rows_run, n_rows_base):
    """Names of the settings on which a guided run differs from its baseline."""
    found = []
    for name in MATCH_FIELDS:
        a, b = run[name], base[name]
        if name == "max_length" and run["length_mode"] == "train" and base["length_mode"] == "train":
            continue
        if str(a) != str(b):
            found.append(f"{name} ({a} vs {b})")
    if n_rows_run != n_rows_base:
        found.append(f"rows ({n_rows_run} vs {n_rows_base})")
    return found


def summarise(table, family):
    """Counts and evaluator metrics of one run; `family` is the target (or None for a baseline)."""
    n = len(table)
    valid = table[table["valid"]]
    row = {
        "n_total": n,
        "n_valid": len(valid),
        "n_unscored": n - len(valid),
        "frac_valid": len(valid) / n,
        "frac_unique": table["is_unique"].mean(),
        "nearest_identity_median": valid["nearest_identity"].median(),
        "count_AcyP": int((table["pred_family"] == "AcyP").sum()),
        "count_HPr": int((table["pred_family"] == "HPr").sum()),
        "count_HU": int((table["pred_family"] == "HU").sum()),
    }
    if family:
        row["target_count"] = row[f"count_{family}"]
        row["mean_target_prob"] = valid[f"p_{family}"].mean()
    return row


def build(manifest, families, lambdas, classifier_path, root):
    """The results table and the list of planned runs that are missing."""
    runs = manifest.copy()
    runs["lambda"] = runs["lambda"].astype(float)
    runs["target_family"] = runs["target_family"].fillna("")
    for column in ("max_length", "classifier_checkpoint"):
        runs[column] = runs[column].fillna("")
    baselines = runs[runs["lambda"] == 0]
    if baselines.empty:
        raise SystemExit("The manifest has no lambda = 0 baseline run")

    guiding = {c for c in runs.loc[runs["lambda"] > 0, "classifier_checkpoint"] if c}
    if guiding - {shorten(classifier_path, root)}:
        raise SystemExit(f"guided runs used {sorted(guiding)} but the scorer is {classifier_path}; "
                         f"the same classifier must guide and score")
    tables = {row["run_id"]: load_run(row, classifier_path, root) for _, row in runs.iterrows()}
    for _, row in runs.iterrows():
        if len(tables[row["run_id"]]) != int(row["num_samples"]) and row["run_class"] == "full":
            raise SystemExit(f"{row['run_id']}: declared full with num_samples="
                             f"{row['num_samples']} but {len(tables[row['run_id']])} rows")

    out, missing = [], []
    seeds = sorted(runs["seed"].unique())
    for family in families:
        for _, base in baselines.iterrows():
            summary = summarise(tables[base["run_id"]], family)
            summary["_n"] = summary["n_total"]
            out.append({"target_family": family, "lambda": 0.0, "run_id": base["run_id"],
                        "csv": base["csv"], "seed": base["seed"], "run_class": base["run_class"],
                        "status": "baseline", **summary})
        for lam in lambdas:
            for seed in seeds:
                found = runs[(runs["target_family"] == family) & (runs["lambda"] == lam)
                             & (runs["seed"] == seed)]
                if found.empty:
                    if not any(b["seed"] == seed for _, b in baselines.iterrows()):
                        continue
                    base = baselines[baselines["seed"] == seed].iloc[0]
                    missing.append((family, lam, seed, base))
                    out.append({"target_family": family, "lambda": lam, "seed": seed,
                                "csv": csv_name(family, lam, seed), "status": "missing",
                                "run_class": "none"})
                    continue
                run = found.iloc[0]
                same_seed = baselines[baselines["seed"] == run["seed"]]
                summary = summarise(tables[run["run_id"]], family)
                entry = {"target_family": family, "lambda": lam, "run_id": run["run_id"],
                         "csv": run["csv"], "seed": run["seed"], "run_class": run["run_class"],
                         **summary}
                if same_seed.empty:
                    entry["status"] = "no_baseline_with_same_seed"
                    out.append(entry)
                    continue
                base = same_seed.iloc[0]
                base_table, run_table = tables[base["run_id"]], tables[run["run_id"]]
                base_summary = summarise(base_table, family)
                bad = mismatches(run, base, len(run_table), len(base_table))
                entry["status"] = "matched" if not bad else "MISMATCH: " + "; ".join(bad)
                entry["baseline_run_id"] = base["run_id"]
                entry["baseline_target_count"] = base_summary["target_count"]
                entry["baseline_n_total"] = base_summary["n_total"]
                k1, n1 = entry["target_count"], entry["n_total"]
                k0, n0 = base_summary["target_count"], base_summary["n_total"]
                low, high = newcombe(k1, n1, k0, n0)
                entry["delta_target_pp"] = 100 * (k1 / n1 - k0 / n0)
                entry["delta_target_pp_ci_low"] = 100 * low
                entry["delta_target_pp_ci_high"] = 100 * high
                entry["delta_frac_valid_pp"] = 100 * (summary["frac_valid"] - base_summary["frac_valid"])
                entry["delta_frac_unique_pp"] = 100 * (summary["frac_unique"] - base_summary["frac_unique"])
                entry["delta_identity_median"] = (summary["nearest_identity_median"]
                                                  - base_summary["nearest_identity_median"])
                low, high = bootstrap_median_diff(
                    run_table.loc[run_table["valid"], "nearest_identity"],
                    base_table.loc[base_table["valid"], "nearest_identity"])
                entry["delta_identity_median_ci_low"] = low
                entry["delta_identity_median_ci_high"] = high
                out.append(entry)

    frame = pd.DataFrame(out)
    frame["target_frac"] = frame["target_count"] / frame["n_total"]
    frame["target_frac_of_scored"] = frame["target_count"] / frame["n_valid"]
    ci = [wilson(k, n) if n == n else (np.nan, np.nan)
          for k, n in zip(frame["target_count"], frame["n_total"])]
    frame["target_frac_ci_low"] = [c[0] for c in ci]
    frame["target_frac_ci_high"] = [c[1] for c in ci]
    frame["baseline_target_frac"] = frame["baseline_target_count"] / frame["baseline_n_total"]
    # The baseline rows are filled with their own fractions so each family reads as one block.
    is_base = frame["status"] == "baseline"
    frame.loc[is_base, "baseline_target_count"] = frame.loc[is_base, "target_count"]
    frame.loc[is_base, "baseline_n_total"] = frame.loc[is_base, "n_total"]
    frame.loc[is_base, "baseline_target_frac"] = frame.loc[is_base, "target_frac"]
    frame = frame.drop(columns="_n")
    frame["scorer"] = "guiding classifier (agreement, not independent family membership)"
    lead = ["target_family", "lambda", "seed", "status", "run_class", "target_count", "n_total",
            "target_frac", "target_frac_ci_low", "target_frac_ci_high", "baseline_target_count",
            "baseline_n_total", "baseline_target_frac", "delta_target_pp",
            "delta_target_pp_ci_low", "delta_target_pp_ci_high", "n_valid", "n_unscored",
            "target_frac_of_scored", "mean_target_prob", "frac_valid", "frac_unique",
            "nearest_identity_median", "delta_frac_valid_pp", "delta_frac_unique_pp",
            "delta_identity_median", "delta_identity_median_ci_low",
            "delta_identity_median_ci_high"]
    rest = [c for c in frame.columns if c not in lead]
    frame = frame[lead + rest]
    order = {family: i for i, family in enumerate(families)}
    frame = frame.sort_values(["target_family", "lambda", "seed"],
                              key=lambda s: s.map(order) if s.name == "target_family" else s)
    return frame.reset_index(drop=True), missing


def plot(frame, path, lambdas):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = [f for f in FAMILIES if f in set(frame["target_family"])]
    fig, axes = plt.subplots(1, len(families), figsize=(4.3 * len(families), 4.2), sharey=True,
                             squeeze=False)
    any_smoke = (frame["run_class"] == "smoke").any()
    for ax, family in zip(axes[0], families):
        sub = frame[frame["target_family"] == family]
        for _, row in sub[sub["status"] == "baseline"].iterrows():
            ax.axhline(100 * row["target_frac"], color="0.5", ls="--", lw=1,
                       label=f"unguided baseline (seed {row['seed']}, {row['target_frac']:.1%})")
            ax.axhspan(100 * row["target_frac_ci_low"], 100 * row["target_frac_ci_high"],
                       color="0.5", alpha=0.12)
        ax.plot([0] * (sub["status"] == "baseline").sum(),
                100 * sub[sub["status"] == "baseline"]["target_frac"], "o", color="0.3")
        have = sub[~sub["status"].isin(["baseline", "missing"])]
        for smoke, marker, color, name in ((False, "o", "tab:blue", "guided"),
                                           (True, "X", "tab:red", "guided - SMOKE TEST (n too small)")):
            part = have[(have["run_class"] == "smoke") == smoke]
            if part.empty:
                continue
            ok = part["status"] == "matched"
            err = [100 * (part["target_frac"] - part["target_frac_ci_low"]),
                   100 * (part["target_frac_ci_high"] - part["target_frac"])]
            ax.errorbar(part["lambda"], 100 * part["target_frac"], yerr=err, fmt=marker,
                        color=color, capsize=3, label=name)
            for x, y, flag in zip(part["lambda"], 100 * part["target_frac"], ok):
                if not flag:
                    ax.annotate("settings differ", (x, y), textcoords="offset points",
                                xytext=(6, -12), fontsize=7, color="tab:red")
        for lam in lambdas:
            if not ((sub["lambda"] == lam) & (sub["status"] != "missing")).any():
                ax.plot([lam], [0], marker="x", color="0.7", ls="none")
        ax.set_xscale("symlog", linthresh=0.5)
        ax.set_xticks([0, *lambdas])
        ax.set_xticklabels(["0", *[f"{lam:g}" for lam in lambdas]])
        ax.set_title(f"target {family}")
        ax.set_xlabel("guidance strength λ")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    axes[0][0].set_ylabel("samples the guiding classifier assigns to target (%)")
    note = ("one point per seed, bars: 95% Wilson CI   |   grey x = run not available   |   "
            "agreement with the guiding classifier, not independent evidence of family")
    fig.text(0.5, 0.005, note, ha="center", fontsize=7)
    if any_smoke:
        fig.suptitle("SMOKE-TEST DATA ONLY - NOT A PROJECT FINDING", color="tab:red",
                     fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default=str(ROOT / "eval/guidance_runs.csv"),
                        help="manifest of the sample CSVs and the settings they were made with")
    parser.add_argument("--classifier", default=str(ROOT / "results/checkpoints/family_classifier.pt"))
    parser.add_argument("--lambdas", type=float, nargs="+", default=list(PLANNED_LAMBDAS),
                        help="the planned non-zero guidance strengths")
    parser.add_argument("--families", nargs="+", default=list(FAMILIES), choices=FAMILIES)
    parser.add_argument("--out-csv", default=str(ROOT / "results/guidance_eval/guidance_agreement.csv"))
    parser.add_argument("--out-fig", default=str(ROOT / "results/guidance_eval/guidance_agreement.png"))
    args = parser.parse_args()

    manifest = pd.read_csv(args.runs, dtype={"max_length": str})
    frame, missing = build(manifest, args.families, args.lambdas, Path(args.classifier), ROOT)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.round(5).to_csv(out_csv, index=False)
    plot(frame, Path(args.out_fig), args.lambdas)

    shown = frame[["target_family", "lambda", "seed", "run_class", "status", "target_count",
                   "n_total", "delta_target_pp", "frac_valid", "frac_unique",
                   "nearest_identity_median"]]
    print(shown.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nWrote {out_csv} and {args.out_fig}")
    if missing:
        print(f"\n{len(missing)} planned run(s) missing; to create them (same settings as the baseline):")
        for family, lam, _, base in missing:
            print("  " + generate_command(family, lam, base, shorten(args.classifier, ROOT)))


if __name__ == "__main__":
    main()
