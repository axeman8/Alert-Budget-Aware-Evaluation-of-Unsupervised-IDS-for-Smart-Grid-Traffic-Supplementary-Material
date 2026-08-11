#!/usr/bin/env python3
"""Regenerate the two manuscript figures from results/summary_budgets.csv."""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

ORDER = ["Dense AE", "LSTM AE", "Hybrid AE", "Isolation Forest", "One-Class SVM"]
MARKERS = ["o", "s", "^", "D", "v"]
MAIN_BUDGETS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="figures_regenerated")
    args = p.parse_args()
    results = Path(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(results / "summary_budgets.csv")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for model, marker in zip(ORDER, MARKERS):
        d = df[(df["model"] == model) & (df["target_fpr_budget_pct"].isin(MAIN_BUDGETS))].copy()
        d = d.sort_values("target_fpr_budget_pct")
        y = 100 * d["attack_f1_mean"]
        yerr = [
            100 * (d["attack_f1_mean"] - d["attack_f1_ci95_low"]),
            100 * (d["attack_f1_ci95_high"] - d["attack_f1_mean"]),
        ]
        ax.errorbar(d["target_fpr_budget_pct"], y, yerr=yerr, marker=marker,
                    linewidth=1.5, markersize=5, capsize=2.5, label=model)
    ax.set_xscale("log")
    ax.set_xlabel("Nominal validation FPR budget (%)")
    ax.set_ylabel("Attack F1 (%)")
    ax.set_xticks(MAIN_BUDGETS)
    ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10"])
    ax.grid(True, which="both", linewidth=0.5, alpha=0.35)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "multiseed_budget_f1.pdf", bbox_inches="tight")
    fig.savefig(out / "multiseed_budget_f1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for model, marker in zip(ORDER, MARKERS):
        d = df[(df["model"] == model) & (df["target_fpr_budget_pct"].isin(MAIN_BUDGETS))].copy()
        d = d.sort_values("target_fpr_budget_pct")
        ax.plot(d["target_fpr_budget_pct"], d["raw_false_triggers_per_day_mean"],
                marker=marker, linewidth=1.5, markersize=5, label=model)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Nominal validation FPR budget (%)")
    ax.set_ylabel("Raw false-positive windows per day")
    ax.set_xticks(MAIN_BUDGETS)
    ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10"])
    ax.grid(True, which="both", linewidth=0.5, alpha=0.35)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "raw_false_triggers_per_day.pdf", bbox_inches="tight")
    fig.savefig(out / "raw_false_triggers_per_day.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
