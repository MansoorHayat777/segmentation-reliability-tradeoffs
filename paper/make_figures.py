"""Generate manuscript figures exclusively from locked experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGURES = Path(__file__).resolve().parent / "figures"
PANEL_ORDER = [
    "COCO-val2017",
    "PASCAL-VOC-2012-val",
    "ADE20K-SegFormer-panel",
    "ADE20K-architecture-holdout",
    "Cityscapes-structured-driving",
]
PANEL_LABELS = ["COCO", "VOC", "ADE-S", "ADE-X", "City"]


def load_metrics(run: str) -> dict:
    return json.loads((ROOT / "outputs" / run / "metrics.json").read_text(encoding="utf-8"))


def save(figure: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURES / name, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(figure)


def conflict_and_routing() -> None:
    conflict = load_metrics("E021_conflict_sensitivity")["panels"]
    city = load_metrics("E023_cityscapes_confirmation")["criterion_conflict"][
        "scalar_unavoidable_pairwise_disagreement_lower_bound"
    ]
    routing_metrics = load_metrics("E028_routing_meta_analysis")
    routing = routing_metrics["synthesis"]
    panel_results = routing_metrics["panel_results"]
    bounds, low, high = [], [], []
    for panel in PANEL_ORDER:
        value = city if panel == "Cityscapes-structured-driving" else conflict[panel]["full_bound"]
        bounds.append(value["estimate"])
        low.append(value["ci_low"])
        high.append(value["ci_high"])
    errors = np.asarray([np.asarray(bounds) - low, np.asarray(high) - bounds])

    endpoints = [
        "policy_spearman",
        "operational_spearman",
        "severe_miss_auroc",
        "false_positive_auroc",
    ]
    endpoint_labels = ["Policy rank", "Operational rank", "Severe miss", "False positive"]
    effects = [routing[value]["equal_panel_mean"] for value in endpoints]
    panel_values = {
        endpoint: [panel_results[panel][endpoint]["estimate"] for panel in PANEL_ORDER]
        for endpoint in endpoints
    }
    effect_low = [min(panel_values[value]) for value in endpoints]
    effect_high = [max(panel_values[value]) for value in endpoints]
    effect_errors = np.asarray(
        [np.asarray(effects) - effect_low, np.asarray(effect_high) - effects]
    )

    figure, axes = plt.subplots(1, 2, figsize=(7.25, 2.8), constrained_layout=True)
    axes[0].bar(PANEL_LABELS, bounds, yerr=errors, capsize=2, color="#5B78A6")
    axes[0].set_ylim(0, 0.21)
    axes[0].set_ylabel("Unavoidable disagreement")
    axes[0].set_title("Empirical scalar conflict bound")
    axes[1].bar(
        endpoint_labels,
        effects,
        yerr=effect_errors,
        capsize=2,
        color=["#D37245", "#D37245", "#D37245", "#7A9E59"],
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Endpoint score $-$ mixed score")
    axes[1].set_title("Mean (dot) and observed panel range")
    axes[1].tick_params(axis="x", rotation=25)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    save(figure, "conflict_routing.pdf")


def threshold_sensitivity() -> None:
    metrics = load_metrics("E027_failure_threshold_sensitivity")
    thresholds = metrics["threshold_grid"]
    lookup = {
        (row["panel"], row["threshold"]): row.get("delta_to_best_single", np.nan)
        for row in metrics["records"]
    }
    matrix = np.asarray(
        [[lookup[(panel, threshold)] for threshold in thresholds] for panel in PANEL_ORDER]
    )
    figure, axis = plt.subplots(figsize=(7.1, 2.55), constrained_layout=True)
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-0.13, vmax=0.13, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:+.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(matrix[row, column]) > 0.065 else "black",
            )
    axis.set_xticks(range(len(thresholds)), [f"{value:.2f}" for value in thresholds])
    axis.set_yticks(range(len(PANEL_LABELS)), PANEL_LABELS)
    axis.set_xlabel("High false-positive threshold")
    axis.set_title("Mixed-failure AUROC delta to the strongest single metric")
    figure.colorbar(image, ax=axis, label="AUROC delta", shrink=0.9)
    save(figure, "threshold_sensitivity.pdf")


def cityscapes_mechanism() -> None:
    mechanism = load_metrics("E024_structured_domain_mechanism")["panel_results"][
        "Cityscapes-structured-driving"
    ]["failure_modes"]["category_counts"]
    safety = load_metrics("E025_cityscapes_safety_audit")["class_results"]
    categories = ["neither", "high_fp_only", "both", "severe_only"]
    category_labels = ["Neither", "High-FP only", "Both", "Severe only"]
    class_names = ["person", "rider", "car", "truck", "bus", "train", "bicycle"]
    estimates = [
        safety[name]["comparisons"]["severity_arithmetic_vs_mixed"].get("estimate", np.nan)
        for name in class_names
    ]
    low = [
        safety[name]["comparisons"]["severity_arithmetic_vs_mixed"].get("ci_low", np.nan)
        for name in class_names
    ]
    high = [
        safety[name]["comparisons"]["severity_arithmetic_vs_mixed"].get("ci_high", np.nan)
        for name in class_names
    ]
    errors = np.asarray([np.asarray(estimates) - low, np.asarray(high) - estimates])

    figure, axes = plt.subplots(1, 2, figsize=(7.25, 2.75), constrained_layout=True)
    axes[0].bar(category_labels, [mechanism[value] for value in categories], color="#5B78A6")
    axes[0].set_yscale("symlog", linthresh=1)
    axes[0].set_ylabel("Model--image cases (symlog)")
    axes[0].set_title("Cityscapes failure composition")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(
        [value.replace("_", " ") for value in class_names],
        estimates,
        yerr=errors,
        capsize=2,
        color="#D37245",
    )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Severe-miss AUROC delta")
    axes[1].set_title("Severity arithmetic $-$ mixed score")
    axes[1].tick_params(axis="x", rotation=28)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    save(figure, "cityscapes_mechanism.pdf")


def routing_forest() -> None:
    """Show paired image-cluster effects panel by panel, without meta-pooling."""
    data = load_metrics("E028_routing_meta_analysis")["panel_results"]
    endpoints = [
        ("policy_spearman", "Policy rank"),
        ("operational_spearman", "Operational rank"),
        ("severe_miss_auroc", "Severe-miss AUROC"),
        ("false_positive_auroc", "False-positive AUROC"),
    ]
    colors = ["#35618D", "#538A9B", "#7C9D66", "#D37245", "#9674A6"]
    figure, axis = plt.subplots(figsize=(7.2, 4.1), constrained_layout=True)
    centers = np.arange(len(endpoints))[::-1] * 1.25
    offsets = np.linspace(-0.28, 0.28, len(PANEL_ORDER))
    all_lows, all_highs = [], []
    for panel_index, panel in enumerate(PANEL_ORDER):
        estimates, lows, highs, ys = [], [], [], []
        for endpoint_index, (endpoint, _) in enumerate(endpoints):
            item = data[panel][endpoint]
            estimates.append(item["estimate"])
            lows.append(item["ci_low"])
            highs.append(item["ci_high"])
            ys.append(centers[endpoint_index] + offsets[panel_index])
        all_lows.extend(lows)
        all_highs.extend(highs)
        errors = np.asarray([np.asarray(estimates) - lows, np.asarray(highs) - estimates])
        axis.errorbar(estimates, ys, xerr=errors, fmt="o", ms=3.4,
                      capsize=2, lw=0.9, color=colors[panel_index],
                      label=PANEL_LABELS[panel_index])
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_yticks(centers, [label for _, label in endpoints])
    axis.set_xlabel("Endpoint-specific score minus mixed score")
    axis.set_title("Paired image-cluster effects by evaluation panel")
    observed_low = min(all_lows)
    observed_high = max(all_highs)
    margin = 0.04 * (observed_high - observed_low)
    axis.set_xlim(min(-0.03, observed_low - margin), observed_high + margin)
    axis.grid(axis="x", alpha=0.25)
    axis.legend(ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.31),
                frameon=False, fontsize=7.2)
    save(figure, "routing_forest.pdf")


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    conflict_and_routing()
    threshold_sensitivity()
    cityscapes_mechanism()
    routing_forest()
    print(f"Wrote manuscript figures to {FIGURES}")


if __name__ == "__main__":
    main()
