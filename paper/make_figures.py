"""Paper 2 figures, generated from reports/paper2_numbers.json (run scripts/paper2_numbers.py first)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = Path(__file__).resolve().parent / "figures"
N = json.loads((ROOT / "reports/paper2_numbers.json").read_text(encoding="utf-8"))
ORDER = ["Coco", "Voc", "AdeS", "AdeX", "AdeM", "City", "Kitti", "Amos"]
LABEL = {"Coco": "COCO", "Voc": "VOC", "AdeS": "ADE-S", "AdeX": "ADE-X", "AdeM": "ADE-M", "City": "Cityscapes", "Kitti": "KITTI", "Amos": "AMOS-CT"}
COLOR = {"general": "#2a9d8f", "driving": "#e76f51", "medical": "#8a5a9e"}
DOMAIN = {t: "general" for t in ORDER[:5]} | {"City": "driving", "Kitti": "driving", "Amos": "medical"}


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


def composition_and_gap():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
    x = np.arange(len(ORDER))
    severe = np.array([N[f"SevOnly{t}"] for t in ORDER])
    both = np.array([N[f"Both{t}"] for t in ORDER])
    high = np.array([N[f"HighOnly{t}"] for t in ORDER])
    axes[0].bar(x, severe, color="#264653", label="severe miss only")
    axes[0].bar(x, both, bottom=severe, color="#9aa5ab", label="both")
    axes[0].bar(x, high, bottom=severe + both, color="#f4a261", label="high error only")
    axes[0].set_xticks(x, [LABEL[t] for t in ORDER], rotation=40, ha="right", fontsize=8)
    axes[0].set_ylabel("Records (%)")
    axes[0].set_title("Failure composition", fontsize=9)
    axes[0].legend(frameon=False, fontsize=7)
    est = np.array([N[f"Mvb{t}"] for t in ORDER])
    lo = np.array([N[f"MvbLo{t}"] for t in ORDER])
    hi = np.array([N[f"MvbHi{t}"] for t in ORDER])
    share = np.array([N[f"HighShare{t}"] for t in ORDER])
    for i, t in enumerate(ORDER):
        axes[1].errorbar(share[i], est[i], yerr=[[est[i] - lo[i]], [hi[i] - est[i]]], fmt="o", color=COLOR[DOMAIN[t]], capsize=3)
        axes[1].annotate(LABEL[t], (share[i], est[i]), textcoords="offset points", xytext=(4, 3), fontsize=7)
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[1].set_xlabel("High-error share of failures (%)")
    axes[1].set_ylabel("Mixed $-$ best single (AUROC)")
    axes[1].set_title("When the mixed score helps", fontsize=9)
    save(fig, "composition_gap.pdf")


def weight_curves():
    fig, ax = plt.subplots(figsize=(3.4, 2.5), constrained_layout=True)
    for t in ORDER:
        curve = N[f"curve_{t}"]
        w = np.array([float(k) for k in curve])
        v = np.array(list(curve.values()))
        ax.plot(w, v - v.max(), color=COLOR[DOMAIN[t]], lw=1.4, alpha=0.9, label=LABEL[t] if t in ("Coco", "City", "Amos") else None)
    ax.set_xlabel("Weight on component score $w$ (rest on $q_{fp}$)")
    ax.set_ylabel("AUROC minus panel maximum")
    handles = [plt.Line2D([], [], color=c, lw=1.6) for c in COLOR.values()]
    ax.legend(handles, ["general scenes", "driving", "medical CT"], frameon=False, fontsize=7, loc="lower center")
    save(fig, "weight_curves.pdf")


def gains_forest():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True, constrained_layout=True)
    y = np.arange(len(ORDER))
    for ax, (key, title) in zip(axes, (("Pol", "Policy rank"), ("Op", "Operational rank"), ("Sev", "Severe-miss AUROC"))):
        est = np.array([N[f"G{key}{t}"] for t in ORDER])
        lo = np.array([N[f"GLo{key}{t}"] for t in ORDER])
        hi = np.array([N[f"GHi{key}{t}"] for t in ORDER])
        for i, t in enumerate(ORDER):
            ax.errorbar(est[i], y[i], xerr=[[est[i] - lo[i]], [hi[i] - est[i]]], fmt="o", color=COLOR[DOMAIN[t]], capsize=2)
        ax.axvline(0, color="grey", lw=0.8)
        ax.set_title(title, fontsize=9)
    axes[0].set_yticks(y, [LABEL[t] for t in ORDER], fontsize=8)
    axes[0].invert_yaxis()
    save(fig, "gains_forest.pdf")


if __name__ == "__main__":
    composition_and_gap()
    weight_curves()
    gains_forest()
    print("figures written to", FIG)
