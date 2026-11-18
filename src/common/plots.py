"""
plots.py
========
Grafik bersama untuk Tahap 7–9 (heatmap komunitas x kategori, bar bertumpuk sentimen).
Warna: satu hue biru untuk magnitudo (heatmap); tiga warna tetap untuk sentimen
(positif = biru, netral = abu-abu, negatif = merah) dengan legenda.
"""

from __future__ import annotations

from typing import List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK = "#1f1f1d"
INK_2 = "#3d3d3a"
SEQ_BLUES = ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SENTIMENT_COLORS = {"positif": "#2a78d6", "netral": "#b4b2a9", "negatif": "#d1453b"}


def community_order(ids: Sequence[str], small: str = "kecil", none: str = "tanpa_interaksi") -> List[str]:
    ids = set(ids)
    small_labels = {small, "small"}
    main = sorted(i for i in ids if i not in small_labels and i != none)
    return main + [i for i in (small, "small", none) if i in ids]


def heatmap(mat: pd.DataFrame, path, title: str, cbar_label: str, row_sizes: pd.Series | None = None,
            label_min: float = 10.0, fmt: str = "{:.0f}") -> None:
    """mat: baris = komunitas, kolom = kategori, nilai dalam persen."""
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUES)
    nr, nc = mat.shape
    fig, ax = plt.subplots(figsize=(1.6 + 0.62 * nc, 0.9 + 0.36 * nr))
    vals = mat.values.astype(float)
    vmax = max(10.0, float(np.nanpercentile(vals, 98))) if vals.size else 10.0
    im = ax.imshow(vals, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(range(nc), [str(c) for c in mat.columns], rotation=40, ha="right", fontsize=8.5, color=INK_2)
    ylab = [f"{r}  (n={int(row_sizes.get(r, 0))})" if row_sizes is not None else str(r) for r in mat.index]
    ax.set_yticks(range(nr), ylab, fontsize=8.5, color=INK_2)
    for i in range(nr):
        for j in range(nc):
            v = vals[i, j]
            if v >= label_min:
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7.5,
                        color="white" if v > vmax * 0.55 else INK)
    ax.set_xticks(np.arange(-0.5, nc, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nr, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label(cbar_label, fontsize=8, color=INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5, length=0)
    ax.set_title(title, fontsize=11, loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def stacked_sentiment(share: pd.DataFrame, path, title: str, row_sizes: pd.Series | None = None,
                      labels: Sequence[str] = ("positif", "netral", "negatif")) -> None:
    """share: baris = komunitas, kolom = label sentimen, nilai proporsi 0–1. Bar horizontal 100%."""
    share = share.reindex(columns=list(labels)).fillna(0)
    nr = len(share)
    fig, ax = plt.subplots(figsize=(8.2, 0.9 + 0.34 * nr))
    left = np.zeros(nr)
    y = np.arange(nr)
    for lab in labels:
        w = share[lab].values
        ax.barh(y, w, left=left, height=0.72, color=SENTIMENT_COLORS[lab], label=lab,
                edgecolor="white", linewidth=1.5)
        for yi, (l, wi) in enumerate(zip(left, w)):
            if wi >= 0.12:
                ax.text(l + wi / 2, yi, f"{wi:.0%}", ha="center", va="center", fontsize=7.5,
                        color="white" if lab != "netral" else INK)
        left += w
    ylab = [f"{r}  (n={int(row_sizes.get(r, 0))})" if row_sizes is not None else str(r) for r in share.index]
    ax.set_yticks(y, ylab, fontsize=8.5, color=INK_2)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.tick_params(axis="x", labelsize=8, colors=INK_2, length=0)
    ax.tick_params(axis="y", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=8.5)
    ax.set_title(title, fontsize=11, loc="left", color=INK, pad=24)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
