"""Shared matplotlib style for thesis figures (PDF + PNG)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# okabe–ito, colourblind-safe qualitative palette.
OKABE_ITO = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def savefig(fig, path: Path) -> None:
    """Save both PDF (vector, for the document) and PNG (for quick viewing)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.with_suffix('.pdf')} and .png")


def topic_axis_labels(top_words, interpreted=None, n_kw: int = 3) -> dict:
    """Map topic id -> 'n: kw/kw/kw' or 'n: interpreted label'."""
    labels = {}
    for t, words in top_words.items():
        kws = "/".join(str(words).split(", ")[:n_kw])
        if interpreted is not None and t in interpreted and interpreted[t]:
            labels[t] = f"{t}: {interpreted[t]}"
        else:
            labels[t] = f"{t}: {kws}"
    return labels
