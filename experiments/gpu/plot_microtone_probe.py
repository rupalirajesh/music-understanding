"""Plain-language figure for the microtone probe (gpu/probe_microtone.py).

Two questions, two panels, the answer written on each:

  Q1 "Can it tell WHICH WAY the pitch moved?"  -> yes, even for tiny changes.
      x = size of the pitch change in cents (100c = one piano key)
      y = how often the model's audio encoder got the direction right (50% = guessing)
  Q2 "Can it tell HOW OUT-OF-TUNE one note is?" -> no, no better than ignoring it.
      bar = average error in cents; red line = the score you'd get by always
      guessing "in tune" (i.e. a rep that snaps every note to the piano grid).

  python gpu/plot_microtone_probe.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROBES = Path("results/trackB/probes")
DELTAS = [5, 10, 25, 50, 100]
# friendly names + a highlight for the standout model
NAME = {"whisper": "Whisper", "mert330": "MERT (music)", "clap": "CLAP",
        "qwen25omni_own": "Qwen2.5-Omni", "qwen3omni_own": "Qwen3-Omni",
        "af3_own": "Audio-Flamingo-3", "musicflamingo_own": "Music-Flamingo"}
HERO = "qwen3omni_own"


def main():
    c = pd.read_csv(PROBES / "probe_microtone__cents.csv")
    t = pd.read_csv(PROBES / "probe_microtone__tuning.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2))

    # ---- Q1: direction of a pitch change (best layer per model) ----
    for enc, g in c.groupby("encoder"):
        best = g.loc[g.bal_acc.idxmax()]
        ys = [best[f"acc_{d}c"] * 100 for d in DELTAS]
        hero = enc == HERO
        ax1.plot(DELTAS, ys, "-o", ms=7 if hero else 4,
                 lw=3.5 if hero else 1.4,
                 color="#0a9648" if hero else "#9aa0a6",
                 zorder=5 if hero else 2,
                 label=NAME.get(enc, enc) if hero else None)
    ax1.axhline(50, ls="--", c="#c0392b", lw=1.5)
    ax1.text(5.3, 51.5, "just guessing (50%)", color="#c0392b", fontsize=10)
    ax1.axvspan(5, 10, color="#0a9648", alpha=.08)
    ax1.text(6.8, 34, "smaller than a\nhuman can hear", ha="center",
             fontsize=9, color="#0a7d3a")
    ax1.annotate("✓ YES — the models hear the\ndirection, even for tiny changes",
                 xy=(100, 96), xytext=(18, 88), fontsize=13, fontweight="bold",
                 color="#0a5c2e")
    ax1.plot([], [], "-", color="#9aa0a6", lw=1.4, label="other 6 models")
    ax1.set_xscale("log"); ax1.set_xticks(DELTAS); ax1.set_xticklabels(DELTAS)
    ax1.set_xlabel("how big the pitch change is  (cents — 100 = one piano key)", fontsize=11)
    ax1.set_ylabel("how often it got the direction right  (%)", fontsize=11)
    ax1.set_title("Q1.  Can it tell WHICH WAY the pitch moved?", fontsize=13, fontweight="bold")
    ax1.set_ylim(30, 102); ax1.legend(fontsize=10, loc="lower right"); ax1.grid(alpha=.25)

    # ---- Q2: absolute detune, error in cents vs the snap baseline ----
    best_t = t.loc[t.groupby("encoder").mae_cents.idxmin()].copy()
    best_t["nm"] = best_t.encoder.map(NAME)
    best_t = best_t.sort_values("mae_cents")
    x = np.arange(len(best_t))
    snap = t.snap_mae_cents.iloc[0]
    ax2.bar(x, best_t.mae_cents, color="#9aa0a6", zorder=3)
    ax2.axhline(snap, ls="--", c="#c0392b", lw=2)
    ax2.text(len(x) - 0.5, snap + 0.3,
             "score you get by IGNORING the detuning\n(assuming every note is in tune)",
             ha="right", color="#c0392b", fontsize=10)
    ax2.annotate("✗ NO — every model is as bad as\nnot listening to the tuning at all",
                 xy=(0, 5), xytext=(-0.3, 4), fontsize=13, fontweight="bold",
                 color="#8e1b0f")
    ax2.set_xticks(x); ax2.set_xticklabels(best_t.nm, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("average tuning error  (cents — lower = better)", fontsize=11)
    ax2.set_title("Q2.  Can it tell HOW OUT-OF-TUNE one note is?", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, max(20, snap + 5)); ax2.grid(alpha=.25, axis="y")

    fig.suptitle("Can these AI models actually hear microtones?",
                 fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = PROBES / "microtone_probe_graph.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
