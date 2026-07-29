"""One figure comparing the two 'give the model the pitch it can't hear' fixes
against the audio-only baseline, per task, with each method's OWN paired baseline
and McNemar significance:

  audio only  |  + F0 numbers as TEXT (scalable)  |  + zoomed pitch IMAGE

cents (relative) is fixed by BOTH; absolute tuning only by the IMAGE (its
reference line) — the text lacks a reference. note_count is the negative control.

  PYTHONPATH=. python gpu/plot_trackde_comparison.py
"""
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from musicprobe.config import RESULTS_DIR
from musicprobe.image_jobs import IMAGE_JOBS_PATH
from musicprobe.scoring import parse_response, is_correct

JOBS = pd.read_parquet(IMAGE_JOBS_PATH)
TASKS = ["cents_discrimination", "tuning_judgment", "octave_id", "note_count"]


def load(tag):
    fr = []
    for f in glob.glob(str(RESULTS_DIR / f"responses__qwen25omni-{tag}-s*.parquet")):
        r = pd.read_parquet(f)[["job_id", "seed", "raw_response"]].merge(JOBS, on="job_id")
        r = r[r.raw_response.notna()].copy()
        r["correct"] = [is_correct(x.task, parse_response(x), x.ground_truth).__bool__()
                        for x in r.itertuples()]
        fr.append(r)
    return pd.concat(fr)


def paired(df, task):
    w = (df[(df.task == task) & df.image_condition.isin(["no_image", "image"])]
         .pivot_table(index=["stimulus_id", "seed"], columns="image_condition",
                      values="correct", aggfunc="first").dropna())
    A, B = w["no_image"].astype(bool).values, w["image"].astype(bool).values
    bo, co = int((B & ~A).sum()), int((A & ~B).sum())
    p = binomtest(min(bo, co), bo + co, 0.5).pvalue if bo + co else 1.0
    return A.mean(), B.mean(), p


def main():
    methods = [("f0text", "+ F0 numbers (text)", "#e8710a"),
               ("zoom", "+ zoomed pitch image", "#0a9648"),
               ("pitchfuse", "+ learned pitch stream (fused)", "#7d3ac1")]
    data = {tag: load(tag) for tag, _, _ in methods}
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(TASKS)); w = 0.2
    # audio-only baseline (avg of the two methods' no-image, they pair separately)
    base = [np.mean([paired(data[t], task)[0] for t, _, _ in methods]) for task in TASKS]
    ax.bar(x - w, base, w, label="audio only", color="#9aa0a6")
    for i, (tag, lab, col) in enumerate(methods):
        accs, ps = [], []
        for task in TASKS:
            a, b, p = paired(data[tag], task); accs.append(b); ps.append(p)
        ax.bar(x + i * w, accs, w, label=lab, color=col)
        for j, task in enumerate(TASKS):
            s = "***" if ps[j] < .001 else "**" if ps[j] < .01 else "*" if ps[j] < .05 else "ns"
            ax.text(x[j] + i * w, accs[j] + 0.02, s, ha="center", fontsize=9,
                    fontweight="bold", color=col if ps[j] < .05 else "#999")
    ax.axhline(0.5, ls=":", c="gray", alpha=.5)
    ax.set_xticks(x + w * 0.5); ax.set_xticklabels([t.replace("_", "\n") for t in TASKS])
    ax.set_ylabel("accuracy"); ax.set_ylim(0, 1.05)
    ax.set_title("Injecting the pitch the model can't hear: text numbers vs zoomed image vs fused stream\n"
                 "(Qwen2.5-Omni, 3 seeds, paired McNemar vs each method's own audio-only). "
                 "Reusing a pretrained pathway (text/chart) beats a learned fused feature.",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right"); ax.grid(alpha=.2, axis="y")
    out = RESULTS_DIR / "trackde_injection_comparison.png"
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
